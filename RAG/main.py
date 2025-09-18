import os
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback

from config import RAGConfig
from translation import MultilingualHandler as TechnicalTranslator

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_technical_documents(data_folder: str) -> List:
    """Lädt alle PDF-Dokumente aus dem data Ordner."""
    documents = []
    data_path = Path(data_folder)
    
    if not data_path.exists():
        logger.error(f"Data folder {data_folder} does not exist!")
        return documents
    
    pdf_files = list(data_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"Loading: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            pdf_documents = loader.load()
            
            # Metadaten hinzufügen
            for doc in pdf_documents:
                doc.metadata['source'] = pdf_file.name
                doc.metadata['file_path'] = str(pdf_file)
                doc.metadata['extraction_method'] = 'PyPDF'
            
            documents.extend(pdf_documents)
            logger.info(f"Loaded {len(pdf_documents)} pages from {pdf_file.name}")
            
        except Exception as e:
            logger.error(f"Error loading {pdf_file}: {e}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

def create_text_chunks(documents: List, config: RAGConfig) -> List:
    """Erstellt Text-Chunks aus den Dokumenten."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Chunk-IDs hinzufügen
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i + 1
    
    logger.info(f"Created {len(chunks)} text chunks")
    return chunks

def create_new_vector_store(config, embeddings, max_chunks, vector_store_path):
    """Erstellt eine neue Vector Database."""
    logger.info("Creating new vector store...")
    
    # Dokumente laden
    documents = load_technical_documents("./data")
    
    if not documents:
        raise ValueError("Keine Dokumente gefunden!")
    
    logger.info(f"Technische Dokumente geladen: {len(documents)}")
    
    # Text-Chunks erstellen
    chunks = create_text_chunks(documents, config)
    
    if max_chunks and len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        logger.info(f"Limited to {max_chunks} chunks")
    
    # Vector Store erstellen - NEUER CODE FÜR CHROMA V0.4+
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=vector_store_path
        )
        logger.info(f"Vector store created with {len(chunks)} chunks")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise

def load_existing_vector_store(config, embeddings, vector_store_path):
    """Lädt eine existierende Vector Database."""
    try:
        logger.info(f"Loading existing vector store from {vector_store_path}")
        
        vectorstore = Chroma(
            persist_directory=vector_store_path,
            embedding_function=embeddings
        )
        
        # Teste ob Store funktioniert
        collection = vectorstore._collection
        count = collection.count()
        logger.info(f"Loaded vector store with {count} chunks")
        
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        raise

def initialize_technical_rag_system(language: str = "de", max_chunks: int = None) -> Tuple:
    """Initialisiert das technische RAG-System."""
    try:
        # Konfiguration laden
        config = RAGConfig()
        
        # Embeddings initialisieren
        logger.info("Initializing embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Vector Store Pfad
        vector_store_path = config.vector_store_path
        
        # Prüfe ob Vector Store existiert
        if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
            try:
                vectorstore = load_existing_vector_store(config, embeddings, vector_store_path)
            except Exception as e:
                logger.warning(f"Could not load existing vector store: {e}")
                logger.info("Creating new vector store...")
                vectorstore = create_new_vector_store(config, embeddings, max_chunks, vector_store_path)
        else:
            vectorstore = create_new_vector_store(config, embeddings, max_chunks, vector_store_path)
        
        # LLM initialisieren
        logger.info("Initializing LLM...")
        llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            openai_api_key=config.openai_api_key
        )
        
        # Prompt Template
        template = """Beantworte die Frage nur mit den bereitgestellten Passagen.
                        Wenn du es nicht sicher weißt, sage 'Unklar'.
                        Hänge am Ende eine Quellenliste im Format [Nr] Titel, S. Seite an.

                        Fokussiere dich auf:
                        - Technische Spezifikationen
                        - Hardware-Details  
                        - Anschlüsse und Ports
                        - Performance-Daten
                        - Kompatibilität

                        Frage: {question}
                        Passagen:
                        {context}
                        Antwort:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # QA Chain erstellen
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        # Translator initialisieren
        translator = TechnicalTranslator()
        
        logger.info("Technical RAG system initialized successfully!")
        return qa_chain, translator
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        raise

def ask_technical_question(qa_chain, question: str, translator, translate_query: bool = False, target_language: str = "de") -> Dict[str, Any]:
    """Stellt eine technische Frage an das RAG-System."""
    start_time = time.time()
    
    try:
        # Optional: Frage übersetzen
        if translate_query and target_language != "en":
            original_question = question
            question = translator.translate_text(question, target_lang="en")
            logger.info(f"Translated question: {original_question} -> {question}")
        
        # Frage an QA Chain
        with get_openai_callback() as cb:
            result = qa_chain.invoke({"query": question})
        
        # Response Zeit berechnen
        response_time = time.time() - start_time
        
        # Source Documents verarbeiten
        sources = []
        if result.get("source_documents"):
            for doc in result["source_documents"]:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": getattr(doc, 'relevance_score', 0.5)
                })
        
        # Schlüsselbegriffe extrahieren
        key_terms = extract_key_terms(question, result["result"])
        
        # Optional: Antwort übersetzen
        answer = result["result"]
        if translate_query and target_language != "en":
            answer = translator.translate_text(answer, target_lang=target_language)
        
        return {
            "answer": answer,
            "sources": sources,
            "source_count": len(sources),
            "response_time": response_time,
            "key_terms": key_terms,
            "tokens_used": cb.total_tokens if cb else 0,
            "cost": cb.total_cost if cb else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return {
            "answer": f"Fehler bei der Verarbeitung: {str(e)}",
            "sources": [],
            "source_count": 0,
            "response_time": time.time() - start_time,
            "key_terms": [],
            "tokens_used": 0,
            "cost": 0.0
        }

def extract_key_terms(question: str, answer: str) -> List[str]:
    """Extrahiert Schlüsselbegriffe aus Frage und Antwort."""
    import re
    
    # Technische Begriffe Pattern
    patterns = [
        r'\b\d+\s*GB\b',
        r'\b\d+\s*MHz\b',
        r'\b\d+\s*GHz\b',
        r'\bUSB[\s-]?[0-9C]\b',
        r'\bHDMI\b',
        r'\bCPU\b',
        r'\bRAM\b',
        r'\bSSD\b',
        r'\bHDD\b',
        r'\bGPU\b',
        r'\bIntel\s+\w+\b',
        r'\bAMD\s+\w+\b',
        r'\bNVIDIA\s+\w+\b'
    ]
    
    key_terms = []
    text = f"{question} {answer}".lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        key_terms.extend(matches)
    
    # Entferne Duplikate und sortiere
    return sorted(list(set(key_terms)))

if __name__ == "__main__":
    # Test der Initialisierung
    try:
        qa_chain, translator = initialize_technical_rag_system()
        print("✅ System successfully initialized!")
        
        # Test-Frage
        result = ask_technical_question(
            qa_chain, 
            "Welche CPU-Modelle sind verfügbar?", 
            translator
        )
        print(f"✅ Test question answered: {len(result['answer'])} characters")
        
    except Exception as e:
        print(f"❌ Error: {e}")
