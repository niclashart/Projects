import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import logging
from dotenv import load_dotenv
import time

# Lokale Imports
from config import RAGConfig
from translation import MultilingualHandler
from preprocessing import TechnicalDocumentProcessor
from monitoring import RAGMonitor
from evaluation import RAGEvaluator, TECHNICAL_TEST_QUESTIONS

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def load_technical_documents(config: RAGConfig, processor: TechnicalDocumentProcessor):
    """Lädt technische PDF-Dokumente mit OCR-Fallback."""
    try:
        if not os.path.exists(config.data_path):
            logger.error(f"Datenordner {config.data_path} existiert nicht!")
            return []
        
        pdf_files = [f for f in os.listdir(config.data_path) if f.endswith('.pdf')]
        if not pdf_files:
            logger.warning(f"Keine PDF-Dateien in {config.data_path} gefunden!")
            return []
        
        documents = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(config.data_path, pdf_file)
            logger.info(f"Verarbeite {pdf_file}")
            
            try:
                # Versuche Standard-PDF-Loader
                document_loader = PyPDFDirectoryLoader(config.data_path)
                docs = document_loader.load()
                
                # Filtere nur das aktuelle PDF
                current_docs = [doc for doc in docs if pdf_file in doc.metadata.get('source', '')]
                
                if current_docs and any(len(doc.page_content.strip()) > 50 for doc in current_docs):
                    documents.extend(current_docs)
                    logger.info(f"Standard-Extraktion erfolgreich für {pdf_file}")
                else:
                    raise Exception("Wenig Text gefunden")
                    
            except Exception:
                # Fallback zu OCR
                logger.info(f"Verwende OCR für {pdf_file}")
                text = processor.extract_text_with_ocr(pdf_path)
                if text and len(text.strip()) > 50:
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_file, 
                            "extracted_with": "OCR",
                            "document_type": "technical_datasheet"
                        }
                    )
                    documents.append(doc)
                else:
                    logger.warning(f"Keine verwertbaren Daten aus {pdf_file} extrahiert")
        
        logger.info(f"Technische Dokumente geladen: {len(documents)}")
        return documents
        
    except Exception as e:
        logger.error(f"Fehler beim Laden technischer Dokumente: {e}")
        return []

def split_technical_documents(documents: list[Document], config: RAGConfig):
    """Spezielles Chunking für technische Produktdatenblätter."""
    
    table_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size // 2,
        chunk_overlap=config.chunk_overlap // 2,
        separators=["\n\n", "\n", ":", ";", ",", " "],
        length_function=len,
        is_separator_regex=False
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        length_function=len,
        is_separator_regex=False
    )
    
    technical_chunks = []
    
    for doc in documents:
        content = doc.page_content.lower()
        
        # Erkenne technische Abschnitte
        technical_keywords = [
            'spezifikation', 'technical', 'specification', 'datenblatt',
            'cpu', 'ram', 'memory', 'processor', 'ghz', 'mhz', 'gb', 'tb',
            'anschluss', 'port', 'interface', 'connector', 'usb', 'hdmi',
            'workstation', 'thin client', 'monitor', 'display'
        ]
        
        if any(keyword in content for keyword in technical_keywords):
            chunks = table_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata['chunk_type'] = 'technical'
        else:
            chunks = text_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata['chunk_type'] = 'descriptive'
        
        technical_chunks.extend(chunks)
    
    filtered_chunks = [chunk for chunk in technical_chunks if len(chunk.page_content.strip()) > 30]
    
    logger.info(f"Technische Dokumente in {len(filtered_chunks)} Chunks aufgeteilt")
    return filtered_chunks

def load_or_create_vector_store(chunks: list[Document], config: RAGConfig):
    """Lädt existierenden Vector Store oder erstellt einen neuen."""
    
    if config.use_openai_embeddings:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
    
    if os.path.exists(config.chroma_path):
        logger.info("Lade existierenden Vector Store...")
        try:
            vector_store = Chroma(
                persist_directory=config.chroma_path,
                embedding_function=embeddings
            )
            test_results = vector_store.similarity_search("test", k=1)
            logger.info(f"Vector Store erfolgreich geladen")
            return vector_store
        except Exception as e:
            logger.warning(f"Fehler beim Laden des Vector Store: {e}")
            import shutil
            shutil.rmtree(config.chroma_path)
    
    if config.max_chunks and len(chunks) > config.max_chunks:
        chunks = chunks[:config.max_chunks]
        logger.info(f"Limitiert auf {config.max_chunks} Chunks")

    if not chunks:
        raise ValueError("Keine Chunks zum Erstellen des Vector Store verfügbar!")

    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=config.chroma_path
        )
        vector_store.persist()
        logger.info("Vector Store erfolgreich erstellt und gespeichert")
        return vector_store
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Vector Store: {e}")
        raise

def create_technical_qa_chain(vector_store, config: RAGConfig):
    """Erstellt eine spezialisierte QA-Chain für technische Produktdaten."""
    
    if config.default_language == 'de':
        prompt_template = """Du bist ein Experte für technische Produktdatenblätter von Thin Clients, Workstations und Monitoren.

ANWEISUNGEN:
1. Beantworte Fragen präzise basierend auf den technischen Spezifikationen
2. Bei technischen Daten gib exakte Werte an (z.B. CPU, RAM, Anschlüsse, Auflösung)
3. Strukturiere Antworten bei Listen mit Aufzählungszeichen
4. Wenn spezifische Daten nicht verfügbar sind, sage das deutlich
5. Verwende technische Fachbegriffe korrekt
6. Bei Vergleichen zwischen Produkten liste die Unterschiede auf

TECHNISCHE PRODUKTDATEN:
{context}

BENUTZERANFRAGE: {question}

DETAILLIERTE ANTWORT:"""
    else:
        prompt_template = """You are an expert for technical product datasheets of Thin Clients, Workstations and Monitors.

INSTRUCTIONS:
1. Answer questions precisely based on technical specifications
2. For technical data provide exact values (e.g. CPU, RAM, ports, resolution)
3. Structure answers with bullet points for lists
4. If specific data is not available, state this clearly
5. Use technical terminology correctly
6. For product comparisons, list the differences

TECHNICAL PRODUCT DATA:
{context}

USER QUERY: {question}

DETAILED ANSWER:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    llm = ChatOpenAI(
        model=config.model_name, 
        temperature=config.temperature,
        max_tokens=config.max_tokens
    )
    
    retriever = vector_store.as_retriever(
        search_type=config.retrieval_type,
        search_kwargs={
            "k": config.retrieval_k,
            "fetch_k": config.retrieval_k * 2,
            "lambda_mult": 0.7
        }
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    logger.info("Technische QA Chain erfolgreich erstellt")
    return qa_chain

def initialize_technical_rag_system(language='de', max_chunks=100):
    """Initialisiert das technische RAG-System."""
    try:
        config = RAGConfig.from_env()
        config.default_language = language
        config.max_chunks = max_chunks
        
        # Komponenten initialisieren
        processor = TechnicalDocumentProcessor()
        monitor = RAGMonitor(config.logs_path)
        
        logger.info("Initialisiere technisches RAG-System...")
        
        # Dokumente laden
        documents = load_technical_documents(config, processor)
        if not documents:
            raise ValueError("Keine technischen Dokumente gefunden!")
        
        # Technische Verarbeitung
        processed_docs = processor.extract_technical_sections(documents)
        chunks = split_technical_documents(processed_docs or documents, config)
        
        if not chunks:
            raise ValueError("Keine verwertbaren Chunks erstellt!")
        
        # Vector Store
        vector_store = load_or_create_vector_store(chunks, config)
        
        # QA Chain
        qa_chain = create_technical_qa_chain(vector_store, config)
        
        # Translator
        translator = MultilingualHandler() if config.enable_translation else None
        
        logger.info("Technisches RAG-System erfolgreich initialisiert")
        return qa_chain, translator
        
    except Exception as e:
        logger.error(f"Fehler bei der technischen Initialisierung: {e}")
        raise

def ask_technical_question(qa_chain, question: str, translator=None, translate_query=False, target_language='de'):
    """Stellt eine technische Frage an das RAG-System."""
    try:
        start_time = time.time()
        original_question = question
        
        if translate_query and translator:
            if target_language == 'en':
                question = translator.translate_text(question, 'de', 'en')
                logger.info(f"Frage übersetzt: {original_question} -> {question}")
        
        result = qa_chain({"query": question})
        
        answer = result["result"]
        
        if translate_query and translator and target_language == 'en':
            answer = translator.translate_text(answer, 'en', 'de')
        
        response_time = time.time() - start_time
        
        return {
            "answer": answer,
            "sources": result["source_documents"],
            "source_count": len(result["source_documents"]),
            "original_question": original_question,
            "processed_question": question,
            "response_time": response_time
        }
    except Exception as e:
        logger.error(f"Fehler bei der technischen Abfrage: {e}")
        return {
            "answer": "Es gab einen Fehler bei der Verarbeitung Ihrer technischen Anfrage.",
            "sources": [],
            "source_count": 0,
            "original_question": question,
            "processed_question": question,
            "response_time": 0
        }

def run_evaluation():
    """Führt System-Evaluation durch."""
    print("🔬 Starte System-Evaluation...")
    
    qa_chain, _ = initialize_technical_rag_system(language='de', max_chunks=100)
    evaluator = RAGEvaluator(qa_chain)
    
    results = evaluator.run_benchmark(TECHNICAL_TEST_QUESTIONS)
    
    print(f"📊 Evaluation abgeschlossen:")
    print(f"   • Durchschnittliche Antwortzeit: {results['average_response_time']:.2f}s")
    print(f"   • Getestete Fragen: {results['total_questions']}")
    
    evaluator.save_evaluation(results)

def main():
    """Hauptfunktion für interaktive Nutzung."""
    try:
        print("🚀 Initialisiere technisches RAG-System für Produktdatenblätter...")
        qa_chain, translator = initialize_technical_rag_system(language='de', max_chunks=100)
        print("✅ Technisches RAG-System bereit!")
        
        while True:
            query = input("\n❓ Stellen Sie Ihre technische Frage (oder 'quit' zum Beenden): ")
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Auf Wiedersehen!")
                break
                
            if not query.strip():
                continue
                
            print("🔍 Analysiere technische Daten...")
            result = ask_technical_question(qa_chain, query, translator)
            
            print(f"\n💡 Antwort: {result['answer']}")
            print(f"📚 Quellen: {result['source_count']} technische Dokumente verwendet")
            print(f"⏱️ Antwortzeit: {result['response_time']:.2f}s")
            
    except Exception as e:
        logger.error(f"Fehler in der Hauptfunktion: {e}")
        print("❌ Fehler beim Initialisieren des technischen Systems.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        run_evaluation()
    else:
        main()