import os
import logging
from typing import List, Dict, Any
import time
from pathlib import Path
import re

# LangChain Imports - Updated
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Document Loading
from langchain_community.document_loaders import PyMuPDFLoader
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# Translation
from googletrans import Translator

# Configuration
from config import RAGConfig

def initialize_technical_rag_system(language: str = "de", max_chunks: int = 100):
    """Initialisiert das RAG-System mit technischen Dokumenten."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initialisiere technisches RAG-System...")
        
        # Konfiguration laden
        config = RAGConfig()
        
        # Embeddings Model
        embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Vector Store - Prüfe ob bereits vorhanden
        vector_store_path = "./chroma"
        
        if os.path.exists(vector_store_path) and os.listdir(vector_store_path):
            logger.info(f"Lade existierende Vector Database von {vector_store_path}")
            try:
                vectorstore = Chroma(
                    persist_directory=vector_store_path,
                    embedding_function=embeddings
                )
                
                # Prüfe ob Daten vorhanden sind
                collection_count = vectorstore._collection.count()
                if collection_count > 0:
                    logger.info(f"Vector Database geladen mit {collection_count} Dokumenten")
                else:
                    logger.info("Vector Database ist leer, erstelle neue...")
                    vectorstore = create_new_vector_store(config, embeddings, max_chunks, vector_store_path)
                    
            except Exception as e:
                logger.warning(f"Fehler beim Laden der Vector Database: {e}")
                logger.info("Erstelle neue Vector Database...")
                vectorstore = create_new_vector_store(config, embeddings, max_chunks, vector_store_path)
        else:
            logger.info("Keine existierende Vector Database gefunden, erstelle neue...")
            vectorstore = create_new_vector_store(config, embeddings, max_chunks, vector_store_path)
        
        # QA Chain erstellen
        llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.temperature,
            openai_api_key=config.openai_api_key
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": get_technical_prompt_template(language)
            }
        )
        
        # Translator
        translator = Translator()
        
        logger.info("RAG-System erfolgreich initialisiert")
        return qa_chain, translator
        
    except Exception as e:
        logger.error(f"Fehler bei der Initialisierung: {str(e)}")
        raise

def create_new_vector_store(config, embeddings, max_chunks, vector_store_path):
    """Erstellt eine neue Vector Database."""
    logger = logging.getLogger(__name__)
    
    # Dokumente laden
    documents = load_technical_documents("./data")
    
    if not documents:
        raise ValueError("Keine Dokumente gefunden!")
    
    logger.info(f"Technische Dokumente geladen: {len(documents)}")
    
    # Text Splitting mit erweiterten Metadaten
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents[:max_chunks])
    
    # Erweitere Metadaten für bessere Quellenangaben
    for i, split in enumerate(splits):
        split.metadata['chunk_id'] = i
        split.metadata['char_count'] = len(split.page_content)
        # Stelle sicher, dass wichtige Metadaten existieren
        if 'source' not in split.metadata:
            split.metadata['source'] = 'Unknown'
        if 'page' not in split.metadata:
            split.metadata['page'] = 1
        if 'extraction_method' not in split.metadata:
            split.metadata['extraction_method'] = 'unknown'
    
    logger.info(f"Technische Dokumente in {len(splits)} Chunks aufgeteilt")
    
    # Vector Store erstellen
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=vector_store_path
    )
    
    # Persistieren
    vectorstore.persist()
    logger.info(f"Vector Store erstellt und gespeichert in {vector_store_path}")
    
    return vectorstore

def load_technical_documents(data_path: str) -> List[Document]:
    """Lädt technische Dokumente aus dem Datenverzeichnis."""
    logger = logging.getLogger(__name__)
    documents = []
    
    data_dir = Path(data_path)
    if not data_dir.exists():
        logger.error(f"Datenverzeichnis {data_path} existiert nicht")
        return documents
    
    # PDF-Dateien verarbeiten
    pdf_files = list(data_dir.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            logger.info(f"Verarbeite {pdf_file.name}")
            
            # Versuche zunächst normale PDF-Extraktion
            try:
                loader = PyMuPDFLoader(str(pdf_file))
                pdf_docs = loader.load()
                
                # Prüfe ob Text extrahiert wurde
                total_text = "".join([doc.page_content for doc in pdf_docs])
                if len(total_text.strip()) > 100:
                    # Erweitere Metadaten
                    for doc in pdf_docs:
                        doc.metadata['file_path'] = str(pdf_file)
                        doc.metadata['extraction_method'] = 'pymupdf'
                        doc.metadata['source'] = pdf_file.name  # Stelle sicher, dass source gesetzt ist
                    documents.extend(pdf_docs)
                    logger.info(f"Text aus {pdf_file.name} erfolgreich extrahiert")
                    continue
                else:
                    logger.info(f"Wenig Text gefunden, verwende OCR für {pdf_file.name}")
                    
            except Exception as e:
                logger.warning(f"PDF-Extraktion fehlgeschlagen für {pdf_file.name}: {e}")
                logger.info(f"Verwende OCR für {pdf_file.name}")
            
            # OCR als Fallback
            ocr_docs = extract_text_with_ocr(pdf_file)
            if ocr_docs:
                documents.extend(ocr_docs)
                logger.info(f"OCR-Extraktion für {pdf_file.name} erfolgreich")
            else:
                logger.warning(f"Keine Textextraktion möglich für {pdf_file.name}")
                
        except Exception as e:
            logger.error(f"Fehler beim Verarbeiten von {pdf_file.name}: {str(e)}")
            continue
    
    return documents

def extract_text_with_ocr(pdf_path: Path) -> List[Document]:
    """Extrahiert Text mittels OCR aus PDF."""
    logger = logging.getLogger(__name__)
    documents = []
    
    try:
        pdf_document = fitz.open(str(pdf_path))
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            
            # Konvertiere Seite zu Bild
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # OCR anwenden
            image = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(image, lang='deu+eng')
            
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_num + 1,
                        "extraction_method": "ocr",
                        "file_path": str(pdf_path)
                    }
                )
                documents.append(doc)
        
        pdf_document.close()
        
    except Exception as e:
        logger.error(f"OCR-Fehler für {pdf_path.name}: {str(e)}")
    
    return documents

def is_relevant_chunk(text: str, search_terms: List[str], min_relevance: float = 0.2) -> bool:
    """Prüft ob ein Text-Chunk relevant für die Suchbegriffe ist."""
    if not search_terms:
        return True
    
    text_lower = text.lower()
    
    # Technische Keywords die immer relevant sind
    technical_keywords = {
        'cpu', 'processor', 'prozessor', 'intel', 'amd', 'ryzen', 'core',
        'ram', 'memory', 'arbeitsspeicher', 'gb', 'ddr4', 'ddr5',
        'usb', 'hdmi', 'port', 'anschluss', 'connector', 'jack',
        'display', 'monitor', 'screen', 'bildschirm', 'resolution', 'auflösung',
        'graphics', 'grafik', 'gpu', 'nvidia', 'radeon',
        'storage', 'speicher', 'ssd', 'hdd', 'festplatte',
        'battery', 'akku', 'power', 'watt', 'strom'
    }
    
    # Prüfe technische Keywords
    for keyword in technical_keywords:
        if keyword in text_lower:
            for term in search_terms:
                if term.lower() in text_lower:
                    return True  # Hohe Relevanz wenn technisches Keyword + Suchbegriff
    
    # Standard-Relevanz-Prüfung
    matches = 0
    for term in search_terms:
        if term.lower() in text_lower:
            matches += 1
            # Bonus für längere, spezifische Begriffe
            if len(term) > 4:
                matches += 0.5
    
    relevance_score = matches / len(search_terms)
    return relevance_score >= min_relevance

def highlight_text_in_context(text: str, search_terms: List[str], context_chars: int = 400) -> str:
    """Markiert Suchbegriffe im Text und gibt relevanten Kontext zurück."""
    if not search_terms:
        return text[:context_chars] + "..." if len(text) > context_chars else text
    
    # Erweiterte technische Begriffe für besseres Matching
    extended_terms = search_terms.copy()
    
    # Füge technische Variationen hinzu
    term_variations = {
        'anschluss': ['port', 'connector', 'anschlüsse', 'ports'],
        'anschlüsse': ['port', 'connector', 'anschluss', 'ports'],
        'ports': ['anschluss', 'anschlüsse', 'connector'],
        'usb': ['usb-c', 'usb-a', 'usb3', 'usb2'],
        'ram': ['memory', 'arbeitsspeicher', 'speicher'],
        'cpu': ['processor', 'prozessor'],
        'display': ['monitor', 'bildschirm', 'screen'],
        'grafik': ['graphics', 'gpu', 'video']
    }
    
    for term in search_terms[:]:
        if term.lower() in term_variations:
            extended_terms.extend(term_variations[term.lower()])
    
    # Finde alle Vorkommen
    highlights = []
    for term in extended_terms:
        # Suche nach ganzen Wörtern und Teilwörtern
        for match in re.finditer(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
            highlights.append((match.start(), match.end(), match.group()))
        # Auch Teilwort-Suche für technische Begriffe
        if len(term) > 3:
            for match in re.finditer(re.escape(term), text, re.IGNORECASE):
                highlights.append((match.start(), match.end(), match.group()))
    
    if not highlights:
        # Fallback: Suche nach technischen Mustern
        tech_patterns = [
            r'\b\d+\s*(gb|mb|ghz|mhz|watt|inch|zoll)\b',
            r'\busb[-\s]?[0-9c]\b',
            r'\bhdmi\b',
            r'\b(intel|amd|nvidia)\b',
            r'\b\d+x\d+\b'  # Auflösungen
        ]
        
        for pattern in tech_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                highlights.append((match.start(), match.end(), match.group()))
    
    if not highlights:
        return text[:context_chars] + "..." if len(text) > context_chars else text
    
    # Entferne Duplikate und sortiere
    highlights = list(set(highlights))
    highlights.sort()
    
    # Finde den besten Kontext-Bereich
    if highlights:
        # Erweitere den Kontext um das erste und letzte Highlight
        first_start = highlights[0][0]
        last_end = highlights[-1][1]
        
        start_pos = max(0, first_start - 150)
        end_pos = min(len(text), last_end + 150)
        
        context_text = text[start_pos:end_pos]
        
        # Markiere alle gefundenen Begriffe
        for start, end, term in sorted(highlights, key=lambda x: x[0], reverse=True):
            relative_start = start - start_pos
            relative_end = end - start_pos
            
            if 0 <= relative_start < len(context_text) and 0 < relative_end <= len(context_text):
                context_text = (context_text[:relative_start] + 
                              f"**{context_text[relative_start:relative_end]}**" + 
                              context_text[relative_end:])
        
        if start_pos > 0:
            context_text = "..." + context_text
        if end_pos < len(text):
            context_text = context_text + "..."
            
        return context_text
    
    return text[:context_chars] + "..." if len(text) > context_chars else text

def extract_key_terms_from_question(question: str) -> List[str]:
    """Extrahiert wichtige Begriffe aus der Frage für Highlighting."""
    # Deutsche und englische Stoppwörter
    stop_words = {
        'der', 'die', 'das', 'und', 'oder', 'ist', 'sind', 'welche', 'wie', 'was',
        'gibt', 'es', 'haben', 'hat', 'kann', 'könnte', 'wird', 'werden', 'von',
        'zu', 'mit', 'für', 'auf', 'in', 'an', 'bei', 'über', 'unter', 'durch',
        'the', 'a', 'an', 'and', 'or', 'is', 'are', 'which', 'how', 'what',
        'have', 'has', 'can', 'could', 'will', 'would', 'of', 'to', 'with',
        'for', 'on', 'in', 'at', 'by', 'about', 'under', 'through', 'do', 'does'
    }
    
    # Erweiterte technische Begriffe mit höchster Priorität
    high_priority_terms = {
        'cpu', 'processor', 'prozessor', 'intel', 'amd', 'ryzen', 'core',
        'ram', 'memory', 'arbeitsspeicher', 'ddr4', 'ddr5', 'gb', 'mb',
        'usb', 'usb-c', 'usb-a', 'hdmi', 'displayport', 'vga',
        'anschluss', 'anschlüsse', 'port', 'ports', 'connector',
        'ssd', 'hdd', 'festplatte', 'storage', 'speicher',
        'gpu', 'grafik', 'graphics', 'nvidia', 'amd', 'intel',
        'display', 'monitor', 'bildschirm', 'screen', 'auflösung', 'resolution',
        'battery', 'akku', 'power', 'netzteil', 'watt', 'strom',
        'bluetooth', 'wifi', 'wlan', 'ethernet', 'netzwerk',
        'touchpad', 'keyboard', 'tastatur', 'webcam', 'kamera',
        'audio', 'speaker', 'lautsprecher', 'mikrofon', 'headphone'
    }
    
    # Technische Modell-Namen und Spezifikationen
    technical_patterns = [
        r'\b\d+\s*(gb|mb|ghz|mhz|watt|inch|zoll)\b',
        r'\bi[357]-\w+\b',  # Intel Prozessoren
        r'\bryzen\s+\w+\b',  # AMD Prozessoren
        r'\bgtx\s+\w+\b',   # Nvidia Grafikkarten
        r'\b\d+x\d+\b',     # Auflösungen
        r'\busb\s*[0-9c]\b' # USB Versionen
    ]
    
    # Tokenisiere die Frage
    words = re.findall(r'\b\w+\b', question.lower())
    key_terms = []
    
    # 1. Sammle High-Priority technische Begriffe
    for word in words:
        if word in high_priority_terms:
            key_terms.append(word)
    
    # 2. Sammle technische Muster
    for pattern in technical_patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        key_terms.extend([match.lower() for match in matches])
    
    # 3. Sammle andere relevante Begriffe (nicht Stoppwörter)
    for word in words:
        if (word not in stop_words and 
            word not in high_priority_terms and 
            len(word) > 2 and 
            len(key_terms) < 8):  # Mehr Begriffe zulassen
            key_terms.append(word)
    
    # Entferne Duplikate und behalte Reihenfolge
    seen = set()
    unique_terms = []
    for term in key_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    return unique_terms[:8]  # Erweitert auf 8 Begriffe

def get_technical_prompt_template(language: str = "de") -> PromptTemplate:
    """Erstellt ein spezialisiertes Prompt-Template für technische Dokumente."""
    
    if language == "de":
        template = """Du bist ein Experte für technische Produktdatenblätter von Computerhardware.
        
Beantworte die folgende Frage basierend auf den bereitgestellten technischen Dokumenten.

Kontext:
{context}

Frage: {question}

Anweisungen:
- Antworte präzise und technisch korrekt
- Verwende die spezifischen technischen Daten aus den Dokumenten
- Wenn Produktvergleiche gewünscht sind, stelle die Daten tabellarisch dar
- Gib bei Spezifikationen genaue Werte an (RAM, CPU, Anschlüsse, etc.)
- Falls die Information nicht in den Dokumenten steht, sage das klar

Antwort:"""
    else:
        template = """You are an expert for technical product datasheets of computer hardware.

Answer the following question based on the provided technical documents.

Context:
{context}

Question: {question}

Instructions:
- Answer precisely and technically correct
- Use specific technical data from the documents
- If product comparisons are requested, present data in tabular format
- Provide exact values for specifications (RAM, CPU, ports, etc.)
- If information is not in the documents, state this clearly

Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def ask_technical_question(qa_chain, question: str, translator=None, translate_query: bool = False, target_language: str = "de") -> Dict[str, Any]:
    """Stellt eine technische Frage an das RAG-System."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        # Optional: Frage übersetzen
        original_question = question
        if translate_query and translator and target_language != "de":
            try:
                translated = translator.translate(question, src=target_language, dest="de")
                question = translated.text
                logger.info(f"Frage übersetzt: {original_question} -> {question}")
            except Exception as e:
                logger.warning(f"Übersetzung fehlgeschlagen: {e}")
        
        # Extrahiere Schlüsselbegriffe für Highlighting
        key_terms = extract_key_terms_from_question(question)
        
        # Frage an QA-Chain
        result = qa_chain.invoke({"query": question})
        
        # Response-Zeit berechnen
        response_time = time.time() - start_time
        
        # Antwort optional zurück übersetzen
        answer = result["result"]
        if translate_query and translator and target_language != "de":
            try:
                translated = translator.translate(answer, src="de", dest=target_language)
                answer = translated.text
            except Exception as e:
                logger.warning(f"Antwort-Übersetzung fehlgeschlagen: {e}")
        
        # KORRIGIERT: Zähle Original-Source-Documents
        original_source_count = len(result.get("source_documents", []))
        
        # Erweitere Source Documents mit Highlighting und Relevanzfilter
        enhanced_sources = []
        for doc in result.get("source_documents", []):
            # Immer alle Sources verarbeiten, auch die weniger relevanten
            highlighted_content = highlight_text_in_context(
                doc.page_content, 
                key_terms, 
                context_chars=400
            )
            
            # Berechne Relevanz-Score
            relevance_score = len([term for term in key_terms if term.lower() in doc.page_content.lower()]) / max(len(key_terms), 1)
            
            enhanced_doc = {
                'content': doc.page_content,
                'highlighted_content': highlighted_content,
                'metadata': doc.metadata,
                'preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                'relevance_score': relevance_score
            }
            enhanced_sources.append(enhanced_doc)
        
        # Sortiere nach Relevanz, aber behalte alle
        enhanced_sources.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Ergebnis formatieren - KORRIGIERT: Verwende original_source_count
        response = {
            "answer": answer,
            "source_count": original_source_count,  # Originale Anzahl für Monitoring
            "sources": enhanced_sources,  # Alle erweiterten Sources für UI
            "response_time": response_time,
            "original_question": original_question,
            "key_terms": key_terms
        }
        
        logger.info(f"Frage beantwortet in {response_time:.2f}s mit {original_source_count} Quellen")
        return response
        
    except Exception as e:
        logger.error(f"Fehler beim Beantworten der Frage: {str(e)}")
        return {
            "answer": f"Entschuldigung, es gab einen Fehler beim Verarbeiten Ihrer Frage: {str(e)}",
            "source_count": 0,
            "sources": [],
            "response_time": time.time() - start_time,
            "original_question": original_question,
            "key_terms": []
        }

if __name__ == "__main__":
    # Für Testing
    logging.basicConfig(level=logging.INFO)
    
    try:
        qa_chain, translator = initialize_technical_rag_system()
        
        test_questions = [
            "Welche CPU-Modelle sind verfügbar?",
            "Wie viel RAM unterstützen die Geräte?",
            "Welche Anschlüsse sind vorhanden?"
        ]
        
        for question in test_questions:
            print(f"\nFrage: {question}")
            result = ask_technical_question(qa_chain, question, translator)
            print(f"Antwort: {result['answer']}")
            print(f"Quellen: {result['source_count']}")
            print(f"Zeit: {result['response_time']:.2f}s")
            
    except Exception as e:
        print(f"Fehler: {e}")
