import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import os
import time
import re
from typing import List
from main import initialize_technical_rag_system, ask_technical_question
from config import RAGConfig
from monitoring import RAGMonitor
import logging

st.set_page_config(
    page_title="Technisches RAG-System",
    page_icon="🔧",
    layout="wide"
)

@st.cache_resource
def load_rag_system(language, max_chunks):
    """Cached System-Initialisierung."""
    if os.path.exists("./chroma") and os.listdir("./chroma"):
        st.info("🔄 Lade existierende Vector Database...")
    else:
        st.info("🔄 Erstelle neue Vector Database...")
    
    return initialize_technical_rag_system(language=language, max_chunks=max_chunks)

def process_question(question, language, max_chunks, enable_translation):
    """Verarbeitet eine Frage und gibt das Ergebnis zurück."""
    try:
        # System initialisieren
        if "rag_system" not in st.session_state:
            st.session_state.monitor = RAGMonitor()
            st.session_state.monitor.log_system_event("System initialization started")
            qa_chain, translator = load_rag_system(language, max_chunks)
            st.session_state.rag_system = (qa_chain, translator)
            st.session_state.monitor.log_system_event("System initialization completed")
        
        qa_chain, translator = st.session_state.rag_system
        monitor = st.session_state.monitor
        
        # Logge die eingehende Frage
        monitor.log_system_event("Processing question", f"Language: {language}, Translation: {enable_translation}")
        
        # Frage stellen
        result = ask_technical_question(
            qa_chain, question, translator, 
            translate_query=enable_translation,
            target_language=language
        )
        
        # DEBUG: Logge Source-Count
        actual_source_count = result["source_count"]
        enhanced_sources_count = len(result.get("sources", []))
        
        if actual_source_count == 0:
            monitor.log_system_event("Warning", f"No sources found for question: {question}")
        
        # Monitoring mit korrekter Source-Count
        monitor.log_query(
            question, 
            result["answer"], 
            result["response_time"], 
            actual_source_count  # Verwende die korrekte Anzahl
        )
        
        return result
        
    except Exception as e:
        if "monitor" in st.session_state:
            st.session_state.monitor.log_error(str(e), f"Question: {question}")
        
        return {
            "answer": f"❌ Fehler: {str(e)}",
            "source_count": 0,
            "response_time": 0.0,
            "sources": [],
            "key_terms": []
        }

def show_document_preview(sources, key_terms):
    """Zeigt eine verbesserte Dokumentvorschau mit relevanten Passagen."""
    if not sources:
        st.info("Keine Quellen verfügbar")
        return
    
    st.subheader("📄 Verwendete Quellen mit Vorschau")
    
    # Verbesserte Relevanz-Bewertung
    relevant_sources = []
    for source in sources:
        relevance = source.get('relevance_score', 0)
        content = source.get('content', '')
        
        # Zusätzliche Relevanz-Checks
        content_lower = content.lower()
        bonus_score = 0
        
        # Bonus für technische Begriffe
        tech_indicators = ['usb', 'hdmi', 'cpu', 'ram', 'gpu', 'port', 'anschluss', 
                          'display', 'monitor', 'processor', 'memory', 'storage']
        for indicator in tech_indicators:
            if indicator in content_lower:
                bonus_score += 0.1
        
        # Bonus für Spezifikationen (Zahlen + Einheiten)
        spec_patterns = [r'\d+\s*(gb|mb|ghz|mhz|watt|inch|zoll)', 
                        r'\d+x\d+', r'usb\s*[0-9c]', r'\d+\s*port']
        for pattern in spec_patterns:
            if re.search(pattern, content_lower):
                bonus_score += 0.15
        
        # Finale Relevanz-Bewertung
        final_relevance = min(1.0, relevance + bonus_score)
        source['final_relevance'] = final_relevance
        
        # Nur relevante Sources anzeigen
        if final_relevance > 0.15 or len(content) > 100:
            relevant_sources.append(source)
    
    # Sortiere nach finaler Relevanz
    relevant_sources.sort(key=lambda x: x.get('final_relevance', 0), reverse=True)
    
    if not relevant_sources:
        st.warning("Keine relevanten Quellen gefunden für diese Frage.")
        return
    
    st.caption(f"📊 {len(relevant_sources)} relevante Quellen gefunden (von {len(sources)} insgesamt)")
    
    # Zeige nur die relevantesten Quellen
    for i, source in enumerate(relevant_sources[:4]):  # Limitiert auf 4 beste
        relevance = source.get('final_relevance', 0)
        
        # Relevanz-Kategorien
        if relevance > 0.5:
            indicator = "�"
            category = "Sehr relevant"
            color = "success"
        elif relevance > 0.3:
            indicator = "🟡"
            category = "Relevant"
            color = "warning"
        else:
            indicator = "�"
            category = "Wenig relevant"
            color = "secondary"
        
        source_name = source['metadata'].get('source', 'Unbekannt')
        page_num = source['metadata'].get('page', 'N/A')
        
        with st.expander(f"{indicator} **{category}** - {source_name} (Seite {page_num}) - {relevance:.0%}", expanded=(i==0)):
            
            # Metadaten in kompakter Form
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Extraction", source['metadata'].get('extraction_method', 'Standard').title())
            with col2:
                st.metric("Relevanz", f"{relevance:.0%}")
            with col3:
                st.metric("Zeichen", len(source['content']))
            with col4:
                chunk_id = source['metadata'].get('chunk_id', i + 1)
                st.metric("Chunk", chunk_id)
            
            # Improved content display
            tabs = st.tabs(["🔍 Relevanter Inhalt", "📝 Volltext", "ℹ️ Details"])
            
            with tabs[0]:
                st.markdown("**Gefundene relevante Passage:**")
                
                # Bessere Hervorhebung
                if 'highlighted_content' in source and source['highlighted_content'].strip():
                    highlighted = source['highlighted_content']
                    # Verbessere Markdown-Formatierung
                    highlighted = highlighted.replace('**', '`').replace('`', '**')
                    st.markdown(highlighted)
                else:
                    # Fallback: Zeige wichtigsten Teil
                    content = source['content']
                    if key_terms:
                        # Finde relevanteste Passage
                        best_snippet = find_best_snippet(content, key_terms)
                        st.markdown(best_snippet)
                    else:
                        st.markdown(content[:300] + "..." if len(content) > 300 else content)
                
                # Zeige gefundene Schlüsselbegriffe
                if key_terms:
                    found_terms = [term for term in key_terms if term.lower() in source['content'].lower()]
                    if found_terms:
                        st.markdown(f"**� Gefundene Begriffe:** {', '.join(found_terms)}")
            
            with tabs[1]:
                st.markdown("**Vollständiger Textinhalt:**")
                with st.container():
                    st.text_area(
                        "Volltext", 
                        source['content'], 
                        height=200, 
                        key=f"full_content_{i}_{hash(source['content'][:50])}",
                        help="Vollständiger Text dieses Dokument-Chunks"
                    )
            
            with tabs[2]:
                st.markdown("**Technische Details:**")
                details = {
                    "📄 Datei": source['metadata'].get('source', 'Unbekannt'),
                    "📖 Seite": source['metadata'].get('page', 'Unbekannt'),
                    "🔧 Extraktion": source['metadata'].get('extraction_method', 'Standard'),
                    "📊 Relevanz-Score": f"{relevance:.2%}",
                    "📏 Textlänge": f"{len(source['content'])} Zeichen",
                    "🆔 Chunk-ID": source['metadata'].get('chunk_id', 'Unbekannt')
                }
                
                for label, value in details.items():
                    st.text(f"{label}: {value}")
    
    # Zusätzliche Info
    if len(relevant_sources) < len(sources):
        remaining = len(sources) - len(relevant_sources)
        st.info(f"ℹ️ {remaining} weitere Quellen mit geringer Relevanz ausgeblendet")

def find_best_snippet(text: str, key_terms: List[str], snippet_length: int = 200) -> str:
    """Findet den relevantesten Textausschnitt."""
    if not key_terms:
        return text[:snippet_length] + "..." if len(text) > snippet_length else text
    
    # Finde alle Positionen der Schlüsselbegriffe
    positions = []
    text_lower = text.lower()
    
    for term in key_terms:
        term_lower = term.lower()
        start = 0
        while True:
            pos = text_lower.find(term_lower, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
    
    if not positions:
        return text[:snippet_length] + "..." if len(text) > snippet_length else text
    
    # Finde den Bereich mit den meisten Begriffen
    best_start = 0
    best_count = 0
    
    for pos in positions:
        start = max(0, pos - snippet_length // 2)
        end = min(len(text), start + snippet_length)
        
        # Zähle Begriffe in diesem Bereich
        snippet = text[start:end].lower()
        count = sum(1 for term in key_terms if term.lower() in snippet)
        
        if count > best_count:
            best_count = count
            best_start = start
    
    # Extrahiere besten Snippet
    end = min(len(text), best_start + snippet_length)
    snippet = text[best_start:end]
    
    # Markiere gefundene Begriffe
    for term in key_terms:
        snippet = re.sub(
            re.escape(term), 
            f"**{term}**", 
            snippet, 
            flags=re.IGNORECASE
        )
    
    if best_start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet

def main():
    st.title("🔧 Technisches RAG-System für Produktdatenblätter")
    st.markdown("*Entwickelt für Thin Clients, Workstations und Monitore*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        
        # Konfiguration
        language = st.selectbox("Sprache", ["de", "en"], index=0)
        max_chunks = st.slider("Max. Chunks", 50, 200, 100)
        enable_translation = st.checkbox("Übersetzung aktivieren", value=False)
        
        # Erweiterte Einstellungen
        with st.expander("🔧 Erweiterte Einstellungen"):
            temperature = st.slider("LLM Temperatur", 0.0, 1.0, 0.1)
            retrieval_k = st.slider("Retrieval K", 3, 10, 5)
            show_sources = st.checkbox("Quellenvorschau anzeigen", value=True)
        
        st.divider()
        
        # System Status
        st.header("System Status")
        
        # Prüfe Ordner und Dateien
        if os.path.exists("./data"):
            pdf_files = [f for f in os.listdir("./data") if f.endswith('.pdf')]
            st.success(f"{len(pdf_files)} PDF-Dateien gefunden")
            with st.expander("PDF-Dateien anzeigen"):
                for pdf in pdf_files:
                    st.text(f"{pdf}")
        else:
            st.error("./data Ordner nicht gefunden!")
        
        if os.path.exists("./chroma"):
            st.success("Vector Store vorhanden")
        else:
            st.info("Vector Store wird erstellt")
        
        if os.path.exists(".env"):
            st.success(".env Datei gefunden")
        else:
            st.error(".env Datei fehlt!")
        
        # Log-Dateien Status
        if os.path.exists("./logs"):
            log_files = [f for f in os.listdir("./logs") if f.endswith('.log')]
            if log_files:
                st.success(f"{len(log_files)} Log-Dateien vorhanden")
            else:
                st.info("Keine Log-Dateien vorhanden")
        else:
            st.info("Logs-Ordner wird erstellt")
        
        # Statistiken
        if "monitor" in st.session_state:
            monitor = st.session_state.monitor
            stats = monitor.get_statistics()
            st.subheader("📈 Statistiken")
            
            # Basis-Metriken
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Anzahl Anfragen", stats.get("total_queries", 0))
                if stats.get("avg_response_time", 0) > 0:
                    st.metric(
                        "Ø Antwortzeit", 
                        f"{stats['avg_response_time']:.2f}s",
                        delta=f"Min: {stats.get('min_response_time', 0):.2f}s"
                    )
            with col2:
                if stats.get("avg_source_count", 0) > 0:
                    st.metric(
                        "Ø Quellen", 
                        f"{stats['avg_source_count']:.1f}",
                        delta=f"Total: {stats.get('total_sources', 0)}"
                    )
                if stats.get("max_response_time", 0) > 0:
                    st.metric(
                        "Max Antwortzeit", 
                        f"{stats['max_response_time']:.2f}s"
                    )
            
            # Erweiterte Statistiken in Expander
            with st.expander("📊 Detaillierte Statistiken"):
                detailed_stats = monitor.get_detailed_statistics()
                if detailed_stats:
                    st.json(detailed_stats)
            
            # Reset Button
            if st.button("🔄 Statistiken zurücksetzen", use_container_width=True):
                monitor.reset_statistics()
                st.success("Statistiken zurückgesetzt!")
                st.rerun()
        
        st.divider()
        
        # Actions
        st.header("🔧 Aktionen")
        
        if st.button("🗑️ Chat löschen", use_container_width=True):
            if "monitor" in st.session_state:
                st.session_state.monitor.log_system_event("Chat cleared", f"Messages deleted: {len(st.session_state.messages)}")
            st.session_state.messages = []
            st.rerun()
        
        if st.button("System neu starten", use_container_width=True):
            if "monitor" in st.session_state:
                st.session_state.monitor.log_system_event("System restart initiated")
            
            # Cache leeren
            st.cache_resource.clear()
            # Session State zurücksetzen
            for key in ["rag_system", "monitor"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("System wird neu gestartet!")
            st.rerun()
        
        if st.button("Evaluation starten", use_container_width=True):
            if "monitor" in st.session_state:
                st.session_state.monitor.log_system_event("Evaluation started")
            
            with st.spinner("Führe Evaluation durch..."):
                st.info("Evaluation würde hier gestartet werden...")
    
    # Hauptbereich - 2-spaltig
    main_col, example_col = st.columns([3, 1])
    
    with main_col:
        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Chat History Container
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and "metadata" in message:
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.caption(f"{message['metadata']['source_count']} Quellen")
                        with col_b:
                            st.caption(f"{message['metadata'].get('response_time', 0):.2f}s")
                        with col_c:
                            if st.button("Quellen anzeigen", key=f"sources_{i}"):
                                # Verwende Session State für ausgewählte Nachricht
                                st.session_state.selected_message_sources = message['metadata'].get('sources', [])
                                st.session_state.selected_message_terms = message['metadata'].get('key_terms', [])
                                st.session_state.show_source_preview = True
        
        # Zeige Quellenvorschau wenn ausgewählt
        if st.session_state.get('show_source_preview', False):
            sources = st.session_state.get('selected_message_sources', [])
            key_terms = st.session_state.get('selected_message_terms', [])
            
            with st.container():
                show_document_preview(sources, key_terms)
                
                if st.button("Vorschau schließen"):
                    st.session_state.show_source_preview = False
                    st.rerun()
        
        # Prüfe auf neue Fragen (von Beispiel-Buttons)
        if "pending_question" in st.session_state:
            question = st.session_state.pending_question
            del st.session_state.pending_question
            
            # User Message hinzufügen
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Assistant Response verarbeiten
            with st.chat_message("assistant"):
                with st.spinner("Analysiere technische Daten..."):
                    result = process_question(question, language, max_chunks, enable_translation)
                    
                    # Antwort anzeigen
                    st.markdown(result["answer"])
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.caption(f"{result['source_count']} Quellen verwendet")
                    with col_b:
                        st.caption(f"{result['response_time']:.2f}s")
                    with col_c:
                        if result['source_count'] > 0:
                            st.caption("Technische Daten gefunden")
                    
                    # Message speichern - KORRIGIERT
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": {
                            "source_count": result['source_count'],
                            "response_time": result['response_time'],
                            "sources": result.get('sources', []),  # Bereits erweiterte Sources
                            "key_terms": result.get('key_terms', [])
                        }
                    })
                
                st.rerun()

    with example_col:
        st.header("Beispiel-Fragen")
        
        example_categories = {
            "Hardware": [
                "Welche CPU-Modelle sind verfügbar?",
                "Wie viel RAM unterstützen die Geräte?",
                "Welche Grafikkarten sind verbaut?"
            ],
            "Anschlüsse": [
                "Welche Anschlüsse sind vorhanden?",
                "Gibt es USB-C Ports?",
                "Welche Display-Anschlüsse existieren?"
            ],
            "Vergleiche": [
                "Vergleiche die Monitore",
                "Unterschiede zwischen Workstations",
                "Welches Gerät hat die beste Performance?"
            ],
            "Spezifikationen": [
                "Zeige technische Spezifikationen",
                "Was ist die maximale Auflösung?",
                "Welche Betriebssysteme werden unterstützt?"
            ]
        }
        
        for category, questions in example_categories.items():
            with st.expander(category):
                for i, question in enumerate(questions):
                    if st.button(question, key=f"example_{category}_{i}", use_container_width=True):
                        # Logge Beispiel-Fragen-Auswahl
                        if "monitor" in st.session_state:
                            st.session_state.monitor.log_system_event("Example question selected", f"Category: {category}, Question: {question}")
                        
                        # Setze pending question für Verarbeitung
                        st.session_state.pending_question = question
                        st.rerun()
        
    # Chat Input am Ende - immer sichtbar
    st.divider()
    
    # User Input
    if prompt := st.chat_input("Technische Frage stellen..."):
        # User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Assistant Response verarbeiten
        with st.spinner("🔍 Analysiere technische Daten..."):
            result = process_question(prompt, language, max_chunks, enable_translation)
            
            # Message speichern - KORRIGIERT
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "metadata": {
                    "source_count": result['source_count'],
                    "response_time": result['response_time'],
                    "sources": result.get('sources', []),  # Bereits erweiterte Sources
                    "key_terms": result.get('key_terms', [])
                }
            })
        
        st.rerun()

if __name__ == "__main__":
    main()
