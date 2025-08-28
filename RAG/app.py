import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import os
import time
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
    return initialize_technical_rag_system(language=language, max_chunks=max_chunks)

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
        
        st.divider()
        
        # System Status
        st.header("📊 System Status")
        
        # Prüfe Ordner und Dateien
        if os.path.exists("./data"):
            pdf_files = [f for f in os.listdir("./data") if f.endswith('.pdf')]
            st.success(f"✅ {len(pdf_files)} PDF-Dateien gefunden")
            with st.expander("📄 PDF-Dateien anzeigen"):
                for pdf in pdf_files:
                    st.text(f"📄 {pdf}")
        else:
            st.error("❌ ./data Ordner nicht gefunden!")
        
        if os.path.exists("./chroma"):
            st.success("✅ Vector Store vorhanden")
        else:
            st.info("ℹ️ Vector Store wird erstellt")
        
        if os.path.exists(".env"):
            st.success("✅ .env Datei gefunden")
        else:
            st.error("❌ .env Datei fehlt!")
        
        # Statistiken
        if "monitor" in st.session_state:
            monitor = st.session_state.monitor
            stats = monitor.get_statistics()
            st.subheader("📈 Statistiken")
            st.metric("Anzahl Anfragen", stats.get("total_queries", 0))
            if stats.get("avg_response_time"):
                st.metric("Ø Antwortzeit", f"{stats['avg_response_time']:.2f}s")
    
    # Hauptbereich
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "metadata" in message:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.caption(f"📚 {message['metadata']['source_count']} Quellen")
                    with col_b:
                        st.caption(f"⏱️ {message['metadata'].get('response_time', 0):.2f}s")
                    with col_c:
                        if st.button("📋 Quellen anzeigen", key=f"sources_{len(st.session_state.messages)}"):
                            with st.expander("Verwendete Quellen"):
                                for source in message['metadata'].get('sources', []):
                                    st.text(f"📄 {source}")
        
        # User Input
        if prompt := st.chat_input("Technische Frage stellen..."):
            # User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Analysiere technische Daten..."):
                    try:
                        # System initialisieren
                        if "rag_system" not in st.session_state:
                            qa_chain, translator = load_rag_system(language, max_chunks)
                            st.session_state.rag_system = (qa_chain, translator)
                            st.session_state.monitor = RAGMonitor()
                        
                        qa_chain, translator = st.session_state.rag_system
                        monitor = st.session_state.monitor
                        
                        # Frage stellen
                        result = ask_technical_question(
                            qa_chain, prompt, translator, 
                            translate_query=enable_translation,
                            target_language=language
                        )
                        
                        # Monitoring
                        monitor.log_query(
                            prompt, 
                            result["answer"], 
                            result["response_time"], 
                            result["source_count"]
                        )
                        
                        # Antwort anzeigen
                        st.markdown(result["answer"])
                        
                        # Metrics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.caption(f"📚 {result['source_count']} Quellen verwendet")
                        with col_b:
                            st.caption(f"⏱️ {result['response_time']:.2f}s")
                        with col_c:
                            if result['source_count'] > 0:
                                st.caption("✅ Technische Daten gefunden")
                        
                        # Message speichern
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "metadata": {
                                "source_count": result['source_count'],
                                "response_time": result['response_time'],
                                "sources": [s.metadata.get('source', 'Unknown') for s in result['sources'][:3]]
                            }
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Fehler: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
    
    with col2:
        st.header("💡 Beispiel-Fragen")
        
        example_categories = {
            "🖥️ Hardware": [
                "Welche CPU-Modelle sind verfügbar?",
                "Wie viel RAM unterstützen die Geräte?",
                "Welche Grafikkarten sind verbaut?"
            ],
            "🔌 Anschlüsse": [
                "Welche Anschlüsse sind vorhanden?",
                "Gibt es USB-C Ports?",
                "Welche Display-Anschlüsse existieren?"
            ],
            "📊 Vergleiche": [
                "Vergleiche die Monitore",
                "Unterschiede zwischen Workstations",
                "Welches Gerät hat die beste Performance?"
            ],
            "📋 Spezifikationen": [
                "Zeige technische Spezifikationen",
                "Was ist die maximale Auflösung?",
                "Welche Betriebssysteme werden unterstützt?"
            ]
        }
        
        for category, questions in example_categories.items():
            with st.expander(category):
                for question in questions:
                    if st.button(question, key=f"example_{question}", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": question})
                        st.rerun()
        
        st.divider()
        
        # Actions
        st.header("🔧 Aktionen")
        
        if st.button("🗑️ Chat löschen", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 System neu starten", use_container_width=True):
            # Cache leeren
            st.cache_resource.clear()
            # Session State zurücksetzen
            for key in ["rag_system", "monitor"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("System wird neu gestartet!")
            st.rerun()
        
        if st.button("📊 Evaluation starten", use_container_width=True):
            with st.spinner("Führe Evaluation durch..."):
                # Hier würden Sie die Evaluation starten
                st.info("Evaluation würde hier gestartet werden...")

if __name__ == "__main__":
    main()