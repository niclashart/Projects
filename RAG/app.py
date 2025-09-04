import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import os
import time
import re
import shutil
import subprocess
from typing import List, Tuple
from main import initialize_technical_rag_system, ask_technical_question
from config import RAGConfig
from monitoring import RAGMonitor
from auto_update import DatabaseAutoUpdater
import logging

# Für Database-Diagnose
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(
    page_title="Technisches RAG-System",
    page_icon="🔧",
    layout="wide"
)

def safe_remove_directory(path):
    """Sicheres Entfernen eines Verzeichnisses mit Windows-Kompatibilität."""
    if not os.path.exists(path):
        return True
    
    try:
        # Erste Methode: Normale Löschung
        shutil.rmtree(path)
        return True
    except PermissionError:
        try:
            # Zweite Methode: Erst schreibgeschützte Dateien entsperren
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.chmod(file_path, 0o777)
                    except:
                        pass
            
            time.sleep(0.5)  # Kurz warten
            shutil.rmtree(path)
            return True
        except:
            try:
                # Dritte Methode: Windows rmdir Befehl
                subprocess.run(['rmdir', '/s', '/q', path], 
                             shell=True, capture_output=True)
                return not os.path.exists(path)
            except:
                return False
    except Exception as e:
        st.warning(f"Warnung beim Löschen von {path}: {str(e)}")
        return False

def auto_initialize_system():
    """Automatische System-Initialisierung beim Start."""
    try:
        # Prüfe auf Änderungen
        updater = DatabaseAutoUpdater()
        should_rebuild, reason = updater.should_rebuild_database()
        
        if should_rebuild:
            st.info(f"🔄 **Auto-Update:** {reason}")
            
            # Robuste Database-Löschung
            with st.spinner("🔄 Aktualisiere Database automatisch..."):
                chroma_path = "./chroma"
                
                if os.path.exists(chroma_path):
                    st.info("🗑️ Lösche alte Vector Database...")
                    
                    # Versuche sicheres Löschen
                    success = safe_remove_directory(chroma_path)
                    
                    if not success:
                        # Fallback: Verzeichnis umbenennen und später löschen
                        backup_path = f"./chroma_backup_{int(time.time())}"
                        try:
                            os.rename(chroma_path, backup_path)
                            st.warning(f"Database wurde nach {backup_path} verschoben")
                        except:
                            st.error("Konnte alte Database nicht entfernen. Erstelle neue in anderem Pfad.")
                            # Verwende alternativen Pfad
                            chroma_path = f"./chroma_{int(time.time())}"
                
                # Cache leeren
                st.cache_resource.clear()
                
                # Session State zurücksetzen
                for key in ["rag_system", "monitor", "diagnosis"]:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Metadaten aktualisieren
                updater.update_after_rebuild()
            
            st.success("✅ Database automatisch aktualisiert!")
            time.sleep(1)
        
        # Lade System automatisch
        if "rag_system" not in st.session_state:
            with st.spinner("🚀 Initialisiere RAG-System..."):
                st.session_state.monitor = RAGMonitor()
                st.session_state.monitor.log_system_event("System initialization started")
                
                # Verwende Standard-Konfiguration
                qa_chain, translator = initialize_technical_rag_system(language="de", max_chunks=100)
                st.session_state.rag_system = (qa_chain, translator)
                st.session_state.monitor.log_system_event("System initialization completed")
                
                st.success("✅ RAG-System bereit!")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Fehler bei der System-Initialisierung: {str(e)}")
        
        # Erweiterte Fehlerbeschreibung für Chroma-Probleme
        if "'Chroma' object has no attribute 'persist'" in str(e):
            st.markdown("""
            ### 🔧 Chroma-Kompatibilitätsproblem erkannt:
            
            **Problem:** Ihre Chroma-Version ist inkompatibel.
            
            **Lösung:** Aktualisieren Sie Chroma:
            ```bash
            pip install --upgrade chromadb langchain-chroma
            ```
            
            **Alternative:** Löschen Sie den ./chroma Ordner und starten neu.
            """)
        else:
            st.markdown("""
            ### 🔧 Mögliche Lösungen:
            
            1. **Als Administrator ausführen**: Starten Sie die Anwendung als Administrator
            2. **Chroma aktualisieren**: `pip install --upgrade chromadb langchain-chroma`
            3. **Database-Ordner löschen**: Verwenden Sie den Notfall-Button unten
            4. **Antivirus prüfen**: Deaktivieren Sie temporär den Antivirus-Scanner
            """)
        
        # Notfall-Modus
        if st.button("🆘 Notfall-Modus: Database löschen und neu starten"):
            try:
                # Lösche Chroma komplett
                safe_remove_directory("./chroma")
                # Cache leeren
                st.cache_resource.clear()
                # Session State leeren
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("✅ Database gelöscht - Seite wird neu geladen...")
                time.sleep(2)
                st.rerun()
            except Exception as emergency_error:
                st.error(f"Notfall-Modus fehlgeschlagen: {str(emergency_error)}")
        
        return False

@st.cache_resource
def load_rag_system(language, max_chunks):
    """Cached System-Initialisierung."""
    return initialize_technical_rag_system(language=language, max_chunks=max_chunks)

def process_question(question, language="de", max_chunks=100, enable_translation=False):
    """Verarbeitet eine Frage und gibt das Ergebnis zurück."""
    try:
        # System automatisch initialisieren falls nicht vorhanden
        if "rag_system" not in st.session_state:
            st.session_state.monitor = RAGMonitor()
            qa_chain, translator = load_rag_system(language, max_chunks)
            st.session_state.rag_system = (qa_chain, translator)
        
        qa_chain, translator = st.session_state.rag_system
        monitor = st.session_state.monitor
        
        # Frage stellen
        result = ask_technical_question(
            qa_chain, question, translator, 
            translate_query=enable_translation,
            target_language=language
        )
        
        # Monitoring
        monitor.log_query(
            question, 
            result["answer"], 
            result["response_time"], 
            result["source_count"]
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
                          'display', 'monitor', 'processor', 'memory', 'storage', 
                          'specification', 'datenblatt', 'technical', 'model']
        for indicator in tech_indicators:
            if indicator in content_lower:
                bonus_score += 0.1
        
        # Bonus für Spezifikationen (Zahlen + Einheiten)
        spec_patterns = [r'\d+\s*(gb|mb|ghz|mhz|watt|inch|zoll)', 
                        r'\d+x\d+', r'usb\s*[0-9c]', r'\d+\s*port',
                        r'cpu|processor|ram|memory|storage|display']
        for pattern in spec_patterns:
            if re.search(pattern, content_lower):
                bonus_score += 0.15
        
        # Finale Relevanz-Bewertung
        final_relevance = min(1.0, relevance + bonus_score)
        source['final_relevance'] = final_relevance
        
        # Nur relevante Sources anzeigen (niedrigere Schwelle)
        if final_relevance > 0.1 or len(content) > 50:
            relevant_sources.append(source)
    
    # Sortiere nach finaler Relevanz
    relevant_sources.sort(key=lambda x: x.get('final_relevance', 0), reverse=True)
    
    if not relevant_sources:
        st.warning("Keine relevanten Quellen gefunden für diese Frage.")
        relevant_sources = sources[:2]
    
    st.caption(f"📊 {len(relevant_sources)} relevante Quellen gefunden (von {len(sources)} insgesamt)")
    
    # Zeige nur die relevantesten Quellen
    for i, source in enumerate(relevant_sources[:5]):
        relevance = source.get('final_relevance', 0)
        
        # Relevanz-Kategorien
        if relevance > 0.4:
            indicator = "🔴"
            category = "Sehr relevant"
        elif relevance > 0.2:
            indicator = "🟡"
            category = "Relevant"
        else:
            indicator = "🟢"
            category = "Grundinformation"
        
        source_name = source['metadata'].get('source', 'Unbekannt')
        page_num = source['metadata'].get('page', 'N/A')
        
        with st.expander(f"{indicator} **{category}** - {source_name} (Seite {page_num}) - {relevance:.0%}", expanded=(i==0 and relevance > 0.3)):
            
            # Metadaten in kompakter Form
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                extraction_method = source['metadata'].get('extraction_method', 'Standard')
                st.metric("Extraction", extraction_method.title())
            with col2:
                st.metric("Relevanz", f"{relevance:.0%}")
            with col3:
                st.metric("Zeichen", len(source['content']))
            with col4:
                chunk_id = source['metadata'].get('chunk_id', i + 1)
                st.metric("Chunk", chunk_id)
            
            # Content display
            tabs = st.tabs(["🔍 Relevanter Inhalt", "📝 Volltext"])
            
            with tabs[0]:
                st.markdown("**Gefundene relevante Passage:**")
                
                if 'highlighted_content' in source and source['highlighted_content'].strip():
                    highlighted = source['highlighted_content']
                    st.markdown(highlighted)
                else:
                    content = source['content']
                    if key_terms:
                        best_snippet = find_best_snippet(content, key_terms)
                        st.markdown(best_snippet)
                    else:
                        preview = content[:400] + "..." if len(content) > 400 else content
                        st.markdown(preview)
                
                if key_terms:
                    found_terms = [term for term in key_terms if term.lower() in source['content'].lower()]
                    if found_terms:
                        st.markdown(f"**🔍 Gefundene Begriffe:** {', '.join(found_terms)}")
            
            with tabs[1]:
                st.markdown("**Vollständiger Textinhalt:**")
                st.text_area(
                    "Volltext", 
                    source['content'], 
                    height=200, 
                    key=f"full_content_{i}_{hash(source['content'][:50])}",
                    help="Vollständiger Text dieses Dokument-Chunks"
                )

def find_best_snippet(text: str, key_terms: List[str], snippet_length: int = 300) -> str:
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
    
    # Automatische System-Initialisierung beim ersten Start
    if "system_initialized" not in st.session_state:
        with st.container():
            st.subheader("🚀 System wird initialisiert...")
            if auto_initialize_system():
                st.session_state.system_initialized = True
                st.success("✅ System bereit!")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ System-Initialisierung fehlgeschlagen")
                st.stop()
    
    # Einfache Sidebar nur mit Status
    with st.sidebar:
        st.header("📊 System Status")
        
        # System Status
        if os.path.exists("./data"):
            pdf_files = [f for f in os.listdir("./data") if f.endswith('.pdf')]
            st.success(f"✅ {len(pdf_files)} PDF-Dateien gefunden")
        else:
            st.error("❌ ./data Ordner nicht gefunden!")
        
        # Flexiblere Chroma-Erkennung
        chroma_paths = ["./chroma", f"./chroma_{int(time.time()/86400)}"]  # Alternative Pfade
        chroma_found = False
        for path in chroma_paths:
            if os.path.exists(path):
                st.success(f"✅ Vector Database aktiv ({path})")
                chroma_found = True
                break
        
        if not chroma_found:
            st.warning("⚠️ Vector Database wird erstellt...")
        
        if os.path.exists(".env"):
            st.success("✅ Konfiguration geladen")
        else:
            st.error("❌ .env Datei fehlt!")
        
        # Statistiken
        if "monitor" in st.session_state:
            monitor = st.session_state.monitor
            stats = monitor.get_statistics()
            
            if stats.get("total_queries", 0) > 0:
                st.subheader("📈 Statistiken")
                st.metric("Anfragen", stats.get("total_queries", 0))
                if stats.get("avg_response_time", 0) > 0:
                    st.metric("Ø Antwortzeit", f"{stats['avg_response_time']:.2f}s")
                if stats.get("avg_source_count", 0) > 0:
                    st.metric("Ø Quellen", f"{stats['avg_source_count']:.1f}")
        
        st.divider()
        
        # Nur minimale Aktionen + Notfall-Funktionen
        if st.button("🗑️ Chat löschen", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 System neu starten", use_container_width=True):
            st.cache_resource.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Notfall-Sektion
        with st.expander("🆘 Notfall-Funktionen"):
            st.markdown("**Bei Problemen:**")
            
            if st.button("🔧 Database-Ordner bereinigen", use_container_width=True):
                with st.spinner("Bereinige Database-Ordner..."):
                    success = safe_remove_directory("./chroma")
                    if success:
                        st.success("✅ Database-Ordner bereinigt!")
                    else:
                        st.warning("⚠️ Vollständige Bereinigung nicht möglich")
                    st.rerun()
            
            if st.button("📁 Als Administrator starten", use_container_width=True):
                st.info("Starten Sie die Anwendung als Administrator neu!")
                st.code("Rechtsklick auf cmd/PowerShell → 'Als Administrator ausführen'")

    # Hauptbereich - 2-spaltig
    main_col, example_col = st.columns([3, 1])
    
    with main_col:
        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Chat History
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "metadata" in message:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.caption(f"📚 {message['metadata']['source_count']} Quellen")
                    with col_b:
                        st.caption(f"⏱️ {message['metadata'].get('response_time', 0):.2f}s")
                    with col_c:
                        if st.button("📋 Quellen anzeigen", key=f"sources_{i}"):
                            st.session_state.selected_message_sources = message['metadata'].get('sources', [])
                            st.session_state.selected_message_terms = message['metadata'].get('key_terms', [])
                            st.session_state.show_source_preview = True

        # Zeige Quellenvorschau wenn ausgewählt
        if st.session_state.get('show_source_preview', False):
            sources = st.session_state.get('selected_message_sources', [])
            key_terms = st.session_state.get('selected_message_terms', [])
            
            show_document_preview(sources, key_terms)
            
            if st.button("❌ Vorschau schließen"):
                st.session_state.show_source_preview = False
                st.rerun()
        
        # Prüfe auf neue Fragen (von Beispiel-Buttons)
        if "pending_question" in st.session_state:
            question = st.session_state.pending_question
            del st.session_state.pending_question
            
            # User Message hinzufügen
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Analysiere technische Daten..."):
                    result = process_question(question)
                    
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
                            "sources": result.get('sources', []),
                            "key_terms": result.get('key_terms', [])
                        }
                    })
            
            st.rerun()

    with example_col:
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
            ],
            "🔍 Diagnose": [
                "Auf welche Datenblätter hast du Zugriff?",
                "Welche Produktmodelle sind verfügbar?",
                "Zeige mir alle verfügbaren Dokumente"
            ]
        }
        
        for category, questions in example_categories.items():
            with st.expander(category):
                for i, question in enumerate(questions):
                    if st.button(question, key=f"example_{category}_{i}", use_container_width=True):
                        st.session_state.pending_question = question
                        st.rerun()
        
    # Chat Input
    st.divider()
    
    if prompt := st.chat_input("Technische Frage stellen..."):
        # User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Assistant Response
        with st.spinner("🔍 Analysiere technische Daten..."):
            result = process_question(prompt)
            
            # Message speichern
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "metadata": {
                    "source_count": result['source_count'],
                    "response_time": result['response_time'],
                    "sources": result.get('sources', []),
                    "key_terms": result.get('key_terms', [])
                }
            })
        
        st.rerun()

if __name__ == "__main__":
    main()
