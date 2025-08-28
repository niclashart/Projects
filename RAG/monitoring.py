import logging
import json
from datetime import datetime
from pathlib import Path

class RAGMonitor:
    """Monitoring und Logging für das RAG-System."""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup Logging
        self.setup_logging()
        
        # Metriken
        self.query_log = []
        
    def setup_logging(self):
        """Konfiguriert detailliertes Logging."""
        log_file = self.log_dir / f"rag_system_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def log_query(self, question: str, answer: str, response_time: float, source_count: int):
        """Loggt Benutzeranfragen für Analyse."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "response_time": response_time,
            "source_count": source_count
        }
        
        self.query_log.append(log_entry)
        
        # Speichere regelmäßig
        if len(self.query_log) % 10 == 0:
            self.save_query_log()
    
    def save_query_log(self):
        """Speichert Query-Log als JSON."""
        log_file = self.log_dir / f"queries_{datetime.now().strftime('%Y%m%d')}.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.query_log, f, ensure_ascii=False, indent=2)