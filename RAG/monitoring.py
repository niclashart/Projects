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
        self.logger = self.setup_logging()
        self.logger = self.setup_logging()
        
        # Metriken - erweiterte Tracking-Variablen
        # Metriken - erweiterte Tracking-Variablen
        self.query_log = []
        self.total_response_time = 0.0
        self.total_queries = 0
        self.total_sources = 0
        self.total_response_time = 0.0
        self.total_queries = 0
        self.total_sources = 0
        
    def setup_logging(self):
        """Konfiguriert detailliertes Logging."""
        log_file = self.log_dir / f"rag_system_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Erstelle einen spezifischen Logger für RAG
        logger = logging.getLogger('rag_system')
        logger.setLevel(logging.INFO)
        
        # Entferne alle existierenden Handler
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File Handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Handler hinzufügen
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Verhindere, dass Logs an den Root Logger weitergegeben werden
        logger.propagate = False
        
        return logger
        # Erstelle einen spezifischen Logger für RAG
        logger = logging.getLogger('rag_system')
        logger.setLevel(logging.INFO)
        
        # Entferne alle existierenden Handler
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File Handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Handler hinzufügen
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Verhindere, dass Logs an den Root Logger weitergegeben werden
        logger.propagate = False
        
        return logger
    
    def log_query(self, question: str, answer: str, response_time: float, source_count: int):
        """Loggt Benutzeranfragen für Analyse."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer[:500] + "..." if len(answer) > 500 else answer,
            "response_time": response_time,
            "source_count": source_count
        }
        
        self.query_log.append(log_entry)
        
        # Aktualisiere die laufenden Totale
        self.total_queries += 1
        self.total_response_time += response_time
        self.total_sources += source_count
        
        # Logge die Query
        self.logger.info(f"Query processed - Question: '{question}' - Sources: {source_count} - Time: {response_time:.2f}s")
        
        # Speichere regelmäßig
        if len(self.query_log) % 5 == 0:
            self.save_query_log()
    
    def get_statistics(self):
        """Gibt korrekte Statistiken über das System zurück."""
        if self.total_queries == 0:
            return {
                "total_queries": 0,
                "avg_response_time": 0.0,
                "avg_source_count": 0.0,
                "min_response_time": 0.0,
                "max_response_time": 0.0,
                "total_response_time": 0.0
            }
        
        # Berechne Statistiken aus den gespeicherten Queries
        response_times = [entry["response_time"] for entry in self.query_log]
        source_counts = [entry["source_count"] for entry in self.query_log]
        
        # Korrekte Durchschnittswerte
        avg_response_time = self.total_response_time / self.total_queries
        avg_source_count = self.total_sources / self.total_queries
        
        stats = {
            "total_queries": self.total_queries,
            "avg_response_time": avg_response_time,
            "avg_source_count": avg_source_count,
            "min_response_time": min(response_times) if response_times else 0.0,
            "max_response_time": max(response_times) if response_times else 0.0,
            "total_response_time": self.total_response_time,
            "total_sources": self.total_sources
        }
        
        self.logger.info(f"Statistics requested - Total queries: {self.total_queries}, Avg response time: {avg_response_time:.2f}s, Avg sources: {avg_source_count:.1f}")
        
        return stats

    def get_detailed_statistics(self):
        """Gibt detaillierte Statistiken zurück."""
        if not self.query_log:
            return {
                "message": "Keine Queries vorhanden",
                "basic_stats": self.get_statistics()
            }
        
        response_times = [entry["response_time"] for entry in self.query_log]
        source_counts = [entry["source_count"] for entry in self.query_log]
        
        # Zeitbasierte Analyse
        recent_queries = self.query_log[-10:] if len(self.query_log) >= 10 else self.query_log
        recent_avg_time = sum(q["response_time"] for q in recent_queries) / len(recent_queries) if recent_queries else 0
        
        # Performance-Analyse
        fast_queries = [q for q in self.query_log if q["response_time"] < 2.0]
        slow_queries = [q for q in self.query_log if q["response_time"] > 5.0]
        
        return {
            "basic_stats": self.get_statistics(),
            "response_time_analysis": {
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "median": sorted(response_times)[len(response_times)//2] if response_times else 0,
                "recent_avg_10": recent_avg_time,
                "fast_queries_count": len(fast_queries),
                "slow_queries_count": len(slow_queries)
            },
            "source_analysis": {
                "min_sources": min(source_counts) if source_counts else 0,
                "max_sources": max(source_counts) if source_counts else 0,
                "avg_sources": sum(source_counts) / len(source_counts) if source_counts else 0,
                "zero_source_queries": len([q for q in self.query_log if q["source_count"] == 0])
            },
            "query_patterns": {
                "total_queries": len(self.query_log),
                "recent_queries": len(recent_queries),
                "questions_with_sources": len([q for q in self.query_log if q["source_count"] > 0]),
                "avg_question_length": sum(len(q["question"]) for q in self.query_log) / len(self.query_log) if self.query_log else 0
            },
            "performance_distribution": {
                "fast_queries_pct": len(fast_queries) / len(self.query_log) * 100 if self.query_log else 0,
                "slow_queries_pct": len(slow_queries) / len(self.query_log) * 100 if self.query_log else 0,
                "normal_queries_pct": (len(self.query_log) - len(fast_queries) - len(slow_queries)) / len(self.query_log) * 100 if self.query_log else 0
            }
        }

    def save_query_log(self):
        """Speichert Query-Log als JSON."""
        try:
            log_file = self.log_dir / f"queries_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Erweitere die Daten mit Statistiken
            log_data = {
                "queries": self.query_log,
                "statistics": self.get_statistics(),
                "detailed_statistics": self.get_detailed_statistics(),
                "saved_at": datetime.now().isoformat()
            }
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Query log saved to {log_file}")
        except Exception as e:
            self.logger.error(f"Failed to save query log: {str(e)}")
    
    def log_system_event(self, event: str, details: str = ""):
        """Loggt System-Events."""
        self.logger.info(f"System Event: {event} - {details}")
    
    def log_error(self, error: str, details: str = ""):
        """Loggt Fehler."""
        self.logger.error(f"Error: {error} - {details}")
    
    def reset_statistics(self):
        """Setzt alle Statistiken zurück."""
        self.query_log = []
        self.total_response_time = 0.0
        self.total_queries = 0
        self.total_sources = 0
        self.logger.info("Statistics reset")
    
    def get_recent_queries(self, count: int = 5):
        """Gibt die letzten N Queries zurück."""
        return self.query_log[-count:] if len(self.query_log) >= count else self.query_log
    
    def get_performance_summary(self):
        """Gibt eine Performance-Zusammenfassung zurück."""
        if not self.query_log:
            return "Keine Daten verfügbar"
        
        stats = self.get_statistics()
        recent = self.get_recent_queries(5)
        
        summary = f"""
        📊 Performance Summary:
        • Total Queries: {stats['total_queries']}
        • Avg Response Time: {stats['avg_response_time']:.2f}s
        • Avg Sources: {stats['avg_source_count']:.1f}
        • Fastest Query: {stats['min_response_time']:.2f}s
        • Slowest Query: {stats['max_response_time']:.2f}s
        
        🔄 Recent Performance (last 5):
        """
        
        for i, query in enumerate(recent, 1):
            summary += f"\n  {i}. {query['response_time']:.2f}s - {query['source_count']} sources"
        
        return summary
