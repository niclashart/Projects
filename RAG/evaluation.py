import time
from typing import List, Dict
import json
from datetime import datetime

# Test-Fragen für technische Evaluation
TECHNICAL_TEST_QUESTIONS = [
    "Welche CPU-Modelle sind in den Workstations verfügbar?",
    "Wie viel RAM unterstützen die Thin Clients maximal?",
    "Welche Anschlüsse haben die Monitore?",
    "Was ist die maximale Bildschirmauflösung?",
    "Welche Betriebssysteme werden unterstützt?",
    "Vergleiche die Stromverbrauchswerte der Geräte"
]

class RAGEvaluator:
    """Evaluierung des RAG-Systems für technische Anfragen."""
    
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain
        self.evaluation_results = []
    
    def evaluate_response_time(self, question: str) -> Dict:
        """Misst Antwortzeit und Qualität."""
        start_time = time.time()
        result = self.qa_chain({"query": question})
        end_time = time.time()
        
        response_time = end_time - start_time
        
        return {
            "question": question,
            "answer": result["result"],
            "response_time": response_time,
            "source_count": len(result["source_documents"]),
            "timestamp": datetime.now().isoformat()
        }
    
    def run_benchmark(self, test_questions: List[str] = None) -> Dict:
        """Führt Benchmark mit Test-Fragen durch."""
        if test_questions is None:
            test_questions = TECHNICAL_TEST_QUESTIONS
            
        results = []
        
        for question in test_questions:
            print(f"Evaluiere: {question}")
            result = self.evaluate_response_time(question)
            results.append(result)
            
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        
        return {
            "results": results,
            "average_response_time": avg_response_time,
            "total_questions": len(test_questions),
            "evaluation_date": datetime.now().isoformat()
        }
    
    def save_evaluation(self, results: Dict, filename: str = None):
        """Speichert Evaluationsergebnisse."""
        if filename is None:
            filename = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)