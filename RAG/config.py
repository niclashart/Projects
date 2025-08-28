import os
from dotenv import load_dotenv

# .env Datei laden
load_dotenv()

class RAGConfig:
    """Konfiguration für das RAG-System."""
    
    def __init__(self):
        # OpenAI API
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm_model = "gpt-3.5-turbo"
        self.temperature = 0.1
        
        # Embedding Model
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Text Splitting
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Vector Store
        self.vector_store_path = "./chroma"
        
        # Daten
        self.data_path = "./data"
        
        # Retrieval
        self.retrieval_k = 5
        
        # Validierung
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY nicht gefunden in .env Datei")
