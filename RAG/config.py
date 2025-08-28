import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGConfig:
    """Konfiguration für das RAG-System."""
    
    # Pfade
    data_path: str = "./data"
    chroma_path: str = "./chroma"
    logs_path: str = "./logs"
    
    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 80
    max_chunks: Optional[int] = 100
    
    # Retrieval
    retrieval_k: int = 5
    retrieval_type: str = "mmr"  # similarity, mmr
    
    # LLM
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_openai_embeddings: bool = False
    
    # Sprachen
    default_language: str = "de"
    enable_translation: bool = True
    
    # OCR
    ocr_languages: str = "deu+eng"
    ocr_dpi: int = 300
    
    @classmethod
    def from_env(cls):
        """Lädt Konfiguration aus Umgebungsvariablen."""
        return cls(
            chunk_size=int(os.getenv("CHUNK_SIZE", 800)),
            max_chunks=int(os.getenv("MAX_CHUNKS", 100)) if os.getenv("MAX_CHUNKS") else None,
            model_name=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
            use_openai_embeddings=os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
        )