import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class RAGConfig:
    # LLM Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    llm_model: str = "gpt-5-mini"
    temperature: float = 0.1
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Text Splitting Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Vector Store Configuration
    vector_store_path: str = "./chroma"
    
    # Auto-update Configuration - NEU
    auto_rebuild_enabled: bool = True
    check_for_new_files: bool = True
    metadata_file: str = "./data/.metadata.json"
    
    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY muss in der .env Datei gesetzt sein")
