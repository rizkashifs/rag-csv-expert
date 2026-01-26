from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi"
    FAISS_INDEX_PATH: str = "data/faiss_index"
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"
    EMBEDDING_PROVIDER: str = "huggingface" # Options: ollama, huggingface
    HUGGINGFACE_MODEL: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"

settings = Settings()
