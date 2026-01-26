from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    FAISS_INDEX_PATH: str = "data/faiss_index"
    
    class Config:
        env_file = ".env"

settings = Settings()
