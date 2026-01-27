from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi"
    FAISS_INDEX_PATH: str = "data/faiss_index"
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-3-haiku-20240307"
    EMBEDDING_PROVIDER: str = "huggingface" # Options: ollama, huggingface
    HUGGINGFACE_MODEL: str = "all-MiniLM-L6-v2"
    LLM_PROVIDER: str = "ollama" # Options: ollama, anthropic, bedrock
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"
    AWS_REGION: str = "us-east-1"
    
    class Config:
        env_file = ".env"

settings = Settings()
