import os
import logging
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class DirectHuggingFaceEmbeddings:
    """Wrapper to make SentenceTransformer compatible with LangChain VectorStore"""
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode(text).tolist()
    def __call__(self, text): # Required by some langchain versions
        return self.embed_query(text)

class VectorEngine:
    """
    Data engine for semantic search operations.
    """
    def __init__(self):
        if settings.EMBEDDING_PROVIDER == "huggingface":
            logger.info(f"Initializing direct SentenceTransformer: {settings.HUGGINGFACE_MODEL}")
            self.embeddings = DirectHuggingFaceEmbeddings(settings.HUGGINGFACE_MODEL)
        else:
            logger.info(f"Initializing Ollama embeddings: {settings.OLLAMA_MODEL}")
            self.embeddings = OllamaEmbeddings(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL
            )

    def search(self, query: str, index_name: str, k: int = 5) -> List[str]:
        """
        Performs similarity search on a specific index.
        """
        save_path = os.path.join(settings.FAISS_INDEX_PATH, index_name)
        if not os.path.exists(save_path):
            logger.warning(f"Index not found at {save_path}")
            return []
            
        logger.info(f"Loading local index from {save_path}...")
        vector_store = FAISS.load_local(
            save_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info(f"Performing similarity search for: {query[:50]}...")
        docs = vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def create_index(self, texts: List[str], index_name: str):
        """
        Creates and saves a new FAISS index.
        """
        try:
            logger.info(f"Creating vector index for {len(texts)} chunks...")
            vector_store = FAISS.from_texts(texts, self.embeddings)
            save_path = os.path.join(settings.FAISS_INDEX_PATH, index_name)
            os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
            logger.info(f"Saving index to {save_path}...")
            vector_store.save_local(save_path)
            return True
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            raise ConnectionError(f"Embedding/Vector store error: {e}")

# Singleton
vector_engine = VectorEngine()
