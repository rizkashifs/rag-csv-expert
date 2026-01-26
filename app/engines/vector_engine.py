from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings
import os
from typing import List

class VectorEngine:
    """
    Data engine for semantic search operations.
    """
    def __init__(self):
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
            return []
            
        vector_store = FAISS.load_local(
            save_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        docs = vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def create_index(self, texts: List[str], index_name: str):
        """
        Creates and saves a new FAISS index.
        """
        vector_store = FAISS.from_texts(texts, self.embeddings)
        save_path = os.path.join(settings.FAISS_INDEX_PATH, index_name)
        os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
        vector_store.save_local(save_path)

# Singleton
vector_engine = VectorEngine()
