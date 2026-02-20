import os
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from app.utils.logger import logger

class DirectHuggingFaceEmbeddings:
    """Wrapper to make SentenceTransformer compatible with LangChain VectorStore"""
    def __init__(self, model_name):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not installed. "
                "Please run `pip install sentence-transformers` to use huggingface embeddings."
            )
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

    def search(self, query: str, index_name: str, k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Performs similarity search on a specific index.
        Returns result consistent with SQL engine structure: {"relevant_rows": [...]}
        """
        save_path = os.path.join(settings.FAISS_INDEX_PATH, index_name)
        if not os.path.exists(save_path):
            logger.warning(f"Index not found at {save_path}")
            return {"relevant_rows": []}
            
        logger.info(f"Loading local index from {save_path}...")
        vector_store = FAISS.load_local(
            save_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info(f"Performing similarity search for: {query[:50]}...")
        
        # Use similarity_search_with_score to get distance metrics
        docs_and_scores: List[Tuple[Any, float]] = vector_store.similarity_search_with_score(query, k=k)
        
        unique_results = []
        seen_content = set()
        
        for doc, score in docs_and_scores:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                unique_results.append({
                    "content": content,
                    "vector_score": float(score),
                    "_summary": f"Semantic Match (Score: {score:.4f}): {content[:100]}..."
                })
        
        return {"relevant_rows": unique_results}

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
