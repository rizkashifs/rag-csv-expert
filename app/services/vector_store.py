from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings
import os

def create_vector_store(texts: list, index_name: str = "default"):
    """
    Creates a FAISS vector store from a list of texts.
    """
    embeddings = OllamaEmbeddings(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL
    )
    
    vector_store = FAISS.from_texts(texts, embeddings)
    
    # Save the index
    save_path = os.path.join(settings.FAISS_INDEX_PATH, index_name)
    os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
    vector_store.save_local(save_path)
    
    return vector_store

def load_vector_store(index_name: str = "default"):
    """
    Loads a FAISS vector store from disk.
    """
    embeddings = OllamaEmbeddings(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL
    )
    
    save_path = os.path.join(settings.FAISS_INDEX_PATH, index_name)
    if not os.path.exists(save_path):
        return None
        
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

def search_vector_store(query: str, index_name: str = "default", k: int = 5):
    """
    Searches the vector store for relevant documents.
    """
    vector_store = load_vector_store(index_name)
    if not vector_store:
        return []
        
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
