import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import config

logger = logging.getLogger(__name__)

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFaceEmbeddings model
    configured in config.py.
    """
    logger.info(f"Initializing embedding model: {config.EMBEDDING_MODEL_NAME}")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},  # Default to CPU for maximum compatibility
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        raise

def build_vectorstore(chunks: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Builds a FAISS vector database from a list of Document chunks and embeddings.
    """
    logger.info(f"Building FAISS vector database with {len(chunks)} chunks...")
    try:
        db = FAISS.from_documents(chunks, embeddings)
        logger.info("FAISS vector database built successfully.")
        return db
    except Exception as e:
        logger.error(f"Error building FAISS vector database: {e}")
        raise

def save_vectorstore(db: FAISS, path: Path) -> None:
    """
    Persists the FAISS index to the local filesystem.
    """
    logger.info(f"Saving FAISS index to directory: {path}")
    try:
        path.parent.mkdir(exist_ok=True, parents=True)
        db.save_local(str(path))
        logger.info("FAISS index saved successfully.")
    except Exception as e:
        logger.error(f"Error saving FAISS database: {e}")
        raise

def load_vectorstore(path: Path, embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Loads a persisted FAISS index from the filesystem.
    """
    logger.info(f"Loading FAISS index from: {path}")
    try:
        if not path.exists():
            raise FileNotFoundError(f"FAISS index folder {path} does not exist.")
        db = FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully.")
        return db
    except Exception as e:
        logger.error(f"Error loading FAISS database: {e}")
        raise

def get_retriever(db: FAISS):
    """
    Returns an MMR-based retriever configured with search settings from config.py.
    This guarantees source document diversity and factual coverage.
    """
    logger.info(f"Configuring retriever with search_type={config.SEARCH_TYPE}, k={config.RETRIEVER_K}, fetch_k={config.RETRIEVER_FETCH_K}")
    retriever = db.as_retriever(
        search_type=config.SEARCH_TYPE,
        search_kwargs={
            "k": config.RETRIEVER_K,
            "fetch_k": config.RETRIEVER_FETCH_K
        }
    )
    return retriever
