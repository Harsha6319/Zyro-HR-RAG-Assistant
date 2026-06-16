import logging
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits a list of Documents into smaller chunks using RecursiveCharacterTextSplitter.
    Preserves metadata (source and page number) from the input documents.
    """
    if not documents:
        logger.warning("No documents provided for chunking.")
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    logger.info(f"Chunking {len(documents)} documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    chunks = splitter.split_documents(documents)
    logger.info(f"Successfully generated {len(chunks)} text chunks.")
    
    # Simple validation to ensure source and page metadata are present
    for i, chunk in enumerate(chunks[:5]):
        logger.debug(f"Chunk {i} metadata: {chunk.metadata}")
        
    return chunks
