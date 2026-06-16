import re
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Cleans extracted PDF text to remove excessive whitespaces,
    unusual page-break symbols, and normalize newlines.
    """
    if not text:
        return ""
    
    # Replace null bytes
    text = text.replace("\x00", "")
    
    # Normalize whitespaces: convert tabs and multiple consecutive spaces to a single space
    text = re.sub(r"[ \t]+", " ", text)
    
    # Normalize multiple newlines: collapse multiple consecutive newlines into a max of two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Strip leading/trailing whitespace
    return text.strip()

def load_pdf_document(file_path: Path) -> List[Document]:
    """
    Loads a single PDF document using PyPDFLoader and cleans the text on each page.
    Preserves metadata like filename and page number.
    """
    logger.info(f"Loading document: {file_path.name}")
    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        
        cleaned_pages = []
        for page in pages:
            # Extract basic metadata
            source_name = file_path.name
            # PyPDFLoader page numbers are typically 0-indexed. We store it directly.
            page_num = page.metadata.get("page", 0)
            
            # Clean the text
            cleaned_content = clean_text(page.page_content)
            
            # Reconstruct the document with clean text and strictly preserved metadata
            cleaned_doc = Document(
                page_content=cleaned_content,
                metadata={
                    "source": source_name,
                    "page": page_num + 1 # Convert to 1-indexed for user friendly display
                }
            )
            cleaned_pages.append(cleaned_doc)
            
        logger.info(f"Successfully loaded {len(cleaned_pages)} pages from {file_path.name}")
        return cleaned_pages
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

def load_all_policy_documents(data_dir: Path) -> List[Document]:
    """
    Scans the given data directory and loads all PDF files.
    """
    if not data_dir.exists():
        logger.warning(f"Data directory {data_dir} does not exist.")
        return []
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}.")
        return []
    
    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}.")
    all_documents = []
    
    for pdf_file in pdf_files:
        docs = load_pdf_document(pdf_file)
        all_documents.extend(docs)
        
    logger.info(f"Loaded a total of {len(all_documents)} raw page documents.")
    return all_documents
