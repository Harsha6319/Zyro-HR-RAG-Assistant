import time
import logging
from dotenv import load_dotenv
from utils.document_loader import load_all_policy_documents
from utils.chunking import split_documents
from utils.retrieval import get_embeddings, build_vectorstore, save_vectorstore
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ingest")

def main():
    logger.info("Starting Zyro HR Assistant Ingestion Pipeline...")
    start_time = time.time()
    
    # Load environment variables
    load_dotenv()
    
    # 1. Load and clean documents
    logger.info("Phase 1: Loading PDF documents...")
    documents = load_all_policy_documents(config.DATA_DIR)
    if not documents:
        logger.error(
            f"No documents were loaded from '{config.DATA_DIR}'. "
            "Please ensure you run 'python download_data.py' first or add PDF policy documents to the 'data/' folder."
        )
        return
        
    # 2. Chunk documents
    logger.info("Phase 2: Chunking documents...")
    chunks = split_documents(
        documents,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    if not chunks:
        logger.error("No document chunks created. Aborting ingestion.")
        return
        
    # 3. Get embedding model
    logger.info("Phase 3: Initializing local embedding model...")
    try:
        embeddings = get_embeddings()
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        return
        
    # 4. Build vector database
    logger.info("Phase 4: Building FAISS vector store...")
    try:
        db = build_vectorstore(chunks, embeddings)
    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        return
        
    # 5. Persist FAISS database
    logger.info("Phase 5: Saving FAISS index locally...")
    try:
        save_vectorstore(db, config.VECTORSTORE_DIR)
    except Exception as e:
        logger.error(f"Failed to save vector store: {e}")
        return
        
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info("=========================================================")
    logger.info("Ingestion pipeline completed successfully!")
    logger.info(f"Total chunks indexed: {len(chunks)}")
    logger.info(f"Database saved to: {config.VECTORSTORE_DIR}")
    logger.info(f"Total time taken: {elapsed:.2f} seconds")
    logger.info("=========================================================")

if __name__ == "__main__":
    main()
