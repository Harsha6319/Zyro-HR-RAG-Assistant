import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# Configuration for Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Configuration for Vector Store & Embeddings
# Pinned to a single, easily configurable location
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Configuration for Retriever (MMR settings)
SEARCH_TYPE = "mmr"
RETRIEVER_K = 4
RETRIEVER_FETCH_K = 20

# Configuration for LLMs
# Llama-3.3-70b-versatile is recommended for high quality reasoning/answering on Groq
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
# Llama-3.1-8b-instant or Llama3-8b-8192 is used for fast and cheap classification/guardrails
GROQ_CLASSIFIER_MODEL = "llama-3.1-8b-instant"

# Grounding temperature should be 0.0 for deterministic factual grounding
LLM_TEMPERATURE = 0.0

# Kaggle challenge evaluation files configuration
EVAL_QUESTIONS_FILE = DATA_DIR / "evaluation_questions.csv" # Adjust based on actual downloaded competition files
SUBMISSION_FILE = BASE_DIR / "submission.csv"

# Streamlit & LangSmith submission configuration URLs
STREAMLIT_APP_URL = os.getenv("STREAMLIT_APP_URL", "https://share.streamlit.io/yourusername/zyro-hr-assistant/app.py")
LANGSMITH_PROJECT_URL = os.getenv("LANGSMITH_PROJECT_URL", "https://smith.langchain.com/o/arm/projects/p/zyro-rag-challenge")
