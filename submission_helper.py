import os
import pandas as pd
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

from utils.retrieval import get_embeddings, load_vectorstore, get_retriever
from guardrails import classify_query
from rag_chain import create_rag_chain
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("submission_helper")

# Placeholders for submission requirements
STREAMLIT_APP_URL = "https://share.streamlit.io/yourusername/zyro-hr-assistant/app.py"  # REPLACE WITH YOUR STREAMLIT DEPLOYMENT URL
LANGSMITH_TRACE_URL = "https://smith.langchain.com/o/arm/projects/p/zyro-rag-challenge" # REPLACE WITH YOUR LANGSMITH PROJECT URL

def generate_mock_evaluation_questions(file_path: Path) -> None:
    """
    Generates a mock evaluation questions CSV file if the official one is not yet downloaded.
    This enables immediate testing of the entire submission pipeline.
    """
    logger.info(f"Generating mock evaluation questions CSV at: {file_path}")
    questions = [
        ("Q01", "What is the work from home policy and who is eligible?"),
        ("Q02", "How many days of casual leave can an employee take in a year?"),
        ("Q03", "What is the code of conduct regarding conflicts of interest?"),
        ("Q04", "How is performance reviewed and what is the rating scale?"),
        ("Q05", "What are the core working hours of the company?"),
        ("Q06", "What constitutes a POSH violation and how do I report it?"),
        ("Q07", "What is the weather like in New York today?"),  # Out-of-scope test
        ("Q08", "What is the travel expense reimbursement limit?"),
        ("Q09", "How can I write a Python function to sort a list?"),  # Out-of-scope test
        ("Q10", "What are the data security practices for company laptops?"),
        ("Q11", "What is the separation notice period for employees?"),
        ("Q12", "What benefits are included in the health insurance policy?"),
        ("Q13", "Who won the football world cup in 2022?"),  # Out-of-scope test
        ("Q14", "What is the company's profile and founding year?"),
        ("Q15", "Can you explain the onboarding policy for new hires?")
    ]
    df = pd.DataFrame(questions, columns=["Question_ID", "Question"])
    df.to_csv(file_path, index=False)
    logger.info("Mock evaluation questions generated successfully.")

def main():
    load_dotenv()
    
    # 1. Load FAISS database and setup RAG chain
    logger.info("Initializing RAG pipeline for batch processing...")
    
    if not config.VECTORSTORE_DIR.exists():
        logger.error(
            f"FAISS index folder not found at '{config.VECTORSTORE_DIR}'. "
            "Please run 'python ingest.py' first to build the vector store."
        )
        return
        
    try:
        embeddings = get_embeddings()
        db = load_vectorstore(config.VECTORSTORE_DIR, embeddings)
        retriever = get_retriever(db)
        rag_chain = create_rag_chain(retriever)
    except Exception as e:
        logger.error(f"Error setting up RAG components: {e}")
        return

    # 2. Locate evaluation questions
    eval_file = config.EVAL_QUESTIONS_FILE
    if not eval_file.exists():
        logger.warning(f"Evaluation questions file not found at: {eval_file}")
        # Search for any CSV file in data/ first
        csv_files = list(config.DATA_DIR.glob("*.csv"))
        if csv_files:
            eval_file = csv_files[0]
            logger.info(f"Using found CSV file: {eval_file}")
        else:
            # Generate mock evaluation file if none is present
            generate_mock_evaluation_questions(eval_file)

    # 3. Read evaluation questions
    try:
        df_questions = pd.read_csv(eval_file)
        # Standardize column names
        df_questions.columns = [c.strip() for c in df_questions.columns]
        
        # Verify columns
        id_col = None
        q_col = None
        for col in df_questions.columns:
            if "id" in col.lower() or "q" in col.lower() and "question" not in col.lower():
                id_col = col
            if "question" in col.lower() or "query" in col.lower():
                q_col = col
                
        if not id_col:
            id_col = df_questions.columns[0]
            logger.warning(f"Could not identify ID column clearly. Using first column: '{id_col}'")
        if not q_col:
            q_col = df_questions.columns[1]
            logger.warning(f"Could not identify Question column clearly. Using second column: '{q_col}'")
            
        logger.info(f"Reading questions from columns: ID='{id_col}', Question='{q_col}'")
    except Exception as e:
        logger.error(f"Error reading evaluation questions: {e}")
        return

    # 4. Run pipeline on questions
    results = []
    logger.info(f"Processing {len(df_questions)} evaluation questions...")
    
    for idx, row in df_questions.iterrows():
        q_id = str(row[id_col]).strip()
        question = str(row[q_col]).strip()
        
        logger.info(f"Processing {q_id}: '{question[:50]}...'")
        start_t = time.time()
        
        try:
            # Apply guardrail classifier
            is_in_scope, guardrail_msg = classify_query(question)
            
            if not is_in_scope:
                answer = guardrail_msg
                logger.info(f"-> Guardrail triggered for {q_id}. Out of scope.")
            else:
                # Run through the RAG chain
                output = rag_chain.invoke(question)
                answer = output.get("answer", "No answer generated.")
                
                # Extract and format sources for citations verification
                docs = output.get("context", [])
                sources = sorted(list(set(doc.metadata.get("source", "Unknown Policy") for doc in docs)))
                if sources:
                    source_str = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)
                    # Check if the model already added sources, if not, append them
                    if "Sources:" not in answer:
                        answer += source_str
            
            # Record result
            results.append({
                "Question_ID": q_id,
                "Answer": answer
            })
            
            elapsed = time.time() - start_t
            logger.info(f"-> Completed {q_id} in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing question {q_id}: {e}")
            results.append({
                "Question_ID": q_id,
                "Answer": "The provided policy documents do not contain that information. (Execution Error)"
            })

    # 5. Output submission.csv
    df_submission = pd.DataFrame(results)
    
    # Save output
    df_submission.to_csv(config.SUBMISSION_FILE, index=False)
    
    logger.info("=========================================================")
    logger.info(f"Submission file created successfully at: {config.SUBMISSION_FILE}")
    logger.info(f"Total entries written: {len(df_submission)}")
    logger.info("=========================================================")
    logger.info("Metadata Requirements:")
    logger.info(f"STREAMLIT_APP_URL: {STREAMLIT_APP_URL}")
    logger.info(f"LANGSMITH_TRACE_URL: {LANGSMITH_TRACE_URL}")
    logger.info("=========================================================")

if __name__ == "__main__":
    main()
