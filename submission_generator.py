# -*- coding: utf-8 -*-
import base64
import logging
import os
import pandas as pd
from dotenv import load_dotenv

import config
from rag_chain import answer_question
from utils.tracing import verify_tracing_status

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("submission_generator")

def encrypt_text(text: str) -> str:
    """
    Kaggle Competition Answer Encryption Placeholder.
    
    If the Kaggle evaluation cell requires a custom encryption function,
    replace this Base64 encoder with the official competition encryption utility.
    """
    if not text:
        return ""
    # Base64 encode the string to act as the encrypted representation
    encoded_bytes = base64.b64encode(text.encode("utf-8"))
    return encoded_bytes.decode("utf-8")

def generate_submission(df_competition: pd.DataFrame) -> pd.DataFrame:
    """
    Exposed function that processes the competition DataFrame.
    
    Expected input DataFrame columns:
      - question_id
      - question_enc (optional, or as metadata)
      - question (the raw decrypted query text to run through RAG)
      
    Returns a validated DataFrame with exactly:
      - question_id
      - question_enc
      - answer_enc
      - streamlit_link
      - langsmith_link
    """
    logger.info("Initializing submission generation...")
    
    # 1. Load configuration and check tracing status
    load_dotenv()
    verify_tracing_status()
    
    # Check if necessary input columns are present
    if "question" not in df_competition.columns:
        raise KeyError("Input competition DataFrame must contain a 'question' column for the RAG chatbot.")
    
    if "question_id" not in df_competition.columns:
        raise KeyError("Input competition DataFrame must contain a 'question_id' column.")
        
    # Standardize/Create question_enc if not present in the input
    if "question_enc" not in df_competition.columns:
        logger.warning("'question_enc' column not found in input. Generating placeholder from 'question'...")
        df_competition["question_enc"] = df_competition["question"].apply(encrypt_text)

    # 2. Apply prediction logic to every question
    answers = []
    logger.info(f"Processing {len(df_competition)} competition questions...")
    for idx, row in df_competition.iterrows():
        question = str(row["question"]).strip()
        q_id = str(row["question_id"]).strip()
        
        logger.info(f"Answering {q_id}...")
        # Get answer (unified prediction function handles scope & retriever constraints)
        answer = answer_question(question)
        answers.append(answer)
        
    df_competition["answer_raw"] = answers
    
    # 3. Generate answer_enc values
    logger.info("Encrypting generated answers...")
    df_competition["answer_enc"] = df_competition["answer_raw"].apply(encrypt_text)
    
    # 4. Populate Streamlit & LangSmith links
    df_competition["streamlit_link"] = config.STREAMLIT_APP_URL
    df_competition["langsmith_link"] = config.LANGSMITH_PROJECT_URL
    
    # 5. Extract and order the final submission columns
    required_cols = [
        "question_id",
        "question_enc",
        "answer_enc",
        "streamlit_link",
        "langsmith_link"
    ]
    df_submission = df_competition[required_cols].copy()
    
    # 6. Apply strict pre-saving validation checks
    validate_submission_data(df_submission)
    
    return df_submission

def validate_submission_data(df: pd.DataFrame) -> None:
    """
    Performs verification checks on the submission dataframe.
    Raises ValueError or AssertionError if formatting constraints are violated.
    """
    logger.info("Running pre-export validation checks...")
    
    # Check 1: Row Count check
    row_count = len(df)
    if row_count != 15:
        raise ValueError(f"Validation Error: Submission DataFrame must contain exactly 15 rows. Found {row_count} rows.")
        
    # Check 2: Column list and ordering check
    expected_cols = ["question_id", "question_enc", "answer_enc", "streamlit_link", "langsmith_link"]
    current_cols = list(df.columns)
    if current_cols != expected_cols:
        raise ValueError(f"Validation Error: Column list or ordering mismatch.\nExpected: {expected_cols}\nFound: {current_cols}")
        
    # Check 3: Missing value check (nulls or empty strings)
    if df.isnull().any().any():
        null_locations = df.isnull().sum()
        raise ValueError(f"Validation Error: Null values detected in submission data:\n{null_locations}")
        
    for col in df.columns:
        empty_str_count = (df[col].astype(str).str.strip() == "").sum()
        if empty_str_count > 0:
            raise ValueError(f"Validation Error: Found {empty_str_count} empty strings in column '{col}'.")
            
    # Check 4: Duplicate IDs check
    dup_ids_count = df["question_id"].duplicated().sum()
    if dup_ids_count > 0:
        duplicate_ids = df[df["question_id"].duplicated()]["question_id"].tolist()
        raise ValueError(f"Validation Error: Duplicate question_id values detected: {duplicate_ids}")
        
    logger.info("Pre-export validation completed successfully: All checks passed.")

def main():
    """
    CLI interface to run the generator on local CSV file
    """
    # Look for evaluation CSV
    eval_file = config.EVAL_QUESTIONS_FILE
    if not eval_file.exists():
        logger.warning(f"Evaluation questions CSV not found at {eval_file}")
        # Search for any CSV file in data/
        csv_files = list(config.DATA_DIR.glob("*.csv"))
        if csv_files:
            eval_file = csv_files[0]
            logger.info(f"Using found evaluation file: {eval_file}")
        else:
            logger.error("No evaluation questions CSV file found. Cannot run generator.")
            return

    try:
        # Load local questions
        df_eval = pd.read_csv(eval_file)
        
        # Standardize column mappings (expecting ID and Question columns)
        # If the columns are not named question_id and question, let's map them
        mappings = {}
        for col in df_eval.columns:
            col_lower = col.strip().lower()
            if "id" in col_lower or col_lower == "q":
                mappings[col] = "question_id"
            elif "question" in col_lower or "query" in col_lower:
                mappings[col] = "question"
                
        df_eval = df_eval.rename(columns=mappings)
        
        # Run pipeline
        df_submission = generate_submission(df_eval)
        
        # Save output
        df_submission.to_csv(config.SUBMISSION_FILE, index=False)
        logger.info(f"Successfully generated and validated {config.SUBMISSION_FILE}")
        
    except Exception as e:
        logger.exception(f"Submission generation failed: {e}")

if __name__ == "__main__":
    main()
