# -*- coding: utf-8 -*-
import logging
import pandas as pd
from pathlib import Path
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate_submission")

def validate_submission_file(file_path: Path) -> bool:
    """
    Validates the structure and content of submission.csv.
    Returns True if valid, False if errors were encountered.
    """
    logger.info(f"Starting validation on submission file: {file_path}")
    
    # Check 1: File existence
    if not file_path.exists():
        logger.error(f"Validation Failed: Submission file does not exist at: {file_path}")
        return False
        
    try:
        # Load the CSV
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Validation Failed: Could not read CSV file. Error: {e}")
        return False
        
    has_errors = False
    
    # Check 2: Row count check (Exactly 15 rows)
    row_count = len(df)
    if row_count != 15:
        logger.error(f"Validation Error: Row count must be exactly 15. Found {row_count} rows.")
        has_errors = True
    else:
        logger.info("Row count check: PASSED (15 rows).")
        
    # Check 3: Column count and ordering
    expected_cols = [
        "question_id",
        "question_enc",
        "answer_enc",
        "streamlit_link",
        "langsmith_link"
    ]
    current_cols = list(df.columns)
    if current_cols != expected_cols:
        logger.error(f"Validation Error: Columns mismatch or incorrect ordering.\nExpected: {expected_cols}\nFound: {current_cols}")
        has_errors = True
    else:
        logger.info("Column name and ordering check: PASSED.")
        
    # Check 4: Null values and empty strings check
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.error(f"Validation Error: Null values detected in columns:\n{null_counts}")
        has_errors = True
        
    for col in df.columns:
        empty_strings = (df[col].astype(str).str.strip() == "").sum()
        if empty_strings > 0:
            logger.error(f"Validation Error: Found {empty_strings} empty string values in column '{col}'.")
            has_errors = True
            
    # Check 5: Duplicate check
    dup_count = df["question_id"].duplicated().sum()
    if dup_count > 0:
        dup_ids = df[df["question_id"].duplicated()]["question_id"].tolist()
        logger.error(f"Validation Error: Duplicate question_ids detected: {dup_ids}")
        has_errors = True
        
    # Check 6: Streamlit and LangSmith URL format checking
    for idx, row in df.iterrows():
        q_id = row["question_id"]
        st_link = str(row["streamlit_link"])
        ls_link = str(row["langsmith_link"])
        
        # Check Streamlit URL
        if not (st_link.startswith("http://") or st_link.startswith("https://")):
            logger.error(f"Validation Error in row {idx} ({q_id}): streamlit_link '{st_link}' must start with http:// or https://")
            has_errors = True
            
        # Check LangSmith URL
        if not (ls_link.startswith("http://") or ls_link.startswith("https://")):
            logger.error(f"Validation Error in row {idx} ({q_id}): langsmith_link '{ls_link}' must start with http:// or https://")
            has_errors = True
            
    if has_errors:
        logger.error("=========================================================")
        logger.error("VALIDATION RESULT: FAILED. Please resolve errors listed above.")
        logger.error("=========================================================")
        return False
    else:
        logger.info("=========================================================")
        logger.info("VALIDATION RESULT: SUCCESS. CSV is fully formatted and ready to submit!")
        logger.info("=========================================================")
        return True

def main():
    submission_path = config.SUBMISSION_FILE
    success = validate_submission_file(submission_path)
    if not success:
        exit(1)

if __name__ == "__main__":
    main()
