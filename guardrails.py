import json
import logging
import os
from typing import Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from prompts import GUARDRAILS_SYSTEM_PROMPT
import config

logger = logging.getLogger(__name__)

def classify_query(query: str) -> Tuple[bool, str]:
    """
    Classifies if a user query is related to company HR policies or out-of-scope.
    Returns:
        Tuple[bool, str]: (is_in_scope, response_if_out_of_scope)
    """
    # Quick static local validation to handle obvious out-of-scope or empty queries
    cleaned_query = query.strip()
    if not cleaned_query:
        return False, "Please ask a question."
        
    # Check for Groq API key
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY is missing. Cannot run query classifier.")
        # Fail open to allow RAG chain to run and print API error, rather than blocking silently
        return True, ""

    try:
        logger.info(f"Classifying user query: '{cleaned_query[:60]}...'")
        
        # Initialize Groq LLM for fast classification
        # We enforce JSON mode via model_kwargs
        llm = ChatGroq(
            model=config.GROQ_CLASSIFIER_MODEL,
            temperature=0.0,
            model_kwargs={"response_format": {"type": "json_object"}},
            max_retries=2
        )
        
        # Build prompt
        prompt = ChatPromptTemplate.from_template(GUARDRAILS_SYSTEM_PROMPT)
        
        # Create classification chain
        classification_chain = prompt | llm | JsonOutputParser()
        
        # Run classification
        result = classification_chain.invoke({"question": cleaned_query})
        
        logger.info(f"Classifier output: {result}")
        
        is_hr_related = result.get("is_hr_related", True)
        explanation = result.get("explanation", "Reason not provided by classifier.")
        
        if not is_hr_related:
            logger.info(f"Query classified as OUT-OF-SCOPE. Reason: {explanation}")
            return False, "I can only answer HR-related questions from Zyro Dynamics policy documents."
            
        logger.info("Query classified as IN-SCOPE.")
        return True, ""
        
    except Exception as e:
        logger.error(f"Error during query classification: {e}. Falling back to default (In-Scope).")
        # In case of an unexpected exception (like rate limits or transient errors),
        # we fall back to in-scope to prevent locking out valid queries.
        return True, ""
