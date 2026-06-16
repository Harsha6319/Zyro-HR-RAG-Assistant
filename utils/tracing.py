import os
import logging

logger = logging.getLogger(__name__)

def verify_tracing_status() -> None:
    """
    Verifies that the environment variables required for LangSmith tracing
    are properly set, and logs instruction details on how to view execution traces.
    """
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT", "zyro-rag-challenge")
    
    if tracing_enabled and api_key:
        logger.info("========================================= LANGSMITH TRACING STATUS =========================================")
        logger.info(f"LangSmith Tracing is ENABLED.")
        logger.info(f"Tracing Project Name: {project}")
        logger.info("All pipeline events (retrievals, guardrail evaluations, prompts, and LLM responses) will be tracked.")
        logger.info("To view your execution runs and debug traces, log in to your LangSmith Dashboard:")
        logger.info(f"   URL: https://smith.langchain.com/ (Select project: '{project}')")
        logger.info("==============================================================================================================")
    else:
        logger.info("========================================= LANGSMITH TRACING STATUS =========================================")
        logger.info("LangSmith Tracing is currently DISABLED or INCOMPLETE.")
        logger.info("To track your RAG pipeline, check retrieval latency, and debug LLM generations, configure:")
        logger.info("   1. LANGCHAIN_TRACING_V2=true")
        logger.info("   2. LANGCHAIN_API_KEY=your_langsmith_key")
        logger.info("   3. LANGCHAIN_PROJECT=zyro-rag-challenge")
        logger.info("in your local .env file. Sign up at https://smith.langchain.com/")
        logger.info("==============================================================================================================")
