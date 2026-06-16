import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_groq import ChatGroq
from prompts import RAG_SYSTEM_PROMPT
import config

logger = logging.getLogger(__name__)

def format_docs(docs: List[Document]) -> str:
    """
    Formats retrieved LangChain documents into a single text block
    to supply as grounding context to the LLM.
    Includes source document and page number headers for explicit citation.
    """
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Policy")
        page = doc.metadata.get("page", "Unknown Page")
        # Format the document chunk for inclusion in LLM context
        chunk_str = f"Document Source: {source} (Page {page})\nContent:\n{doc.page_content}"
        formatted_chunks.append(chunk_str)
    
    return "\n\n=========================================\n\n".join(formatted_chunks)

def create_rag_chain(retriever) -> RunnableParallel:
    """
    Constructs and returns the LCEL RAG chain.
    The chain maps:
      Question -> Retrieved Docs -> Grounding Prompt -> ChatGroq -> parsed string answer
    
    Outputs a dictionary:
      {
        "question": str,
        "context": List[Document],
        "answer": str
      }
    """
    logger.info("Initializing Groq Chat LLM for RAG chain...")
    
    # Initialize the primary Chat LLM on Groq
    llm = ChatGroq(
        model=config.GROQ_LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_retries=3
    )
    
    # Initialize prompt template
    prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
    
    # 1. Component chain to map {context, question} dictionary -> LLM generated string response
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 2. Main RAG chain that retrieves documents first, then runs rag_chain_from_docs
    # Using LCEL RunnableParallel and assign to preserve the source documents list
    full_rag_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    logger.info("LCEL RAG chain pipeline constructed successfully.")
    return full_rag_chain


# Global reference for lazy initialization
_rag_chain = None

def get_or_create_chain() -> RunnableParallel:
    """
    Lazily initializes the FAISS vector database and LCEL pipeline
    to avoid slow module imports and startup overhead.
    """
    global _rag_chain
    if _rag_chain is None:
        logger.info("Lazily initializing FAISS index and retriever components...")
        from utils.retrieval import get_embeddings, load_vectorstore, get_retriever
        
        try:
            embeddings = get_embeddings()
            db = load_vectorstore(config.VECTORSTORE_DIR, embeddings)
            retriever = get_retriever(db)
            _rag_chain = create_rag_chain(retriever)
        except Exception as e:
            logger.error(f"Error lazily initializing RAG pipeline: {e}")
            raise RuntimeError(f"RAG system is not initialized. Make sure vectorstore exists at {config.VECTORSTORE_DIR}.") from e
            
    return _rag_chain

def answer_question(question: str) -> str:
    """
    Unified prediction function matching the Kaggle NIAT Masterclass requirements.
    
    1. Runs out-of-scope check first.
    2. Refuses to answer non-HR queries.
    3. Invokes RAG chain.
    4. Enforces strict refusal if no documents are retrieved.
    """
    # Import guardrails here to avoid circular imports
    from guardrails import classify_query

    cleaned_q = question.strip()
    if not cleaned_q:
        return "The provided policy documents do not contain that information."

    # 1. Apply out-of-scope detection guardrails
    is_in_scope, guardrail_msg = classify_query(cleaned_q)
    if not is_in_scope:
        logger.info(f"Guardrail triggered for question: '{cleaned_q[:50]}'")
        return "I can only answer HR-related questions from Zyro Dynamics policy documents."

    try:
        # 2. Lazy load the LCEL chain
        chain = get_or_create_chain()
        
        # 3. Invoke the chain
        response = chain.invoke(cleaned_q)
        
        # 4. Enforce strict factual grounding: check if any documents were retrieved
        docs = response.get("context", [])
        if not docs:
            logger.warning("Retriever returned NO relevant documents. Outputting strict refusal.")
            return "The provided policy documents do not contain that information."
            
        answer = response.get("answer", "").strip()
        
        # Secondary fallback in case LLM outputs an empty answer
        if not answer:
            return "The provided policy documents do not contain that information."
            
        return answer
        
    except Exception as e:
        logger.error(f"Exception encountered during answer generation: {e}")
        # Always default to strict refusal on errors to prevent hallucinations
        return "The provided policy documents do not contain that information."

