# -*- coding: utf-8 -*-
import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import project utilities
from utils.retrieval import get_embeddings, load_vectorstore, get_retriever
from utils.tracing import verify_tracing_status
from guardrails import classify_query
from rag_chain import create_rag_chain
import config

# Page configuration
st.set_page_config(
    page_title="Zyro HR Assistant",
    page_icon="\U0001F3E2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2b5876, #4e4376);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 20px;
    }
    .source-badge {
        display: inline-block;
        padding: 4px 8px;
        margin: 4px;
        border-radius: 4px;
        background-color: #f1f3f5;
        border: 1px solid #dee2e6;
        font-size: 0.85rem;
        color: #495057;
        font-weight: 500;
    }
    .citation-container {
        margin-top: 15px;
        padding: 10px;
        border-left: 3px solid #4e4376;
        background-color: #f8f9fa;
        border-radius: 0 4px 4px 0;
    }
    .dark-mode-container {
        padding: 15px;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED PIPELINE INITIALIZATION -----------------

@st.cache_resource(show_spinner="Initializing Embeddings & FAISS Index...")
def initialize_rag_pipeline():
    """
    Loads embeddings and local FAISS index once and caches them across app refreshes.
    This guarantees sub-second chatbot response startup.
    """
    if not config.VECTORSTORE_DIR.exists():
        return None, "FAISS index folder not found. Please run 'python ingest.py' first to build the index."
        
    try:
        # Load local embeddings
        embeddings = get_embeddings()
        # Load local FAISS vectorstore
        db = load_vectorstore(config.VECTORSTORE_DIR, embeddings)
        # Setup MMR retriever
        retriever = get_retriever(db)
        # Create LCEL RAG chain
        rag_chain = create_rag_chain(retriever)
        
        # Check LangSmith status (runs logs in terminal)
        verify_tracing_status()
        
        return rag_chain, None
    except Exception as e:
        return None, f"Failed to initialize RAG components: {str(e)}"

# ----------------- SESSION STATE -----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------- SIDEBAR -----------------

with st.sidebar:
    st.markdown("### \U0001F916 Zyro HR Assistant")
    
    # Showcase active model variables
    st.info("System Settings (read-only configuration)")
    st.write(f"**Embedding Model:** `{config.EMBEDDING_MODEL_NAME}`")
    st.write(f"**LLM Model:** `{config.GROQ_LLM_MODEL}`")
    st.write(f"**Guardrail Model:** `{config.GROQ_CLASSIFIER_MODEL}`")
    st.write(f"**Chunk Size:** `{config.CHUNK_SIZE}` characters")
    
    st.markdown("---")
    
    # Indexed Documents List
    st.markdown("### \U0001F4C4 Indexed HR Policies")
    policies = [
        "Company Profile", "Employee Handbook", "Leave Policy",
        "Work From Home Policy", "Code of Conduct", "Performance Review Policy",
        "Compensation & Benefits Policy", "IT & Data Security Policy",
        "POSH Policy", "Onboarding & Separation Policy", "Travel & Expense Policy"
    ]
    for p in policies:
        st.markdown(f"- {p}")
        
    st.markdown("---")
    
    # Operations
    if st.button("\U0001F9F9 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------- MAIN UI -----------------

st.markdown("<div class='main-title'>Zyro HR Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Production-Grade RAG Chatbot grounded strictly in Zyro Dynamics Company Policies.</div>", unsafe_allow_html=True)

# Try loading RAG pipeline
rag_chain, init_error = initialize_rag_pipeline()

if init_error:
    st.error(init_error)
    st.warning("👉 Make sure your `.env` contains a valid `GROQ_API_KEY`, and you have executed `python ingest.py` to index the policy documents.")
    st.stop()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show citation metrics if present
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            st.markdown("<div class='citation-container'><strong>Sources Cited:</strong></div>", unsafe_allow_html=True)
            for source in message["sources"]:
                st.markdown(f"<span class='source-badge'>\U0001F4C4 {source}</span>", unsafe_allow_html=True)
                
            with st.expander("\U0001F50D Inspect Matching Text Passages"):
                for idx, doc in enumerate(message["contexts"]):
                    st.markdown(f"**Excerpt {idx+1} from {doc['source']} (Page {doc['page']}):**")
                    st.caption(doc["page_content"])
                    st.markdown("---")

# User Input
if prompt := st.chat_input("Ask a question about leave, WFH, POSH, data security, code of conduct..."):
    
    # 1. Add user query to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Process query
    with st.chat_message("assistant"):
        # Apply guardrail check
        is_in_scope, guardrail_msg = classify_query(prompt)
        
        if not is_in_scope:
            # Short-circuit query processing for out-of-scope questions
            answer_text = guardrail_msg
            st.markdown(answer_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": [],
                "contexts": []
            })
        else:
            # In-scope query: run retrieval and answering in a spinner block
            with st.spinner("Analyzing documents & generating grounded response..."):
                try:
                    # Run the LCEL RAG chain
                    response = rag_chain.invoke(prompt)
                    
                    answer_text = response.get("answer", "No response generated.")
                    retrieved_docs = response.get("context", [])
                    
                    # Deduplicate and extract sources from document metadata
                    sources = []
                    contexts = []
                    for doc in retrieved_docs:
                        src = doc.metadata.get("source", "Unknown Policy")
                        pg = doc.metadata.get("page", "?")
                        sources.append(src)
                        contexts.append({
                            "source": src,
                            "page": pg,
                            "page_content": doc.page_content
                        })
                    
                    # Deduplicate sources list
                    unique_sources = sorted(list(set(sources)))
                    
                    # Display response text
                    st.markdown(answer_text)
                    
                    # Display sources cited
                    if unique_sources:
                        st.markdown("<div class='citation-container'><strong>Sources Cited:</strong></div>", unsafe_allow_html=True)
                        for source in unique_sources:
                            st.markdown(f"<span class='source-badge'>\U0001F4C4 {source}</span>", unsafe_allow_html=True)
                            
                        # Allow deep dive inspector
                        with st.expander("\U0001F50D Inspect Matching Text Passages"):
                            for idx, ctx in enumerate(contexts):
                                st.markdown(f"**Excerpt {idx+1} from {ctx['source']} (Page {ctx['page']}):**")
                                st.caption(ctx["page_content"])
                                st.markdown("---")
                                
                    # Record in session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": unique_sources,
                        "contexts": contexts
                    })
                    
                except Exception as e:
                    error_msg = f"An error occurred while compiling your answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                        "contexts": []
                    })
