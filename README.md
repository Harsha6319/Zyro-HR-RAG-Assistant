# Zyro HR Assistant - Production RAG Chatbot

An internal HR RAG chatbot named **Zyro HR Assistant**, designed to answer employee questions strictly based on the company's HR policy documents. 

Developed with a production-grade technology stack for the **Kaggle NIAT Masterclass RAG Challenge**.

---

## Technical Stack
- **Core Engine:** Python 3.11+, LangChain (LCEL)
- **Vector Database:** FAISS
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (runs locally, completely free, fast)
- **Language Models (Groq API):**
  - **Generator:** `llama-3.3-70b-specdec` (state-of-the-art Reasoning model)
  - **Guardrail Classifier:** `llama-3.1-8b-instant` (ultra-fast routing)
- **Monitoring & Tracing:** LangSmith Tracing
- **User Interface:** Streamlit UI

---

## Project Structure
```
project/
│
├── data/                             # Scanned PDF policy documents
├── vectorstore/                      # Local FAISS index persistence directory
│
├── app.py                            # Streamlit chat application
├── ingest.py                         # PDF document parsing and indexing pipeline
├── download_data.py                  # Kaggle competition downloader script
├── submission_helper.py              # Kaggle submission parser for 15 eval questions
│
├── config.py                         # Centralized configuration variables
├── prompts.py                        # Strictly grounded RAG prompts
├── guardrails.py                     # Out-of-scope query guardrail classifier
├── requirements.txt                  # Dependency list
├── .env.example                      # Configuration template
└── README.md                         # This file
```

---

## Setup Instructions

### 1. Prerequisites & Installation
Ensure you have Python 3.11+ installed. Run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and configure:
```env
GROQ_API_KEY=your_groq_api_key_here
KAGGLE_USERNAME=your_kaggle_username_here
KAGGLE_KEY=your_kaggle_key_here

# Optional: LangSmith Tracing
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=zyro-rag-challenge
```

### 3. Fetch Competition Files
Run the automated downloader script to authenticate with the Kaggle API, fetch the competition PDFs and csv, and load them into the `data/` folder:

```bash
python download_data.py
```

### 4. Build Vector Store Index
Load, clean, split, and index all policies from the `data/` folder into the local FAISS database:

```bash
python ingest.py
```

---

## Running the Applications

### Launch Streamlit Chatbot Dashboard
Start the production-ready interactive Streamlit interface:

```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Generate Kaggle Submission CSV
Run the evaluation test suite to parse the 15 challenge questions through the classifier and RAG chain, generating the formatted output:

```bash
python submission_helper.py
```
This writes the final output to `submission.csv`.

---

## RAG Design Decisions

1. **Local Embeddings for Cost & Speed:** Using `all-MiniLM-L6-v2` locally avoids network request overhead during embedding calculations and operates without OpenAI api dependencies.
2. **Double-layer Guardrail Router:** Out-of-scope questions (e.g. weather, coding, general knowledge) are filtered by `guardrails.py` using `llama-3.1-8b-instant` *before* invoking retriever queries, maintaining strict grounding and reducing unnecessary computation.
3. **Structured Citation Tracking:** The LCEL pipeline uses `RunnableParallel` assignments to retain and return the matching source Document objects along with the final answer string, enabling the web app to display precise sources and matching page excerpts.
