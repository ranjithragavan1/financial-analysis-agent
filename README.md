# 🤖 Enterprise Financial Analysis Agent (Hybrid RAG)

**Current Status:** 🟢 Active Development | Preparing for v2 release

An autonomous AI agent designed to analyze complex financial documents (10-K reports). Built with a **Hybrid Cloud Architecture** to optimize for data privacy and inference costs.

## 🚀 Key Features
- **Hybrid RAG Pipeline:** Combines **Local GPU Embeddings** (HuggingFace/CUDA) for zero-latency retrieval with **Google Gemini Pro** for high-level reasoning.
- **Agentic Decision Making:** Uses **LangGraph** to dynamically decide when to retrieve internal documents vs. answer generally.
- **Cost Optimized:** Reduces embedding API costs by 100% using local inference.
- **Enterprise Ready:** Includes strict type checking and environment variable security.

## 🛠️ Tech Stack
- **Orchestration:** LangChain / LangGraph
- **LLM:** Google Gemini 1.5 Flash
- **Vector DB:** FAISS (Local)
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2) on GPU
- **Language:** Python 3.10+

## ⚡ How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `GOOGLE_API_KEY` to `.env`
4. Run ingestion: `python ingest.py`
5. Start agent: `python agent.py`