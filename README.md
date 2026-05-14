#  Intent-Aware Multi-Agent RAG Debugging Assistant

> An intelligent Retrieval-Augmented Generation (RAG) system designed to analyze code and documentation with intent-aware query routing. Get precise answers about your codebase using natural language.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Current Status](#current-status)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **AI Debugging RAG Assistant** is a sophisticated document and code retrieval system that helps developers understand, debug, and navigate their codebase by answering questions in natural language. 

Key capabilities:
-  **Intelligent Query Classification** - Automatically identifies query intent (debug, flow, definition, code lookup, general)
- **Hybrid Retrieval** - Searches both code and documentation simultaneously for comprehensive answers
- **Context-Aware Responses** - Grounds answers in actual code/docs to prevent hallucinations
-  **Multimodal Support** - Handles code, documentation, and images
-  **Fast Local Processing** - Uses FAISS for efficient vector similarity search

---

## ✨ Features

### Core Functionality
-  **Data Ingestion**
  - Load Python code files with enriched metadata (source path, filename, type)
  - Support for multiple document formats: `.pdf`, `.txt`, `.md`
  - Automatic chunking and preprocessing of long documents/code files

-  **Vector Store Management**
  - FAISS-based vector database for fast similarity search
  - Persistent index storage and efficient reloading
  - Embedding via pre-trained models (e.g., Hugging Face embeddings)

-  **Intelligent Query Processing**
  - Fast query classification without LLM overhead
  - Multi-label routing (code, flow, definition, debug, general)
  - Dynamic retriever selection based on query type

- **Retrieval & Ranking**
  - Semantic similarity search across code and documentation
  - Metadata-based filtering (type, source path)
  - Context window management for token optimization

-  **Answer Generation**
  - LLM-powered response synthesis with context grounding
  - Source attribution with code/doc references
  - Built-in guardrails against hallucinations

-  **Interactive CLI**
  - Real-time chat loop for exploration
  - Query type visualization
  - Source tracking and relevance scoring

---

##  Architecture

```
┌─────────────────────────────────────┐
│   User Query (Natural Language)      │
└──────────────┬──────────────────────┘
               │
               ↓
        ┌──────────────┐
        │  Classifier  │ ← Query Type Detection (debug, flow, code, etc.)
        └──────┬───────┘
               │
               ↓
        ┌──────────────────────┐
        │  Retriever Router    │
        └──────┬───────────┬───┘
               │           │
        ┌──────↓────┐  ┌───↓──────┐
        │Code Index │  │ Doc Index │
        │(FAISS)    │  │ (FAISS)   │
        └─────┬──────┘  └──────┬───┘
              │                │
              └────────┬───────┘
                       ↓
            ┌──────────────────────┐
            │ Retrieved Context    │
            │ (Code + Docs)        │
            └──────────┬───────────┘
                       ↓
            ┌──────────────────────┐
            │  LLM Response Gen    │
            │  + Source Attribution│
            └──────────┬───────────┘
                       ↓
            ┌──────────────────────┐
            │ Final Answer + Refs  │
            └──────────────────────┘
```

---

##  Project Structure

```
AIDebugging_btp/
├── main.py                      # CLI entry point with query loop
├── main1.py                     # Alternative implementation
├── app.py                       # FastAPI/API skeleton
├── requirements.txt             # Dependencies
│
├── rag/
│   ├── pipeline.py              # Main RAG pipeline orchestration
│   ├── retriever.py             # Retriever builders and query routing
│   └── tempCodeRunnerFile.py    # Test utilities
│
├── vectorstore/
│   ├── vector_db.py             # FAISS vector store creation/loading
│   └── faiss_index/
│       └── index.faiss          # Persisted vector index
│
├── ingestion/
│   ├── load_code.py             # Python code file ingestion
│   ├── load_docs.py             # Document loading (pdf, txt, md)
│
├── embeddings/
│   └── embedding_model.py       # Embedding model initialization
│
├── agents/
│   ├── query_agent.py           # Query processing agent
│   ├── retrieval_agent.py       # Retrieval coordination
│   ├── assistant_engine.py      # Main assistant logic
│   └── a1.py                    # Agent utilities
│
├── multimodals/
│   └── image_reader.py          # Image OCR and text extraction
│
├── evaluation/
│   ├── evaluator.py             # Evaluation metrics
│   └── test_cases.py            # Test case definitions
│
├── data/
│   ├── code/                    # Source code for indexing
│   │   ├── main.py
│   │   ├── core/
│   │   └── api/
│   ├── docs/                    # Documentation files
│   │   └── fastapi_documentation.txt
│   └── images/                  # Image files for multimodal retrieval
│
├── debug_dataset/               # Debugging scenarios and logs
│   ├── browser_cases/
│   ├── dependency_cases/
│   ├── network_cases/
│   ├── system_logs/
│   └── project/
│
├── extras/
│   └── datacode_extras/         # Extra utilities and models
│
└── README.md                    # This file
```

---

##  Installation

### Prerequisites
- Python 3.9+
- pip or conda
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AIDebugging_btp.git
   cd AIDebugging_btp
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\venv\Scripts\Activate.ps1
   
   # Windows (CMD)
   python -m venv venv
   .\venv\Scripts\activate.bat
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download/configure embedding model**
   ```bash
   python embeddings/embedding_model.py
   ```

---

## Usage

### Quick Start: Interactive CLI

```bash
python main.py
```

You'll see:
```
 Intelligent AI Debugging Assistant Ready!
Type 'exit' to quit

💬 Ask your question: How does authentication work?

 AI Detected Type: flow
 Retrieved 5 strong chunks

 Answer: 
Authentication in this system uses JWT tokens...

 Sources:
  • data/code/api/security.py (lines 15-42)
  • data/docs/fastapi_documentation.txt (section: Auth)
```

### Example Queries

**Debug queries:**
```
Why am I getting a 401 error?
What's causing the TypeError in main.py?
```

**Flow queries:**
```
How does the login process work?
What's the process for retrieving user data?
```

**Definition queries:**
```
What is CORS?
Define JWT authentication
```

**Code lookup queries:**
```
Where is the user validation function?
Show me the API routes
```

### Programmatic Usage

```python
from rag.retriever import create_rag_system

# Initialize RAG system
rag = create_rag_system()

# Ask a question
response = rag.query("How does authentication work?")
print(response['answer'])
print("Sources:", response['sources'])
```

---

##  Configuration

### Vector Store Configuration

Edit `vectorstore/vector_db.py`:
```python
# Chunk size for documents
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# FAISS index settings
INDEX_PATH = "vectorstore/faiss_index/index.faiss"
```

### Embedding Model

Configure in `embeddings/embedding_model.py`:
```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
```

### Data Paths

Update ingestion paths in `ingestion/load_code.py` and `ingestion/load_docs.py`:
```python
CODE_DIR = "data/code"
DOCS_DIR = "data/docs"
```

---


##  Current Status

The core RAG pipeline and intelligent retrieval workflows are fully functional and support:

- Intent-aware query routing
- Hybrid code + documentation retrieval
- FAISS-based semantic vector search
- OCR-powered multimodal ingestion
- LLM-grounded response generation
- Source attribution and hallucination reduction
- Modular multi-agent workflow separation
- Evaluation pipelines for retrieval quality and groundedness
- Interactive debugging interface for real-time querying

The system is actively being enhanced with scalability and orchestration improvements.
---

## 🚀 Future Enhancements

- Multi-agent orchestration and memory optimization
- Incremental indexing for real-time codebase updates
- IDE integrations (VS Code extension)
- Optimized deployment pipelines for scalable production usage
---

##  Performance & Benchmarks

| Metric | Status | Details |
|--------|--------|---------|
| Query Latency | ~2-5s | Includes LLM inference |
| Retrieval Time | ~100-200ms | FAISS similarity search |
| Indexing Speed | ~500 docs/min | Depends on chunk size |
| Memory Usage | ~2-4GB | Vector index + model |
| Evaluation Framework | Implemented | Relevance and groundedness scoring pipelines integrated |

---

##  Testing & Evaluation

### Run Tests
```bash
python evaluation/test_cases.py
python test_evaluator.py
```

### Test Dataset
Located in `debug_dataset/` with scenarios:
- Browser errors
- Dependency issues
- Network problems
- System logs
- API debugging cases

---

## 🛠️ Development

### Adding New Documents/Code
1. Place files in `data/code/` or `data/docs/`
2. Regenerate index:
   ```bash
   python vectorstore/vector_db.py --rebuild
   ```

### Extending Query Types
Edit `main.py` `classify_query()` function:
```python
def classify_query(query):
    q = query.lower()
    if "your_keyword" in q:
        return "your_type"
    # ... existing code
```

### Custom Retrievers
Implement in `rag/retriever.py`:
```python
class CustomRetriever:
    def retrieve(self, query, k=5):
        # Your retrieval logic
        pass
```

---

## 📝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Guidelines
- Follow PEP 8 for Python code
- Add docstrings to all functions
- Include tests for new features
- Update README for significant changes

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

