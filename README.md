# Local PDF Chat with RAG

This project implements a local Retrieval-Augmented Generation (RAG) system that allows users to ask questions about a PDF document.

The application uses:

- Python
- LangChain
- Ollama
- Llama 3
- ChromaDB
- Nomic Embeddings
- Streamlit

## How it works

1. A PDF document is loaded.
2. The document is split into smaller text chunks.
3. The chunks are converted into embeddings.
4. The embeddings are stored in ChromaDB.
5. When a user asks a question, the most relevant chunks are retrieved.
6. Llama 3 generates an answer using the retrieved PDF context.

## Architecture

PDF → Text Chunks → Embeddings → ChromaDB → Retrieval → Llama 3 → Answer

## Installation

Create a virtual environment:

```bash
python -m venv .venv