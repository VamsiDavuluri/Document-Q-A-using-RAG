# Document Q&A using RAG

A simple FastAPI backend for uploading documents (PDF/TXT) and asking questions based on their content using RAG (Retrieval-Augmented Generation).

## Features
- **FastAPI**: Modern, fast web framework.
- **Beautiful UI**: Simple and elegant frontend built with HTML/CSS.
- **ChromaDB**: Local vector database for persistent storage.
- **OpenAI**: Uses `text-embedding-3-small` for embeddings and `gpt-4o-mini` for answering.
- **Supports PDF & TXT**: Extracts text using PyMuPDF and standard file reading.

## Project Structure
```text
doc_qa_rag/
  main.py          # FastAPI application & endpoints
  rag.py           # ChromaDB & OpenAI logic
  utils.py         # Text extraction & chunking
  requirements.txt # Dependencies
  .env.example     # Environment variable template
  README.md        # This file
  chroma_db/       # Local vector database (auto-created)
```

## Setup Instructions

1. **Clone or copy the files** into a directory named `doc_qa_rag`.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Models (No Credits Needed!)**:

   ### Option A: Fully Local (Ollama)
   1. Download and install [Ollama](https://ollama.com/) on your Mac.
   2. Run `ollama run llama3.2:1b` (or another model like `llama3`) in your terminal.
   3. The app will now use this local model for free!

   ### Option B: Free API (Groq)
   1. Get a free API key from [Groq Console](https://console.groq.com/).
   2. Add it to your `.env` file:
      ```text
      GROQ_API_KEY=your_groq_key_here
      ```
   
   ### Embeddings
   This version uses `sentence-transformers` automatically. Your documents are embedded **locally** on your Mac for free. No API keys needed for searching!

## How to Run

Start the server using uvicorn:
```bash
python -m doc_qa_rag.main
```
The server will run at `http://localhost:8000`.

## API Usage (CURL Examples)

### 1. Health Check
```bash
curl -X GET http://localhost:8000/health
```

### 2. Upload a Document
Replace `your_file.pdf` with the actual path to your PDF or TXT file.
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your_file.pdf"
```

### 3. Ask a Question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this document?"}'
```

## Implementation Details
- **Chunking**: Splits text into 800-character segments with 150-character overlap.
- **Persistence**: ChromaDB stores embeddings in the `chroma_db/` directory, so data persists across restarts.
- **Strict Answering**: The model is instructed to only answer based on the retrieved context. If not found, it returns: *"I couldn’t find that in the uploaded document."*
