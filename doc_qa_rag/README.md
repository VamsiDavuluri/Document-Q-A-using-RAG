# Document Q&A using RAG (Cloud/Vercel Version)

This version is optimized for deployment on **Vercel** using cloud-based free-tier services. It no longer requires local storage or models.

## Prerequisites

You need the following API keys (all offer free tiers):
1.  **Pinecone**: [Sign up here](https://www.pinecone.io/) for a vector database.
2.  **Groq**: [Sign up here](https://console.groq.com/) for the LLM.
3.  **Hugging Face**: [Sign up here](https://huggingface.co/settings/tokens) for an Inference API token.

## Environment Variables

Add these to your Vercel project or `.env` file:
```text
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=document-qa
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_huggingface_token
```

## Vercel Deployment

1.  Push your code to GitHub.
2.  Import your repository into [Vercel](https://vercel.com/).
3.  Vercel will automatically detect the FastAPI app (via `vercel.json`).
4.  Add the **Environment Variables** listed above in the Vercel dashboard.
5.  Deploy!

## Local Run (For Testing)

1.  `pip install -r doc_qa_rag/requirements.txt`
2.  `python -m doc_qa_rag.main`
3.  Visit `http://localhost:8000`.

## Features
- **Pinecone**: Cloud vector storage.
- **Hugging Face**: Free Inference API for embeddings (all-MiniLM-L6-v2).
- **Groq**: Lightning-fast free-tier LLM (Llama 3).
- **Vercel-Ready**: Serverless configuration via `vercel.json`.
