import os
import chromadb
import requests
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- OPTION 1: OLLAMA (Local & Free) ---
OLLAMA_BASE_URL = "http://localhost:11434/api"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b") # Or "mistral", "llama3", etc.

# --- OPTION 2: GROQ (Free Tier API) ---
# If you have a Groq key (get one for free at console.groq.com)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize OpenAI client (can also be used for Groq)
if GROQ_API_KEY:
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB client with persistence
DB_PATH = "chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# --- LOCAL EMBEDDING FUNCTION (Free) ---
# This runs locally on your CPU/GPU and doesn't need credits.
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def get_or_create_collection(name="rag_local_collection"):
    """Gets or creates a ChromaDB collection using local embeddings."""
    return chroma_client.get_or_create_collection(name=name, embedding_function=local_ef)

def add_to_vector_db(collection, chunks, filename):
    """
    Adds chunks to ChromaDB. Embeddings are generated locally by the collection's EF.
    """
    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]
    
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas
    )

def query_rag(collection, question):
    """
    Queries ChromaDB for relevant chunks and generates an answer locally or via free API.
    """
    # 1. Query ChromaDB (Local embedding is handled automatically)
    results = collection.query(
        query_texts=[question],
        n_results=5
    )
    
    retrieved_chunks = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]
    
    if not retrieved_chunks:
        return "I couldn’t find that in the uploaded document.", []
    
    # 2. Prepare context
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Answer the user's question ONLY based on the provided context below. If the answer is not in the context, respond strictly with: 'I couldn’t find that in the uploaded document.'\n\nContext:\n{context}\n\nQuestion: {question}"
    
    # 3. Generate Answer (Try Ollama first, then Groq, then OpenAI)
    try:
        # Try Ollama (Local)
        response = requests.post(
            f"{OLLAMA_BASE_URL}/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            answer = response.json().get("response", "")
            sources = [m['chunk_index'] for m in retrieved_metadata]
            return answer, sources
    except Exception:
        pass # Fallback to API if Ollama is not running

    # Try Groq or OpenAI
    try:
        model_name = "llama-3.1-8b-instant" if GROQ_API_KEY else "gpt-4o-mini"
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = chat_response.choices[0].message.content
        sources = [m['chunk_index'] for m in retrieved_metadata]
        return answer, sources
    except Exception as e:
        return f"Error: No local LLM running and API failed. {str(e)}", []
