import os
# Force redeploy with latest cloud configuration
import requests
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CLOUD CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "document-qa")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Debug: Print key status (not the keys themselves)
print(f"DEBUG: PINECONE_API_KEY exists: {bool(PINECONE_API_KEY)}")
print(f"DEBUG: GROQ_API_KEY exists: {bool(GROQ_API_KEY)}")
print(f"DEBUG: HF_TOKEN exists: {bool(HF_TOKEN)}")

# Hugging Face Inference API Settings
# We use BAAI/bge-small-en-v1.5 as it defaults to feature-extraction task.
# Dimension is 384, same as MiniLM.
HF_EMBEDDING_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"

# --- INITIALIZE PINECONE ---
pc = None
def get_pinecone_client():
    global pc
    if pc is not None:
        return pc
        
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("ERROR: PINECONE_API_KEY is empty or None in environment!")
        return None
        
    try:
        pc = Pinecone(api_key=api_key)
        return pc
    except Exception as e:
        print(f"ERROR: Failed to initialize Pinecone: {e}")
        return None

def ensure_index_exists():
    """Initializes Pinecone index if it doesn't exist. Called during first operation."""
    client = get_pinecone_client()
    if not client:
        return
        
    try:
        if PINECONE_INDEX_NAME not in [idx.name for idx in client.list_indexes()]:
            client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384, # BGE-small dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
    except Exception as e:
        print(f"Error checking/creating index: {e}")

def get_huggingface_embeddings(text_list):
    """Generates embeddings using Hugging Face Inference API."""
    if not HF_TOKEN:
        raise Exception("HF_TOKEN is missing in environment variables.")
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "X-Wait-For-Model": "true",
        "X-Task": "feature-extraction"
    }
    response = requests.post(HF_EMBEDDING_URL, headers=headers, json={"inputs": text_list})
    
    if response.status_code != 200:
        raise Exception(f"HF API Error: {response.text}")
        
    return response.json()

def add_to_vector_db(chunks, filename):
    """
    Embeds chunks via HF and stores them in Pinecone.
    """
    ensure_index_exists()
    client = get_pinecone_client()
    if not client:
        raise Exception("Pinecone not initialized. Check PINECONE_API_KEY in Vercel settings.")
    
    index = client.Index(PINECONE_INDEX_NAME)
    
    # Generate embeddings
    embeddings = get_huggingface_embeddings(chunks)
    
    # Prepare vectors for Pinecone
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{filename}_{i}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "filename": filename,
                "chunk_index": i
            }
        })
    
    # Upsert to Pinecone
    index.upsert(vectors=vectors)

def query_rag(question):
    """
    Queries Pinecone for relevant chunks and generates an answer via Groq.
    """
    ensure_index_exists()
    client = get_pinecone_client()
    if not client:
        raise Exception("Pinecone client could not be initialized. Check API Key.")
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY is missing.")

    index = client.Index(PINECONE_INDEX_NAME)
    
    # 1. Embed the question
    q_embedding = get_huggingface_embeddings([question])[0]
    
    # 2. Query Pinecone
    results = index.query(
        vector=q_embedding,
        top_k=5,
        include_metadata=True
    )
    
    matches = results.get('matches', [])
    if not matches:
        return "I couldn’t find that in the uploaded document.", []
    
    # 3. Prepare context
    retrieved_chunks = [m['metadata']['text'] for m in matches]
    sources = [int(m['metadata']['chunk_index']) for m in matches]
    context = "\n\n".join(retrieved_chunks)
    
    # 4. Generate Answer via Groq
    prompt = f"""
    Answer the user's question ONLY based on the provided context below.
    If the answer is not in the context, respond strictly with: "I couldn’t find that in the uploaded document."

    Context:
    {context}

    Question: {question}
    """
    
    from openai import OpenAI
    client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    
    chat_response = client.chat.completions.create(
        model="llama3-8b-8192", # Using a common Groq model
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    answer = chat_response.choices[0].message.content
    return answer, sources
