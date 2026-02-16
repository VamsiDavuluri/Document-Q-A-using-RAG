import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from doc_qa_rag.utils import extract_text_from_pdf, extract_text_from_txt, chunk_text
from doc_qa_rag.rag import get_or_create_collection, add_to_vector_db, query_rag

app = FastAPI(title="Document Q&A using RAG")

# Temporary directory for file uploads
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Static files directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(STATIC_DIR, "index.html"), "r") as f:
        return f.read()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and TXT are allowed.")
    
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save file locally
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Extract text
        if file_ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_txt(file_path)
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="The document is empty or text could not be extracted.")
            
        # Chunk text
        chunks = chunk_text(text)
        
        # Store in ChromaDB
        collection = get_or_create_collection()
        add_to_vector_db(collection, chunks, filename)
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": filename,
            "number_of_chunks": len(chunks)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        collection = get_or_create_collection()
        answer, sources = query_rag(collection, request.question)
        
        return {
            "question": request.question,
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
