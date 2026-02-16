import fitz  # PyMuPDF
import os

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extracts text from a TXT file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    """
    Splits text into chunks of specified size with an overlap.
    Returns a list of strings.
    """
    chunks = []
    if not text:
        return chunks
    
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += (chunk_size - overlap)
        
    return chunks
