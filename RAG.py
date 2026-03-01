import fitz
import pytesseract
import faiss
import numpy as np
import os

import io
import requests
from bs4 import BeautifulSoup
from PIL import Image
from mistralai import Mistral

API_KEY = "I7H1l8yO1g8S0y59ohDxQNaqvSXXuAGm"
client = Mistral(api_key=API_KEY)

CHUNK_SIZE = 800
TOP_K = 3
EMBED_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-small-latest"  

# ==========================
# EXTRACTION TEXTE
# ==========================
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text()
        if not page_text.strip():
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            page_text = pytesseract.image_to_string(img)
        text += page_text + "\n"
    return text

def get_text_from_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError(f"Impossible de récupérer la page: {r.status_code}")
    soup = BeautifulSoup(r.text, "html.parser")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i+size] for i in range(0, len(text), size)]

def format_document(source, source_type="pdf"):
    if source_type == "pdf":
        text = extract_text_from_pdf(source)
    elif source_type == "url":
        text = get_text_from_url(source)
    else:
        raise ValueError("source_type doit être 'pdf' ou 'url'")
    if not text.strip():
        raise ValueError(f"Le document '{source}' ne contient aucun texte exploitable.")
    return chunk_text(text)

# ==========================
# PRÉPARATION DES DOCUMENTS RAG
# ==========================
def prepare_documents(pdf_folder=None, url_list=None):
    all_chunks = []
    if pdf_folder:
        for filename in os.listdir(pdf_folder):
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, filename)
                try:
                    chunks = format_document(pdf_path, "pdf")
                    all_chunks.extend(chunks)
                    print(f"[PDF] {filename} → {len(chunks)} chunks")
                except Exception as e:
                    print(f"[PDF] {filename} → Erreur : {e}")
    if url_list:
        for url in url_list:
            try:
                chunks = format_document(url, "url")
                all_chunks.extend(chunks)
                print(f"[URL] {url} → {len(chunks)} chunks")
            except Exception as e:
                print(f"[URL] {url} → Erreur : {e}")
    print(f"\nTotal chunks préparés : {len(all_chunks)}")
    return all_chunks

# ==========================
# EMBEDDINGS ET RAG
# ==========================
def embed_texts(texts):
    response = client.embeddings.create(model=EMBED_MODEL, inputs=texts)
    return np.array([e.embedding for e in response.data]).astype("float32")