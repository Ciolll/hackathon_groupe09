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
from RAG import prepare_documents, embed_texts

API_KEY = "I7H1l8yO1g8S0y59ohDxQNaqvSXXuAGm"
client = Mistral(api_key=API_KEY)

CHUNK_SIZE = 800
TOP_K = 3
EMBED_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-small-latest"  

class RAGAgent:
    def __init__(self, chunks):
        self.chunks = chunks
        self.embeddings = embed_texts(chunks)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def search(self, question, k=TOP_K):
        q_emb = embed_texts([question])
        D, I = self.index.search(q_emb, k)
        return [self.chunks[i] for i in I[0]]

# ==========================
# FONCTION DE DRAFT AVEC RAG
# ==========================
def draft_report(state: dict) -> str:
    documents = prepare_documents("RAG")
    rag_agent = RAGAgent(documents)
    patient_id=state.get("patient_id","no patient id")
    patient_age=state.get("patient_age","no age provided")
    patient_sex=state.get("patient_sex","no gender provided")
    patient_weight=state.get("patient_weight","no weight provided")
    actual = state.get("anomaly_detected","no anomalies currently detected")
    history = state.get("patient_history","no historical data")
    compare=state.get("clinical_comparison","No comparison")
    # 1️⃣ Recherche dans les documents RAG
    retrieved_chunks_actual = rag_agent.search(actual)
    retrieved_chunks_history = rag_agent.search(history)
    context = "\n\n".join(retrieved_chunks_actual + retrieved_chunks_history)

    # 2️⃣ Prompt combinant contexte RAG + données patient
    prompt =f"""You are an expert radiologist. Generate a structured medical report in English.

Here is the Patient Information : id :{patient_id}; age :{patient_age}; gender : {patient_sex} ; Weight :{patient_weight}

Here is the data from the patient's AI analysis:

{actual}

Here is the data from the patient's prior consultations/reports: 

{history} 

Here is a comparison between the historical reports and the image analyses:

{compare}

Here is the complementary medical information from the literature / RAG documents:

{context}


The report must contain:
1. Patient Information (ID, age, gender, weight)
2. Findings (list of nodules with sizes, localization, and volume)
3. Evolution compared to previous exams (if history is available). If a retrospective detection or a past counting error is mentioned, integrate it in a highly professional manner.
4. Conclusion

Generate only the report, with no surrounding text.
"""
    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "draft_report" : response.choices[0].message.content.strip()
    }