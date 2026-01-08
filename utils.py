# utils.py

import torch
import pandas as pd
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ---------- Text utils ----------

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks


# ---------- Embedding ----------

encoder = SentenceTransformer("all-MiniLM-L6-v2")

def encode_text(text):
    return encoder.encode(text, convert_to_tensor=True)


# ---------- Decision logic ----------

def decide_consistency(Q, claim_embedding, threshold=0.55):
    """
    Q: BDH synaptic state (tensor)
    claim_embedding: embedding of backstory claim
    """
    q_vec = Q.mean(dim=0)
    score = F.cosine_similarity(q_vec, claim_embedding, dim=0)
    return "consistent" if score.item() > threshold else "contradict"
