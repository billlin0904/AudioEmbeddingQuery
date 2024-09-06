from chromadb import PersistentClient
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, Query
import torch
import os
from database import AudioEmbeddingDatabase
from separate import AudioSeparate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

app = FastAPI()
db = AudioEmbeddingDatabase()

class EmbeddingItem(BaseModel):
    file_path: str
    audio_id: str

@app.post("/embed_and_save")
async def embed_and_save(item: EmbeddingItem):    
    try:
        db.embed_and_save(item.file_path, item.audio_id)
    except Exception as e:
        print(f"Failed: {item.file_path}; error: {e}")

@app.get("/query_embeddings")
async def query_embeddings(paths: List[str] = Query(None)):    
    results = []
    try:
        for result in db.query_embeddings(paths):
            results.append({
                    "file_path": result[0], 
                    "distances": result[1], 
                    "audio_id": result[2]
                })
    except Exception as e:
        print(f"Error: {e}")
    return results

@app.delete("/flushdb")
async def flushdb():
    try:
        db.flush()
    except Exception as e:
        print(f"Error: {e}")