import librosa
import numpy as np
from chromadb import PersistentClient
from chromadb.config import Settings
from panns_inference import AudioTagging
from typing import List
from pydantic import BaseModel
from typing import Union
from fastapi import FastAPI, Query

class AudioEmbeddingDatabase:
    def __init__(self, chroma_db_settings=None, collection_name="audio_embeddings", device='cpu'):
        # 初始化 ChromaDB 客戶端和集合
        self.client = PersistentClient(settings=chroma_db_settings)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.at = AudioTagging(checkpoint_path=None, device=device)

    def embed_and_save(self, path, audio_id):
        print(f"Process {path}")
        # 讀取音頻文件並進行處理
        audio, _ = librosa.core.load(path, sr=32000, mono=True)
        audio = audio[None, :]  # 添加新的維度

        try:
            # 獲取嵌入向量
            _, embedding = self.at.inference(audio)
            embedding = embedding / np.linalg.norm(embedding)  # 正規化嵌入向量
            embedding = embedding.tolist()[0]  # 轉換為 Python 列表
            
            # 插入嵌入向量到 ChromaDB
            self.collection.add(
                documents=[path],
                embeddings=[embedding],
                metadatas=[{"audio_id": audio_id}],
                ids = [audio_id]
            )

            print(f"Successfully inserted and saved path: {path} with ID: {audio_id}")
        except Exception as e:
            print(f"Failed: {path}; error: {e}")

    def get_embeddings(self, paths):
        # 對待檢索音頻批量抽取特徵，返回embedding
        embedding_list = []
        for x in paths:
            audio, _ = librosa.core.load(x, sr=32000, mono=True)
            audio = audio[None, :]
            try:
                _, embedding = self.at.inference(audio)
                embedding = embedding / np.linalg.norm(embedding)  # 正規化嵌入向量
                embedding_list.append(embedding)
            except Exception as e:
                print(f"Embedding Failed: {x}, Error: {e}")
        return np.array(embedding_list, dtype=np.float32).squeeze()

    def query_embeddings(self, paths, n_results=5):
        # 使用 Chroma 進行向量檢索並返回結果
        embeddings = self.get_embeddings(paths).tolist()
        res = []

        try:
            results = self.collection.query(
                query_embeddings=embeddings,  # 查詢的嵌入向量
                n_results=n_results,  # 每個查詢的返回結果數量限制
                include=["documents", "distances", "metadatas"]  # 返回文檔和距離信息
            )

            # 處理檢索結果
            for x in range(len(results["documents"])):
                file_path = results["documents"][x]  # 返回的結果文件列表
                distances = results["distances"][x]  # 返回的距離列表
                metadata = results["metadatas"][x]
                audio_id = metadata[1]["audio_id"]
                res.append((file_path, distances, audio_id))
        except Exception as e:
            print("Failed to search vectors in Chroma: {}".format(e))
        return res

app = FastAPI()
db = AudioEmbeddingDatabase()

class EmbeddingItem(BaseModel):
    file_path: str
    audio_id: str

@app.post("/embed_and_save")
async def embed_and_save(item: EmbeddingItem):
    db.embed_and_save(item.file_path, item.audio_id)

@app.get("/query_embeddings")
async def query_embeddings(paths: List[str] = Query(None)):
    results = []
    for result in db.query_embeddings(paths):
        results.append({
                "file_path": result[0], 
                "distances": result[1], 
                "audio_id": result[2]
            })
    return results