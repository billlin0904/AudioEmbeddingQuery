import librosa
import numpy as np
from chromadb import PersistentClient
from panns_inference import AudioTagging
from typing import List
from fastapi import Query
from separate import AudioSeparate
import os
from embedding_cache import EmbeddingCache

class AudioEmbeddingDatabase:
    def __init__(self, chroma_db_settings=None, collection_name="audio_embeddings", device='cuda'):
        # 初始化 ChromaDB 客戶端和集合
        self.client = PersistentClient(settings=chroma_db_settings)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.at = AudioTagging(device=device)
        self.separate = AudioSeparate(model_path="uvr5_weights/2_HP-UVR.pth", device=device)
        self.cache = EmbeddingCache()
        self.collection_name = collection_name
        
    def __get_or_create_collection(self):
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def separate_audio(self, file_path: str):        
        ins_root = "instrument"
        name = os.path.basename(file_path)
        self.separate.save_audio(file_path, ins_root = ins_root)
        save_path = os.path.join(ins_root, 'instrument_{}.wav'.format(name)) 
        return save_path

    def embed_and_save(self, path, audio_id):
        self.__get_or_create_collection()
        
         # 嘗試從緩存中讀取嵌入
        cached_embedding = self.cache.get_embedding(audio_id)
        if cached_embedding is not None:
            print(f"Using cached embedding for {audio_id}")
            self.collection.add(
                documents=[path],
                embeddings=[cached_embedding.tolist()],
                metadatas=[{"audio_id": audio_id}],
                ids=[audio_id]
            )
            return
        
        save_path = self.separate_audio(path)
        print(f"Process {save_path}")
        
        # 讀取音頻文件並進行處理
        audio, _ = librosa.core.load(save_path, sr=32000, mono=True)
        audio = audio[None, :]  # 添加新的維度

        try:
            print(f"Get {save_path} embedding")
            
            # 獲取嵌入向量
            _, embedding = self.at.inference(audio)
            embedding = embedding / np.linalg.norm(embedding)  # 正規化嵌入向量
            
            # 保存嵌入到 SQLite 緩存
            self.cache.save_embedding(audio_id, path, embedding)
            
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
        finally:
            os.remove(save_path)
            pass

    def get_embeddings(self, paths):
        # 對待檢索音頻批量抽取特徵，返回embedding
        embedding_list = []
        for x in paths:
            save_path = self.separate_audio(x)
            audio, _ = librosa.core.load(x, sr=32000, mono=True)
            audio = audio[None, :]
            try:
                _, embedding = self.at.inference(audio)
                embedding = embedding / np.linalg.norm(embedding)  # 正規化嵌入向量
                embedding_list.append(embedding)
            except Exception as e:
                print(f"Embedding Failed: {x}, Error: {e}")
            finally:
                os.remove(save_path)
        return np.array(embedding_list, dtype=np.float32).squeeze()

    def query_embeddings(self, paths, n_results=10):
        self.__get_or_create_collection()
        
        # 構建查詢的嵌入向量列表
        embeddings = []
        paths_to_process = []
    
        # 先嘗試從緩存中獲取嵌入
        for path in paths:
            cached_embedding = self.cache.get_embedding_from_file_path(path)
            if cached_embedding is not None:
                print(f"Using cached embedding for {path}")
                embeddings.append(cached_embedding.tolist())
            else:
                # 如果緩存中沒有，則需要進行處理
                paths_to_process.append(path)
        
        # 對緩存中不存在的音頻進行嵌入提取
        if paths_to_process:
            new_embeddings = self.get_embeddings(paths_to_process)
            for i, path in enumerate(paths_to_process):
                audio_id = os.path.basename(path)  # 使用文件名作為 audio_id
                embedding = new_embeddings[i]
                # 保存新提取的嵌入到緩存
                self.cache.save_embedding(audio_id, path, embedding)
                embeddings.append(embedding.tolist())
            
        res = []

        try:
            results = self.collection.query(
                query_embeddings=embeddings,  # 查詢的嵌入向量
                n_results=n_results,  # 每個查詢的返回結果數量限制
                include=["documents", "distances", "metadatas"]  # 返回文檔和距離信息
            )

            # 處理檢索結果
            for x in range(len(results["documents"])):
                for i in range(len(results["documents"][x])):
                    file_path = results["documents"][x][i]
                    distances = results["distances"][x][i]
                    metadata = results["metadatas"][x][i]
                    audio_id = metadata["audio_id"]
                    res.append((file_path, distances, audio_id))                
        except Exception as e:
            print("Failed to search vectors in Chroma: {}".format(e))
        return res
    
    def flush(self):
        self.client.reset()