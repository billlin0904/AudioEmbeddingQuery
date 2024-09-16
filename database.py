import librosa
import numpy as np
from chromadb import PersistentClient
from typing import List
from fastapi import Query
from separate import AudioSeparate
import os
from embedding_cache import EmbeddingCache
import openl3  # 新添加的导入

class AudioEmbeddingDatabase:
    def __init__(self, chroma_db_settings=None, collection_name="audio_embeddings", device='cuda'):
        # 初始化 ChromaDB 客户端和集合
        self.client = PersistentClient(settings=chroma_db_settings)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.separate = AudioSeparate(model_path="uvr5_weights/2_HP-UVR.pth", device=device)
        self.cache = EmbeddingCache()
        self.collection_name = collection_name
        
    def __get_or_create_collection(self):
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def separate_audio(self, file_path: str):        
        ins_root = "instrument"
        name = os.path.basename(file_path)
        self.separate.save_audio(file_path, ins_root=ins_root)
        save_path = os.path.join(ins_root, f'instrument_{name}.wav')
        return save_path

    def embed_and_save(self, path, audio_id):
        self.__get_or_create_collection()
        
        # 尝试从缓存中读取嵌入
        cached_embedding = self.cache.get_embedding(audio_id)
        if cached_embedding is not None:
            print(f"使用缓存的嵌入：{audio_id}")
            self.collection.add(
                documents=[path],
                embeddings=[cached_embedding.tolist()],
                metadatas=[{"audio_id": audio_id}],
                ids=[audio_id]
            )
            return
        
        save_path = self.separate_audio(path)
        print(f"正在处理 {save_path}")
        
        # 读取音频文件并进行处理
        audio, sr = librosa.load(save_path, sr=None, mono=True)

        try:
            print(f"获取 {save_path} 的嵌入")
            
            # 使用 openl3 获取嵌入
            embedding, _ = openl3.get_audio_embedding(
                audio,
                sr,
                hop_size=0.1,
                embedding_size=512,
                content_type='music'  # 或 'env'，根据您的使用场景
            )
            # 对时间帧进行平均，得到单个嵌入向量
            embedding = np.mean(embedding, axis=0)
            embedding = embedding / np.linalg.norm(embedding)  # 归一化嵌入向量
            
            # 将嵌入保存到 SQLite 缓存
            self.cache.save_embedding(audio_id, path, embedding)
            
            embedding = embedding.tolist()  # 转换为 Python 列表
            
            # 将嵌入插入到 ChromaDB
            self.collection.add(
                documents=[path],
                embeddings=[embedding],
                metadatas=[{"audio_id": audio_id}],
                ids=[audio_id]
            )

            print(f"成功插入并保存路径：{path}，ID：{audio_id}")
        except Exception as e:
            print(f"失败：{path}；错误：{e}")
        finally:
            os.remove(save_path)

    def get_embeddings(self, paths):
        # 批量提取查询音频的嵌入，返回 embedding
        embedding_list = []
        for x in paths:            
            try:
                save_path = self.separate_audio(x)
                audio, sr = librosa.load(save_path, sr=None, mono=True)
                # 使用 openl3 获取嵌入
                embedding, _ = openl3.get_audio_embedding(
                    audio,
                    sr,
                    hop_size=0.1,
                    embedding_size=512,
                    content_type='music'
                )
                embedding = np.mean(embedding, axis=0)
                embedding = embedding / np.linalg.norm(embedding)
                embedding_list.append(embedding)
            except Exception as e:
                print(f"嵌入失败：{x}，错误：{e}")
            finally:
                os.remove(save_path)
        return np.array(embedding_list, dtype=np.float32)

    def query_embeddings(self, paths, n_results=10):
        self.__get_or_create_collection()
        
        # 构建查询的嵌入向量列表
        embeddings = []
        not_found = False
    
        # 首先尝试从缓存中获取嵌入
        for path in paths:
            cached_embedding = self.cache.get_embedding_from_file_path(path)
            if cached_embedding is not None:
                print(f"使用缓存的嵌入：{path}")
                embeddings.append(cached_embedding.tolist())
            else:
                not_found = True
                break
                
        if not_found:
            embeddings.clear()
            embeddings = self.get_embeddings(paths)
            
        res = []

        try:
            results = self.collection.query(
                query_embeddings=embeddings,  # 查询的嵌入向量
                n_results=n_results,  # 每个查询的返回结果数量限制
                include=["documents", "distances", "metadatas"]  # 返回文档和距离信息
            )

            # 处理检索结果
            for x in range(len(results["documents"])):
                for i in range(len(results["documents"][x])):
                    file_path = results["documents"][x][i]
                    distances = results["distances"][x][i]
                    metadata = results["metadatas"][x][i]
                    audio_id = metadata["audio_id"]
                    res.append((file_path, distances, audio_id))                
        except Exception as e:
            print(f"在 Chroma 中搜索向量失败：{e}")
        return res
    
    def delete_embedding(self, audio_ids):
        for audio_id in audio_ids:
            try:
                # 从 ChromaDB 集合中删除嵌入
                self.collection.delete(ids=[audio_id])
    
                # 从嵌入缓存中删除嵌入
                self.cache.delete_embedding(audio_id)
    
                print(f"成功删除嵌入，ID：{audio_id}")
            except Exception as e:
                print(f"删除嵌入失败，ID：{audio_id}，错误：{e}")        
            
    def flush(self):
        self.cache.clear_cache()
        self.client.reset()
