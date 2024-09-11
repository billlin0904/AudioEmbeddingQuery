import sqlite3
import numpy as np

class EmbeddingCache:
    def __init__(self, db_path="audio_cache.db"):
        self.db_path = db_path
        self._init_cache_db()

    def _init_cache_db(self):
        """初始化 SQLite 數據庫表結構"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_embeddings (
            audio_id TEXT PRIMARY KEY,
            file_path TEXT,
            embedding BLOB
        )
        ''')
        conn.commit()
        conn.close()

    def save_embedding(self, audio_id, file_path, embedding):
        """將嵌入保存到 SQLite 緩存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 將 numpy array 轉為二進制數據
        embedding_blob = embedding.tobytes()
        cursor.execute('''
        INSERT OR REPLACE INTO audio_embeddings (audio_id, file_path, embedding) 
        VALUES (?, ?, ?)
        ''', (audio_id, file_path, embedding_blob))
        conn.commit()
        conn.close()

    def get_embedding(self, audio_id):
        """從緩存中讀取嵌入向量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT embedding FROM audio_embeddings WHERE audio_id = ?
        ''', (audio_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            # 將二進制數據轉換為 numpy array
            return np.frombuffer(result[0], dtype=np.float32)
        return None
    
    def get_embedding_from_file_path(self, file_path):
        """根據文件路徑從緩存中讀取嵌入向量"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT embedding FROM audio_embeddings WHERE file_path = ?
        ''', (file_path,))
        result = cursor.fetchone()
        conn.close()
        if result:
            # 將二進制數據轉換為 numpy array
            return np.frombuffer(result[0], dtype=np.float32)
        return None

    def clear_cache(self):
        """清空緩存數據"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM audio_embeddings')
        conn.commit()
        conn.close()
