import sqlite3
import numpy as np

class EmbeddingCache:
    def __init__(self, db_path="audio_cache.db"):
        self.db_path = db_path
        # 建立持久化的数据库连接
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_cache_db()

    def _init_cache_db(self):
        """初始化 SQLite 数据库表结构"""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_embeddings (
            audio_id TEXT PRIMARY KEY,
            file_path TEXT,
            embedding BLOB
        )
        ''')
        # 为 file_path 添加索引以提高查询性能
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_path ON audio_embeddings (file_path)')
        self.conn.commit()

    def save_embedding(self, audio_id, file_path, embedding):
        """将嵌入保存到 SQLite 缓存"""
        # 将 numpy array 转为二进制数据，并确保数据类型为 float32
        embedding_blob = embedding.astype(np.float32).tobytes()
        self.cursor.execute('''
        INSERT OR REPLACE INTO audio_embeddings (audio_id, file_path, embedding) 
        VALUES (?, ?, ?)
        ''', (audio_id, file_path, embedding_blob))
        self.conn.commit()

    def get_embedding(self, audio_id):
        """从缓存中读取嵌入向量"""
        self.cursor.execute('''
        SELECT embedding FROM audio_embeddings WHERE audio_id = ?
        ''', (audio_id,))
        result = self.cursor.fetchone()
        if result:
            # 将二进制数据转换为 numpy array
            return np.frombuffer(result[0], dtype=np.float32)
        return None

    def get_embedding_from_file_path(self, file_path):
        """根据文件路径从缓存中读取嵌入向量"""
        self.cursor.execute('''
        SELECT embedding FROM audio_embeddings WHERE file_path = ?
        ''', (file_path,))
        result = self.cursor.fetchone()
        if result:
            # 将二进制数据转换为 numpy array
            return np.frombuffer(result[0], dtype=np.float32)
        return None

    def delete_embedding(self, audio_id):
        """从缓存数据库中删除指定的嵌入"""
        try:
            self.cursor.execute("DELETE FROM audio_embeddings WHERE audio_id = ?", (audio_id,))
            self.conn.commit()
            print(f"已从缓存中删除嵌入，ID：{audio_id}")
        except Exception as e:
            print(f"从缓存中删除嵌入失败，ID：{audio_id}，错误：{e}")

    def clear_cache(self):
        """清空缓存数据"""
        self.cursor.execute('DELETE FROM audio_embeddings')
        self.conn.commit()

    def close(self):
        """关闭数据库连接"""
        self.conn.close()
