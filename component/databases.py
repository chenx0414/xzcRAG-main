import os
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document


class Vectordatabase:
    """使用 LangChain + DashScopeEmbeddings + FAISS 的向量数据库"""

    def __init__(self, docs: List[str] = None, path: str = "database"):
        self.path = path
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key="sk-c322e63a35e6494682b683f3a68736b1"
        )
        self.vectorstore = None
        self.docs = docs or []

    def get_vector(self) -> None:
        """创建向量数据库并自动持久化"""
        if not self.docs:
            raise ValueError("请传入文档列表！")
        
        # 使用 LangChain 官方函数：自动 embedding + 建索引 + 持久化
        documents = [Document(page_content=doc) for doc in self.docs]
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE   # 与原 Chroma 保持一致（余弦相似度）
        )
        # 保存到本地（会生成 index.faiss + index.pkl）
        self.vectorstore.save_local(self.path)
        print(f"✅ FAISS 向量数据库创建完成，已保存至: {self.path}")

    def load_vector(self) -> None:
        """加载已存在的向量数据库"""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"数据库路径不存在: {self.path}。请先调用 get_vector() 创建数据库！")
        
        self.vectorstore = FAISS.load_local(
            folder_path=self.path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True   # LangChain 0.2+ 必须加
        )
        print(f"✅ 已加载 FAISS 向量数据库: {self.path}")

    def query(self, query: str, k: int = 2) -> List[str]:
        """语义检索 Top-k"""
        if self.vectorstore is None:
            self.load_vector()   # 自动加载
        
        # 使用 LangChain 内置相似度检索（接口与原来完全一致）
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]