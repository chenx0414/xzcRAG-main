import os
from typing import List

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
import requests  

class Vectordatabase:
    """使用 LangChain + DashScopeEmbeddings + FAISS + SiliconFlow Reranker"""

    def __init__(self, docs: List[str] = None, path: str = "database"):
        self.path = path
        self.embeddings = DashScopeEmbeddings(
            model="text-embedding-v3",
            dashscope_api_key="sk-"
        )
        self.vectorstore = None
        self.docs = docs or []

        # ==================== 新增：SiliconFlow Reranker 配置 ====================
        self.rerank_api_key = "sk-"  
        self.rerank_model = "BAAI/bge-reranker-v2-m3"   # 推荐（轻量高效）
        # 备选更强模型（2025 新款）：
        # self.rerank_model = "Qwen/Qwen3-Reranker-0.6B"   # 最轻量
        # self.rerank_model = "Qwen/Qwen3-Reranker-4B"     # 平衡最优
        # =====================================================================

    def get_vector(self) -> None:
        if not self.docs:
            raise ValueError("请传入文档列表！")
        documents = [Document(page_content=doc) for doc in self.docs] # 把 List[str] 转成 List[Document]，metadata 默认为空字典 {}
        self.vectorstore = FAISS.from_documents(  # 创建向量数据库
            documents=documents,
            embedding=self.embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        self.vectorstore.save_local(self.path)
        print(f"FAISS向量数据库创建完成，保存至: {self.path}")

    def load_vector(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"数据库路径不存在: {self.path}。请先调用 get_vector() 创建数据库！")
        self.vectorstore = FAISS.load_local(
            folder_path=self.path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"已加载FAISS向量数据库: {self.path}")

    # ====================== Reranker 方法 ======================
    def _rerank_with_siliconflow(self, query: str, documents: List[str], top_n: int = 8) -> List[str]:
        """SiliconFlow /v1/rerank 重排序"""
        url = "https://api.siliconflow.cn/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.rerank_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True,      # 返回原文
            # "instruction": "请根据与查询的相关性从高到低排序文档"  # Qwen3 模型可选
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            # results 已按 relevance_score 从高到低排序
            return [item["document"]["text"] for item in data.get("results", [])]
        except Exception as e:
            print(f"⚠️ Rerank API 调用失败: {e}，回退到原始顺序")
            return documents[:top_n]

    # ====================== query 方法 ======================
    def query(self, query: str, k: int = 3, retrieve_k: int = 25) -> List[str]:
        """向量粗召回 + Reranker 重排序"""
        if self.vectorstore is None:
            self.load_vector()

        # 第1步：向量检索更多候选（提升 recall）
        results = self.vectorstore.similarity_search(query, k=retrieve_k)
        candidates = [doc.page_content for doc in results]

        # 第2步：Reranker 精排（提升 precision）
        if len(candidates) > k and k > 0:
            return self._rerank_with_siliconflow(query, candidates, top_n=k)
        return candidates[:k]