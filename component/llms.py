from langchain_openai import ChatOpenAI
from component.databases import Vectordatabase
from component.data_chunker import ReadFile
from component.prompts import PromptEngineer
from typing import List
import os

class Openai_model:
    def __init__(self, temperature: float = 0.3):
        """初始化大模型 + 向量数据库 + 多策略 Prompt 引擎"""
        
        # 初始化 DeepSeek-V3.2（SiliconFlow）
        self.model = ChatOpenAI(
            model="deepseek-ai/DeepSeek-V3.2",
            api_key="sk-nifczyridmdpikqljtjodsmirloaqqegzbuhrmnizwehuidf",
            base_url="https://api.siliconflow.cn/v1",
            temperature=temperature,
            max_tokens=2048
        )

        # 加载文档并构建向量数据库
        print("正在加载文档并构建 FAISS 向量数据库")
        chunks = ReadFile("data").get_all_chunk_content(max_len=600, cover_len=150)
        
        self.db = Vectordatabase([doc.page_content for doc in chunks])
        
        if os.path.exists("database") and os.path.isdir("database"):
            self.db.load_vector()
        else:
            self.db.get_vector() # 首次运行使用此行    

        # 初始化多策略 Prompt 引擎
        self.prompt_engineer = PromptEngineer()
        
        print("Openai_model 初始化完成 | 多策略 Prompt 已启用 (CoT + Few-shot + 上下文压缩)")

    def chat(self, 
             question: str,
             use_cot: bool = True,
             use_fewshot: bool = True,
             use_compression: bool = True,
             k: int = 6,
             retrieve_k: int = 30) -> str:
        """
        增强版对话接口（默认全策略开启，效果最佳）
        参数说明：
            use_cot          : 是否启用思维链 CoT
            use_fewshot      : 是否注入 Few-shot 示例
            use_compression  : 是否让模型先进行上下文压缩
            k                : 最终返回的参考片段数
            retrieve_k       : 向量检索候选数量（给 Reranker 使用）
        """
        # 1. 向量检索 + SiliconFlow Reranker
        raw_contexts: List[str] = self.db.query(
            query=question, 
            k=k, 
            retrieve_k=retrieve_k
        )

        # 2. 格式化为带编号的上下文（便于模型引用）
        context_str = "\n\n".join([f"[{i+1}] {text}" for i, text in enumerate(raw_contexts)])

        # 3. 生成多策略优化 Prompt
        prompt_template = self.prompt_engineer.build_prompt_template(
            use_cot=use_cot,
            use_fewshot=use_fewshot,
            use_compression=use_compression
        )

        # 直接把 Prompt 和 LLM 拼成一条链（LCEL 语法）
        chain = prompt_template | self.model

        # 4. 执行 Chain
        response = chain.invoke({
            "context": context_str,
            "question": question
        })

        return response.content if hasattr(response, 'content') else str(response)


# ── 测试使用（直接运行本文件即可测试） ─────────────────────────────────────
if __name__ == "__main__":
    model = Openai_model(temperature=0.3)

    test_question = "请问公司2024年的主要业务和核心优势是什么？"

    answer = model.chat(
        question=test_question,
        use_cot=True,
        use_fewshot=True,
        use_compression=True
    )

    print("\n" + "="*70)
    print("用户问题：", test_question)
    print("="*70)
    print("模型回答：\n")
    print(answer)
    print("="*70)