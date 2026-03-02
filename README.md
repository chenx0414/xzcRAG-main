
**一个极简、可魔改、开箱即用的中文 RAG 项目**，专为学习和快速验证而设计。

本项目完整实现了 RAG 的三大核心步骤：

1. **索引** — 加载 PDF/TXT/MD 并智能切块（中文标点优化）
2. **检索** — FAISS 向量粗召回 + SiliconFlow Reranker 精排（业界最佳实践）
3. **生成** — 多策略 Prompt（System + Few-shot + CoT + 上下文压缩） + DeepSeek-V3.2

---

### ✨ 项目亮点

- **自动向量数据库管理**：首次运行自动创建 `database/`，后续自动加载，零手动操作
- **SiliconFlow Reranker**：粗召回 + 精排，显著提升检索精度
- **高级 Prompt 体系**：CoT + Few-shot 示例 + 上下文压缩，三开关自由组合
- **LCEL Chain**：使用 LangChain 最新链式写法，支持流式、异步、输出解析器
- **中文深度优化**：切分器内置中文标点，嵌入模型使用 DashScope text-embedding-v3
- **生产级友好**：异常优雅降级、清晰日志、完整类型提示

---

### 📁 项目结构

```
tinyRAG/
├── README.md
├── requirements.txt
├── test.py                 # 测试入口（推荐）
├── component/
│   ├── data_chunker.py     # 文档加载 + 智能切分
│   ├── databases.py        # FAISS + SiliconFlow Reranker
│   ├── prompts.py          # 多策略 PromptEngineer
│   └── llms.py             # Openai_model（核心对话类）
├── data/                   # 你的知识库文件
│   ├── dpcq.txt
│   ├── README.md
│   └── 中华人民共和国消费者权益保护法.pdf
└── database/               # 自动生成的 FAISS 索引（首次运行自动创建）
    ├── index.faiss
    └── index.pkl
```

---

### 🚀 Quick Start

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

（requirements.txt 已包含最新 LangChain、DashScope、FAISS、requests 等）

#### 2. 准备数据

把你的文档放入 `data/` 文件夹（支持 PDF、TXT、MD）

#### 3. 运行测试

```bash
python test.py
# 或者直接运行核心类文件
python -m component.llms
```

首次运行会自动：
- 读取 `data/` 下所有文档
- 切分 → 向量化 → 保存到 `database/`
- 后续启动自动加载，速度极快

---

### 📖 使用示例

```python
from component.llms import Openai_model

rag = Openai_model(temperature=0.3)

answer = rag.chat(
    question="2024年公司主要业务和核心优势是什么？",
    k=6,                    # 最终给模型的参考片段数
    retrieve_k=30,          # 向量召回候选数
    use_cot=True,           # 思维链
    use_fewshot=True,       # Few-shot 示例
    use_compression=True    # 上下文压缩
)

print(answer)
```

**典型输出格式**（带思考过程 + 引用 + 置信度）：

```
思考过程：
1. 问题分析：询问公司业务和优势
2. 关键信息提取：...
...
答案：
公司2024年主要业务为... [1][3]
置信度：高
```

---

### 🔧 自定义与扩展

- **修改嵌入模型**：在 `databases.py` 修改 `DashScopeEmbeddings`
- **切换大模型**：在 `llms.py` 修改 `ChatOpenAI` 参数（支持任何 OpenAI 兼容接口）
- **调整 Prompt 策略**：调用 `chat()` 时传入 `use_xxx=False` 进行消融实验
- **增加流式输出**：可直接把 `chain` 改成 `chain.stream()`

---

### 🛠️ 实现细节

- **文档处理**：`RecursiveCharacterTextSplitter` + 中文标点分隔符
- **向量库**：FAISS（COSINE 相似度） + 自动持久化
- **检索**：`similarity_search(retrieve_k=30)` + `bge-reranker-v2-m3` 重排序
- **Prompt**：`PromptEngineer` 类动态生成，支持 8 种策略组合

---

### 📝 思考与后续优化（参考）

- 目前切块参数（600/150）对长文档效果很好，可根据实际数据微调
- 已支持 Reranker，后续可加入 HyDE、Query Rewrite、Agentic RAG
- 欢迎提交 PR 一起完善！

---

### 📚 参考文献

- Retrieval-Augmented Generation for Large Language Models: A Survey
- When Large Language Models Meet Vector Databases: A Survey
- Learning to Filter Context for Retrieval-Augmented Generation

**本项目完全开源，欢迎 Star & Fork！**

---

**Made with ❤️ for 学习 RAG 的朋友们**

如有问题或想一起优化，欢迎 Issues / Discussions！
