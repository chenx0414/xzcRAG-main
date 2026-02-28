from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

class ReadFile:

    #传入文件夹路径
    def __init__(self, path):
        self.path = path
        
    # 加载文件夹下所有文档（PDF / MD / TXT），返回 List[Document]
    # 每个 Document 自动携带 metadata（source 文件路径、page 页码等）
    def load_documents(self):
        loaders = [
            DirectoryLoader(self.path, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(self.path, glob="**/*.txt", loader_cls=TextLoader,
                            loader_kwargs={"encoding": "utf-8"}),
            DirectoryLoader(self.path, glob="**/*.md",  loader_cls=TextLoader,
                            loader_kwargs={"encoding": "utf-8"}),
        ]
        all_docs = []
        for loader in loaders:
            all_docs.extend(loader.load())
        return all_docs

    # 切分数据，传入一个字符串，返回一个字块列表
    # 使用langchain的递归拆分法
    @classmethod
    def chunk_documents(cls, documents, max_token_len: int = 600, cover_content: int = 150):
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
            chunk_size=max_token_len,
            chunk_overlap=cover_content,
        )
        return splitter.split_documents(documents)
            

    # 整合函数：加载所有文档并切分，返回 List[Document]
    # 每个 Document 包含：
    #   .page_content  → 文本内容
    #   .metadata      → {"source": "文件路径", "page": 页码} 等信息
    def get_all_chunk_content(self, max_len: int = 600, cover_len: int = 150):
        documents = self.load_documents()
        chunks = self.chunk_documents(documents, max_len, cover_len)
        return chunks
    

# ── 使用示例 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    reader = ReadFile("./documents")
    chunks = reader.get_all_chunk_content(max_len=600, cover_len=150)

    print(f"共切分出 {len(chunks)} 个文本块\n")
    for i, chunk in enumerate(chunks[:3]):  # 打印前3块示例
        print(f"── 块 {i+1} ──────────────────────────")
        print(f"来源: {chunk.metadata}")
        print(f"内容: {chunk.page_content[:100]}...")
        print()
    
         