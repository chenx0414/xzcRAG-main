from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class PromptEngineer:
    """多策略 Prompt 模板体系（CoT + Few-shot + 上下文压缩）"""

    @staticmethod
    def get_system_prompt() -> str:
        return """你是一位严谨、专业、注重事实的中文知识问答助手。
核心规则：
- 必须严格基于「参考上下文」回答，绝不编造任何信息。
- 如果上下文无法充分回答问题，请明确回复：“根据提供的参考资料，我无法确定答案。”
- 使用简洁、专业、清晰的中文作答。
- 重要事实请用 [1][2] 等编号标注引用来源。
- 每条回答最后给出置信度（高/中/低）。"""

    @staticmethod
    def get_fewshot_examples(num_shots: int = 2) -> str:
        examples = [
            """【示例1】
问题：公司的法定代表人是谁？
参考上下文：
[1] 根据2024年工商登记信息，公司法定代表人为李明。
[2] 2025年暂无变更记录。

思考过程：
1. 问题分析：询问法定代表人。
2. 关键信息提取：核心事实为“李明”。
3. 逻辑推理：信息一致且为最新。
答案：
公司法定代表人为李明。[1]
置信度：高""",

            """【示例2】
问题：2025年公司营收是多少？
参考上下文：
[1] 2024年公司营收为2.8亿元。

思考过程：
1. 问题分析：询问2025年营收。
2. 关键信息提取：无2025年数据。
答案：
根据提供的参考资料，我无法确定2025年的公司营收情况。
置信度：中"""
        ]
        return "\n\n".join(examples[:num_shots]) + "\n\n--- 请严格按照以上示例格式进行回答 ---\n\n"

    @staticmethod
    def build_prompt_template(
        use_cot: bool = True,
        use_fewshot: bool = True,
        use_compression: bool = True
    ) -> ChatPromptTemplate:
        """构建最终的多策略 Prompt 模板（与 llms.py 完全匹配）"""
        
        system_msg = SystemMessagePromptTemplate.from_template(
            PromptEngineer.get_system_prompt()
        )

        # 条件指令
        compression_instruction = (
            "请先对参考上下文进行针对性的压缩：提取并总结与问题最相关的关键事实，去除冗余，保留核心证据（如数字、日期、人名、结论等）。\n\n"
            if use_compression else ""
        )

        cot_instruction = (
            "请使用 Chain-of-Thought（思维链）逐步推理：\n"
            "1. 问题分析\n"
            "2. 关键信息提取与压缩\n"
            "3. 逻辑推理\n"
            "4. 最终结论\n\n"
            if use_cot else ""
        )

        fewshot = PromptEngineer.get_fewshot_examples(2) if use_fewshot else ""

        human_template = f"""{fewshot}{compression_instruction}### 参考上下文（已按相关性从高到低排序）
{{context}}

### 用户问题
{{question}}

{cot_instruction}请按以下固定格式输出：

思考过程：
1. 问题分析：...
2. 关键信息提取：...
3. 逻辑推理：...
4. 最终结论：...

答案：
[这里写最终答案，必要时标注引用编号 [1][2]]

置信度：高/中/低"""

        return ChatPromptTemplate.from_messages([
            system_msg,
            HumanMessagePromptTemplate.from_template(human_template)
        ])