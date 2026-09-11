"""LabLens —— 中文检验报告智能体核心包。

分层（详见 README 架构图）：

- ``schema``     三层数据契约（抽取 / 领域对象 / 解读），全部 Pydantic 强校验
- ``units``      单位归一的纯函数（写法归一 + 已登记因子换算）
- ``rules``      **唯一的数值判定层**，100% 确定性、可单测、可复现
- ``retriever``  约束检索 + 拒答（先按指标名硬过滤，再语义排序）
- ``providers``  LLM 传输层，OpenAI 兼容协议，支持云端与本地 vLLM
- ``extract``    Stage 1：视觉模型把报告图抽成结构化裸数据
- ``explain``    Stage 3：解释层，只消费已判定的结果
- ``store``      DuckDB 纵向趋势存储
"""

__version__ = "0.1.0"
