from dataclasses import dataclass

from camel.messages import OpenAIMessage


@dataclass(frozen=True)
class ContextInput:
    """上下文管理器的输入。

    Attributes:
        system_messages: 必须保留的系统消息。
        history_messages: 按时间从旧到新排列的对话历史。
        current_messages: 必须保留的当前轮消息。
        retrieved_messages: 按相关性从高到低排列的长期记忆召回消息。
    """

    system_messages: list[OpenAIMessage]
    history_messages: list[OpenAIMessage]
    current_messages: list[OpenAIMessage]
    retrieved_messages: list[OpenAIMessage]


@dataclass(frozen=True)
class ContextResult:
    """上下文管理器的输出。

    Attributes:
        messages: 最终发送给模型的消息。
        token_count: 最终消息占用的 token 数量。
        dropped_count: 因清洗或预算不足而丢弃的消息数量。
        summarized: 是否执行了摘要。现在阶段始终为 False，未来可能会支持摘要功能。
    """

    messages: list[OpenAIMessage]
    token_count: int
    dropped_count: int
    summarized: bool
