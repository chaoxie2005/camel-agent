from dataclasses import dataclass


@dataclass
class ContextBudget:
    """上下文 token 预算配置。

    Attributes:
        max_tokens: 模型允许的总 token 数量。
        reserved_output_tokens: 为模型回答预留的 token 数量。
    """

    max_tokens: int
    reserved_output_tokens: int

    def __post_init__(self) -> None:
        """校验预算配置。"""
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens 不能小于 0")
        if self.reserved_output_tokens >= self.max_tokens:
            raise ValueError("reserved_output_tokens 必须小于 max_tokens")

    @property
    def input_token_limit(self) -> int:
        """返回可用于输入上下文的 token 数量。"""
        return self.max_tokens - self.reserved_output_tokens
