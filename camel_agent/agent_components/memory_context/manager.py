from camel.messages import OpenAIMessage
from camel.utils.token_counting import BaseTokenCounter

from camel_agent.agent_components.memory.sanitize import (
    sanitize_openai_messages,
)

from .budget import ContextBudget
from .types import ContextInput, ContextResult


class ContextManager:
    """按优先级和 token 预算选择 Agent 上下文。"""

    def __init__(
        self,
        token_counter: BaseTokenCounter,
        budget: ContextBudget,
    ) -> None:
        """初始化上下文管理器。

        Args:
            token_counter: CAMEL token 计数器。
            budget: 上下文 token 预算。
        """
        self.token_counter = token_counter
        self.budget = budget

    def build_context(self, context_input: ContextInput) -> ContextResult:
        """清洗并按预算组装上下文。

        系统消息和当前轮消息固定保留。剩余空间优先容纳最近的历史
        对话，之后再容纳最近的长期记忆召回消息。历史消息以用户消息
        开始的完整轮次为单位选择，避免拆散工具调用链。

        Args:
            context_input: 按来源分组的上下文消息。

        Returns:
            选取后的消息、token 数量和丢弃数量。该简单版本不执行摘要。

        Raises:
            ValueError: 系统消息和当前轮消息本身已经超过输入预算。
        """
        system_messages = sanitize_openai_messages(
            list(context_input.system_messages)
        )
        current_messages = sanitize_openai_messages(
            list(context_input.current_messages)
        )
        must_required = system_messages + current_messages # 必须保留的消息

        must_required_tokens = self._count_tokens(must_required)
        if must_required_tokens > self.budget.input_token_limit:
            raise ValueError(
                "系统消息和当前轮消息超过输入 token 预算："
                f"{must_required_tokens} > {self.budget.input_token_limit}"
            )

        history = sanitize_openai_messages(list(context_input.history_messages))
        retrieved = sanitize_openai_messages(
            list(context_input.retrieved_messages)
        )

        # 选择最近的历史对话 需保证历史对话的连续性
        selected_history = self._select_recent_history(
            self._group_conversation_units(history),
            prefix=system_messages,
            suffix=current_messages,
        )
        # 选择相关性最高的长期记忆召回消息 不要求连续性
        selected_retrieved = self._select_retrieved_messages(
            self._group_conversation_units(retrieved),
            prefix=system_messages,
            suffix=selected_history + current_messages,
        )

        messages = sanitize_openai_messages(
            system_messages # 系统消息
            + selected_retrieved # 长期记忆召回消息
            + selected_history # 历史对话
            + current_messages # 当前轮消息
        )
        token_count = self._count_tokens(messages)
        # 计算输入消息总数
        input_count = sum(
            len(group)
            for group in (
                context_input.system_messages,
                context_input.history_messages,
                context_input.current_messages,
                context_input.retrieved_messages,
            )
        )
        return ContextResult(
            messages=messages,
            token_count=token_count,
            dropped_count=input_count - len(messages),
            summarized=False,
        )

    def _select_recent_history(
        self,
        units: list[list[OpenAIMessage]],
        prefix: list[OpenAIMessage],
        suffix: list[OpenAIMessage],
    ) -> list[OpenAIMessage]:
        """从新到旧连续选择能够完整放入预算的历史轮次。

        遇到第一个无法放入预算的轮次后立即停止，不再选择更早的
        历史，从而保证最终保留的是一段连续的最近对话。

        Args:
            units: 按时间从旧到新排列的历史对话单元。
            prefix: 最终上下文中位于所选单元之前的固定消息。
            suffix: 最终上下文中位于所选单元之后的固定消息。

        Returns:
            按原时间顺序排列的连续历史消息。
        """
        selected: list[list[OpenAIMessage]] = []
        for unit in reversed(units):
            candidate_units = [unit, *selected]
            candidate = (
                prefix
                + self._flatten(candidate_units)
                + suffix
            )
            if self._count_tokens(candidate) <= self.budget.input_token_limit:
                selected = candidate_units
            else:
                break
        return self._flatten(selected)

    def _select_retrieved_messages(
        self,
        units: list[list[OpenAIMessage]],
        prefix: list[OpenAIMessage],
        suffix: list[OpenAIMessage],
    ) -> list[OpenAIMessage]:
        """按相关性顺序选择能够完整放入预算的召回消息。

        召回单元应按相关性从高到低传入。某个单元无法放入预算时，
        跳过该单元并继续检查后面的结果，因为离散的长期记忆不要求
        时间连续。

        Args:
            units: 按相关性从高到低排列的召回消息单元。
            prefix: 最终上下文中位于所选单元之前的固定消息。
            suffix: 最终上下文中位于所选单元之后的固定消息。

        Returns:
            按原相关性顺序排列且能够放入预算的召回消息。
        """
        selected: list[list[OpenAIMessage]] = []
        for unit in units:
            candidate_units = [*selected, unit]
            candidate = (
                prefix
                + self._flatten(candidate_units)
                + suffix
            )
            if self._count_tokens(candidate) <= self.budget.input_token_limit:
                selected = candidate_units
                continue

            # 第一版只截断单条文本召回消息。完整对话单元不能安全地
            # 拆分，因此包含多条消息时仍然整体跳过。
            if len(unit) != 1:
                continue
            truncated = self._truncate_message_to_fit(
                unit[0],
                prefix=prefix + self._flatten(selected),
                suffix=suffix,
            )
            if truncated is not None:
                selected.append([truncated])
        return self._flatten(selected)

    def _truncate_message_to_fit(
        self,
        message: OpenAIMessage,
        prefix: list[OpenAIMessage],
        suffix: list[OpenAIMessage],
    ) -> OpenAIMessage | None:
        """将一条文本消息截断到剩余 token 预算内。

        使用二分查找确定能够保留的最长文本，并在末尾添加截断标记。
        方法会复制消息字典，不会修改调用方传入的原始消息。当前仅处理
        ``content`` 为非空字符串的消息；多模态内容等其他格式返回
        ``None``。

        Args:
            message: 需要尝试截断的召回消息。
            prefix: 最终上下文中位于该消息之前的固定消息。
            suffix: 最终上下文中位于该消息之后的固定消息。

        Returns:
            能够放入预算的截断消息。如果消息不支持文本截断，或者连
            截断标记都无法放入剩余预算，则返回 ``None``。
        """
        content = message.get("content")
        if not isinstance(content, str) or not content:
            return None

        marker = "\n...[内容已截断]"
        left = 0
        right = len(content)
        best: OpenAIMessage | None = None

        while left <= right:
            middle = (left + right) // 2
            candidate = dict(message)
            candidate["content"] = content[:middle].rstrip() + marker
            candidate_messages = prefix + [candidate] + suffix

            if (
                self._count_tokens(candidate_messages)
                <= self.budget.input_token_limit
            ):
                best = candidate  # type: ignore[assignment]
                left = middle + 1
            else:
                right = middle - 1

        return best

    @staticmethod
    def _group_conversation_units(
        messages: list[OpenAIMessage],
    ) -> list[list[OpenAIMessage]]:
        """按用户消息边界划分不可拆分的对话单元。

        每条 user 消息开启一个新单元，之后的 assistant、tool 和 function
        消息都归入该单元，因此一次工具调用及其结果会一起保留或丢弃。
        第一条 user 消息之前的消息分别作为独立单元处理。

        Args:
            messages: 按时间从旧到新排列的消息。

        Returns:
            按原时间顺序排列的消息单元列表。
        """
        units: list[list[OpenAIMessage]] = [] # 对话单元列表
        current: list[OpenAIMessage] = [] # 当前对话单元
        for message in messages:
            if message.get("role") == "user":
                if current:
                    units.append(current)
                current = [message]
            elif current:
                current.append(message)
            else:
                units.append([message])
        if current:
            units.append(current)
        return units

    def _count_tokens(self, messages: list[OpenAIMessage]) -> int:
        """统计一组 OpenAI 消息的 token 数量。"""
        return self.token_counter.count_tokens_from_messages(messages)

    @staticmethod
    def _flatten(
        units: list[list[OpenAIMessage]],
    ) -> list[OpenAIMessage]:
        """将消息单元展平为消息列表。"""
        return [message for unit in units for message in unit]
