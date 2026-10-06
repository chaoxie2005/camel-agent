from typing import Any, Dict, List, Tuple

from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.messages import OpenAIMessage

REASONING_KEY = "reasoning_content"


def _tool_call_id(tool_call: Dict[str, Any]) -> Any:
    """读取工具调用的唯一标识。

    兼容 OpenAI 工具调用使用的 ``id`` 字段，以及部分消息结构使用的
    ``tool_call_id`` 字段。当两个字段同时存在时优先返回 ``id``。

    Args:
        tool_call: 工具调用数据。

    Returns:
        工具调用标识；两个字段都不存在或值为空时返回 ``None``。
    """
    return tool_call.get("id") or tool_call.get("tool_call_id")


def sanitize_openai_messages(
    messages: List[OpenAIMessage],
) -> List[OpenAIMessage]:
    """去除重复消息，并修复 assistant(tool_calls) 与 tool 结果的配对。

    处理过程包括：按角色、内容和工具调用信息去重；删除没有对应
    assistant 工具调用的孤立 tool/function 消息；移除没有结果响应的
    assistant 工具调用。对于 DeepSeek assistant 历史消息，还会在缺失时
    补充空的 ``reasoning_content`` 字段。

    Args:
        messages: 待清洗的 OpenAI 格式消息列表。函数不会修改传入列表，
            但返回结果中未被修补的消息可能仍与输入引用同一字典对象。

    Returns:
        去重并修复工具调用配对后的消息列表，保持保留消息的原始顺序。
    """
    # ---- 1. 去重 ----
    seen: set = set()
    unique: List[OpenAIMessage] = []
    for msg in messages:
        key = (
            msg.get("role"),
            str(msg.get("content")),
            str(msg.get("tool_calls")),
            msg.get("tool_call_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(msg)

    # ---- 2. assistant 声明的 call id 与 tool 结果的交集 = 有效配对 ----
    declared = {
        _tool_call_id(tc)
        for msg in unique
        if msg.get("role") == "assistant" and msg.get("tool_calls")
        for tc in msg["tool_calls"]
    }
    responded = {
        msg.get("tool_call_id")
        for msg in unique
        if msg.get("role") in ("tool", "function") and msg.get("tool_call_id")
    }
    answered = declared & responded

    # ---- 3. 配对修复 ----
    cleaned: List[OpenAIMessage] = []
    for msg in unique:
        role = msg.get("role")
        # DeepSeek 思考模式：assistant 历史消息必须回传 reasoning_content，
        # 缺失（尤其含 tool_calls 时）会 400。camel 序列化不输出该字段，
        # 统一补空串兜底（有真实值则保留）。
        if role == "assistant" and REASONING_KEY not in msg:
            msg = {**msg, REASONING_KEY: ""}
        if role in ("tool", "function"):
            if msg.get("tool_call_id") in answered:
                cleaned.append(msg)
            continue

        if role == "assistant" and msg.get("tool_calls"):
            calls = msg["tool_calls"]
            kept = [tc for tc in calls if _tool_call_id(tc) in answered]
            if len(kept) == len(calls):
                cleaned.append(msg)
            elif kept:
                patched = dict(msg)
                patched["tool_calls"] = kept
                cleaned.append(patched)  # type: ignore[arg-type]
            elif msg.get("content"):
                patched = {
                    k: v for k, v in msg.items() if k != "tool_calls"
                }
                cleaned.append(patched)  # type: ignore[arg-type]
            continue

        cleaned.append(msg)

    return cleaned


class SanitizingContextCreator(ScoreBasedContextCreator):
    """ScoreBasedContextCreator + 发送前消息清洗兜底。"""

    def create_context(
        self, records: List
    ) -> Tuple[List[OpenAIMessage], int]:
        """创建上下文并在发送给模型前清洗消息。

        先使用父类的评分和 token 限制生成上下文，再清理重复消息及无效
        工具调用关系。如果清洗改变了消息数量，则重新计算 token 数量。

        Args:
            records: 用于构建上下文的记忆记录列表，具体记录格式遵循
                ``ScoreBasedContextCreator.create_context`` 的要求。

        Returns:
            一个二元组，第一项是清洗后的 OpenAI 格式消息列表，第二项是
            对应的 token 数量。
        """
        messages, tokens = super().create_context(records)
        cleaned = sanitize_openai_messages(messages)
        if len(cleaned) != len(messages):
            tokens = self.token_counter.count_tokens_from_messages(cleaned)
        return cleaned, tokens
