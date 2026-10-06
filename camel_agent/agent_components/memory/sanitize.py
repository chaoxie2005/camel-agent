from typing import Any, Dict, List, Tuple

from camel.memories.context_creators.score_based import ScoreBasedContextCreator
from camel.messages import OpenAIMessage

REASONING_KEY = "reasoning_content"


def _tool_call_id(tool_call: Dict[str, Any]) -> Any:
    return tool_call.get("id") or tool_call.get("tool_call_id")


def sanitize_openai_messages(
    messages: List[OpenAIMessage],
) -> List[OpenAIMessage]:
    """去除重复消息，并修复 assistant(tool_calls) 与 tool 结果的配对。

    1. 按 (role, content, tool_calls, tool_call_id) 去重（向量召回会注入
       与聊天历史重复的记录）；
    2. 删除 tool_call_id 无对应 assistant tool_calls 的孤儿 tool 消息；
    3. assistant(tool_calls) 中无 tool 结果响应的调用项被剥离；若全部被
       剥离且无文本内容，则整条丢弃；否则保留文本/剩余调用项。
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
        messages, tokens = super().create_context(records)
        cleaned = sanitize_openai_messages(messages)
        if len(cleaned) != len(messages):
            tokens = self.token_counter.count_tokens_from_messages(cleaned)
        return cleaned, tokens
