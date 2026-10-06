from typing import Optional, Union, List, Callable, Type

from camel.agents import ChatAgent
from camel_agent.agent_components.llm.model import ModelClient
from camel.messages import BaseMessage
from camel.memories import AgentMemory, MemoryRecord
from camel.toolkits import FunctionTool
from camel.types import OpenAIBackendRole
from pydantic import BaseModel


class ChatAgentClient:
    """聊天智能体客户端"""

    def __init__(
        self,
        model_client: ModelClient,
        system_prompt: Optional[Union[BaseMessage, str]] = None,
        memory: Optional[AgentMemory] = None,
        message_window_size: Optional[int] = 30,
        summarize_threshold: Optional[int] = 70,
        output_language: Optional[str] = "zh",
        tools: Optional[List[Union[FunctionTool, Callable]]] = None,
    ):
        self.model_client = model_client

        # ChatAgent.__init__ → init_messages() → clear_memory()
        # 会无条件清空 memory（chat_agent.py:2413），持久化历史会丢失。
        # 先快照非 SYSTEM 的历史记录（LongtermAgentMemory.retrieve 会把
        # 向量召回结果夹在中间，需按 timestamp 排回时间序、按 uuid 去重），
        # 待 Agent 初始化清空后再写回。
        saved_records: List[MemoryRecord] = []
        if memory is not None:
            seen: set = set()
            for context_record in memory.retrieve():
                record = context_record.memory_record
                if (
                    record.uuid in seen
                    or record.role_at_backend == OpenAIBackendRole.SYSTEM
                ):
                    continue
                seen.add(record.uuid)
                saved_records.append(record)
            saved_records.sort(key=lambda r: r.timestamp)

        self.agent = ChatAgent(
            model=self.model_client.model,
            system_message=system_prompt,
            memory=memory,
            message_window_size=message_window_size,
            summarize_threshold=summarize_threshold,
            output_language=output_language,
            tools=tools,
        )

        if saved_records:
            # Agent 初始化（清空+写 system）后，恢复历史对话
            self.agent.memory.write_records(saved_records)

    async def run_chat(
    self,
    input_message: Union[BaseMessage, str],
    response_format: Optional[Type[BaseModel]] = None,
    ) -> str:
        """运行聊天智能体"""

        model_result = await self.agent.astep(
            input_message,
            response_format=response_format,
        )

        return model_result.msgs[0].content # type: ignore
    
    
    
