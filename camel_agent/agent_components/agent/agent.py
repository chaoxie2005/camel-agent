from typing import Optional, Union, List, Callable, Type

from camel.agents import ChatAgent
from camel_agent.agent_components.llm.model import ModelClient
from camel.messages import BaseMessage
from camel.memories import AgentMemory
from camel.toolkits import FunctionTool
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

        self.agent = ChatAgent(
            model=self.model_client.model,
            system_message=system_prompt,
            memory=memory,
            message_window_size=message_window_size,
            summarize_threshold=summarize_threshold,
            output_language=output_language,
            tools=tools,
        )

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
    
    
    
