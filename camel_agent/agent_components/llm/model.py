from camel.models import ModelFactory
from camel.types import ModelPlatformType


class ModelClient:
    """模型客户端"""
    def __init__(self, api_key: str, url: str, temperature: float = 0.5):
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type="deepseek-flash",
            api_key=api_key,
            url=url,
            model_config_dict={
                "temperature": temperature,
            },
        )