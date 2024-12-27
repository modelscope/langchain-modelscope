"""Test ChatModelScope chat model."""

from typing import Type

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_modelscope.chat_models import ModelScopeChatEndpoint


class TestModelScopeChatEndpointIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ModelScopeChatEndpoint]:
        return ModelScopeChatEndpoint

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "temperature": 0,
        }

    @property
    def returns_usage_metadata(self) -> bool:
        return False
