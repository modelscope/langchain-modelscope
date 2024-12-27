"""Test ModelScope embeddings."""

from typing import Type

from langchain_tests.integration_tests import EmbeddingsIntegrationTests

from langchain_modelscope.embeddings import ModelScopeEmbeddings


class TestModelScopeEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[ModelScopeEmbeddings]:
        return ModelScopeEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model_id": "damo/nlp_corom_sentence-embedding_english-base"}
