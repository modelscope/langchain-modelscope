"""Test ModelscopeEndpoint API wrapper."""

from typing import AsyncIterator, Iterator

import pytest

from langchain_modelscope.llms import ModelScopeEndpoint


@pytest.fixture
def llm():
    return ModelScopeEndpoint(model="Qwen/Qwen2.5-Coder-32B-Instruct")  # type: ignore


def test_modelscope_call(llm) -> None:
    """Test valid call to Modelscope."""
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)


def test_modelscope_streaming(llm) -> None:
    """Test streaming call to Modelscope."""
    generator = llm.stream("write a quick sort in python")
    stream_results_string = ""
    assert isinstance(generator, Iterator)

    for chunk in generator:
        assert isinstance(chunk, str)
        stream_results_string = chunk
    assert len(stream_results_string.strip()) > 1


async def test_modelscope_call_async(llm) -> None:
    output = await llm.ainvoke("write a quick sort in python")
    assert isinstance(output, str)


async def test_modelscope_streaming_async(llm) -> None:
    generator = llm.astream("write a quick sort in python")
    stream_results_string = ""
    assert isinstance(generator, AsyncIterator)

    async for chunk in generator:
        assert isinstance(chunk, str)
        stream_results_string = chunk
    assert len(stream_results_string.strip()) > 1
