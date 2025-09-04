
from langchain_modelscope.chat_models import ModelScopeChatEndpoint

def test_think():
    llm = ModelScopeChatEndpoint(model="Qwen/Qwen3-235B-A22B", extra_body={"enable_thinking": False})
    prompt = "Tell me a joke."
    response = llm.invoke(prompt)
    print(response)
    
if __name__ == "__main__":
    test_think()