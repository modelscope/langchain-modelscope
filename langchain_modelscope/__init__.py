from importlib import metadata

from langchain_modelscope.chat_models import ModelScopeChatEndpoint
from langchain_modelscope.embeddings import ModelScopeEmbeddings
from langchain_modelscope.llms import ModelScopeEndpoint

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ModelScopeChatEndpoint",
    "ModelScopeEndpoint",
    "ModelScopeEmbeddings",
    "__version__",
]
