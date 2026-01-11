# packages/leann-core/src/leann/__init__.py
import os
import platform

# Fix OpenMP threading issues on macOS ARM64
if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["KMP_BLOCKTIME"] = "0"
    # Additional fixes for PyTorch/sentence-transformers on macOS ARM64 only in CI
    if os.environ.get("CI") == "true":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from .api import LeannBuilder, LeannChat, LeannSearcher
except ImportError as e:
    # Allow leann to be imported even if backends are missing
    # (useful for standalone analysis or CLI tools)
    LeannBuilder = None
    LeannChat = None
    LeannSearcher = None

from .registry import BACKEND_REGISTRY, autodiscover_backends

try:
    autodiscover_backends()
except Exception:
    pass

__all__ = ["BACKEND_REGISTRY", "LeannBuilder", "LeannChat", "LeannSearcher"]
