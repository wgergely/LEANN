"""
ASTChunk - AST-based code chunking library.

This package provides tools for intelligently chunking source code
while preserving syntactic structure and semantic boundaries.
"""

from .astchunk import ASTChunk
from .astchunk_builder import ASTChunkBuilder
from .astnode import ASTNode
from .preprocessing import (
    ByteRange,
    IntRange,
    get_largest_node_in_brange,
    get_nodes_in_brange,
    get_nws_count,
    get_nws_count_direct,
    preprocess_nws_count,
)

__version__ = "0.1.0"

__all__ = [
    "ASTChunk",
    "ASTChunkBuilder",
    "ASTNode",
    "ByteRange",
    "IntRange",
    "get_largest_node_in_brange",
    "get_nodes_in_brange",
    "get_nws_count",
    "get_nws_count_direct",
    "preprocess_nws_count",
]
