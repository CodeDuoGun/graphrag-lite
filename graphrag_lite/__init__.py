# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: graphrag-lite: Minimal GraphRAG implementation
"""

from .core import GraphRAGLite
try:
    from .neo4j_store import Neo4jStore
    __all__ = ["GraphRAGLite", "Neo4jStore"]
except ImportError:
    __all__ = ["GraphRAGLite"]

__version__ = "0.1.2"
