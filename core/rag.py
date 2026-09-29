"""LangChain RAG 装配层：把「用哪个向量库 / 哪个 embedding / 哪个重排模型 / 有没有 Key」
这些**环境决策**集中在一处。

为什么单独一层
--------------
``core/retriever.py`` 只描述"检索应该怎么做"（硬过滤 → 混合召回 → 重排 → 拒答），
并且**全部依赖注入**，因此在没有网络、没有 API Key 的环境里也能被完整单测。
本模块负责另一件事：把环境里的凭证与模型名翻译成那些依赖。

这样拆开的好处是"能不能跑"与"跑得好不好"解耦：

======================  ==========================  ==============================
环境                     向量腿                       重排腿
======================  ==========================  ==============================
有 DASHSCOPE_API_KEY     FAISS + text-embedding-v4   DashScope gte-rerank-v2
无 Key（CI / 断网）       不启用（纯 BM25）            确定性词面重排
======================  ==========================  ==============================

两条路径都在，且离线路径是**默认安全**的那条：没有凭证时系统不会假装自己有语义能力，
而是退化成可解释的词法检索，并在界面上如实标注检索模式。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from langchain_classic.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings

from .retriever import (
    DEFAULT_FETCH_K,
    DEFAULT_MIN_SCORE,
    DEFAULT_RELEVANCE_FLOOR,
    Chunk,
    Retriever,
    load_corpus,
)

__all__ = [
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_RERANK_MODEL",
    "build_embeddings",
    "build_reranker",
    "build_retriever",
    "corpus_index_dir",
    "dashscope_api_key",
]

DEFAULT_EMBED_MODEL = "text-embedding-v4"
DEFAULT_RERANK_MODEL = "gte-rerank-v2"
DEFAULT_RERANK_TOP_N = 12


def dashscope_api_key() -> str | None:
    """DashScope 凭证。embedding 与重排都复用它，所以只需要配一次。"""
    return os.environ.get("DASHSCOPE_API_KEY") or None


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def build_embeddings(
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> Embeddings | None:
    """构建 DashScope embedding（``text-embedding-v4``）。

    没有 Key、或依赖缺失时返回 ``None`` —— 调用方据此走词法检索，
    而不是抛异常让整个应用起不来。
    """
    key = api_key or dashscope_api_key()
    if not key:
        return None
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
    except ImportError:  # pragma: no cover - 依赖缺失属于部署问题
        return None
    try:
        return DashScopeEmbeddings(
            model=model or os.environ.get("LABLENS_EMBED_MODEL", DEFAULT_EMBED_MODEL),
            dashscope_api_key=key,
            **kwargs,
        )
    except Exception:  # noqa: BLE001 - 凭证或参数非法时静默降级
        return None


# --------------------------------------------------------------------------- #
# 重排
# --------------------------------------------------------------------------- #
def build_reranker(
    model: str | None = None,
    api_key: str | None = None,
    top_n: int = DEFAULT_RERANK_TOP_N,
    **kwargs: Any,
) -> BaseDocumentCompressor | None:
    """构建 DashScope 交叉编码器重排器（``gte-rerank-v2``）。

    注意：``langchain_community`` 的 ``DashScopeRerank`` 在**不传 client** 时会把
    ``model`` 覆盖成默认的 ``gte-rerank``。这里显式传入 ``dashscope.TextReRank``
    作为 client，才能用上 v2，并且顺带绕开那次覆盖。
    """
    key = api_key or dashscope_api_key()
    if not key:
        return None
    try:
        import dashscope
        from langchain_community.document_compressors import DashScopeRerank
    except ImportError:  # pragma: no cover - 依赖缺失属于部署问题
        return None
    try:
        return DashScopeRerank(
            client=dashscope.TextReRank,
            model=model or os.environ.get("LABLENS_RERANK_MODEL", DEFAULT_RERANK_MODEL),
            dashscope_api_key=key,
            top_n=top_n,
            **kwargs,
        )
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# 索引缓存
# --------------------------------------------------------------------------- #
def corpus_index_dir(base_dir: str | Path, corpus_path: str | Path) -> Path:
    """按**语料内容的指纹**决定索引目录。

    直接复用固定目录会有陈旧索引的风险（改了语料却还在查旧向量）。
    把内容哈希编进目录名，就让"语料变了 → 目录变了 → 自动重建"成为结构性保证，
    不需要额外的失效逻辑，也不会误用过期索引。
    """
    p = Path(corpus_path)
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else "empty"
    return Path(base_dir) / h


# --------------------------------------------------------------------------- #
# 检索器
# --------------------------------------------------------------------------- #
def build_retriever(
    corpus_path: str | Path,
    *,
    chunks: list[Chunk] | None = None,
    index_dir: str | Path | None = None,
    use_rag: bool | None = None,
    embeddings: Embeddings | None = None,
    reranker: BaseDocumentCompressor | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    relevance_floor: float | None = DEFAULT_RELEVANCE_FLOOR,
    fetch_k: int = DEFAULT_FETCH_K,
    **kwargs: Any,
) -> Retriever:
    """装配生产用的检索器。

    Parameters
    ----------
    corpus_path:
        KB-3 解释语料（JSONL）。
    use_rag:
        ``None``（默认）表示"有 DashScope 凭证就启用向量检索与 API 重排"。
        显式传 ``False`` 可强制走离线词法路径（评测与 CI 用得上）。
    chunks:
        直接给语料对象，跳过读盘（测试用）。
    index_dir:
        FAISS 索引落盘根目录。``None`` 表示不缓存，每次进程内重建。
    """
    corpus = chunks if chunks is not None else load_corpus(corpus_path)

    if use_rag is None:
        use_rag = dashscope_api_key() is not None

    embed_fn: Embeddings | None = None
    rerank: BaseDocumentCompressor | None = None
    if use_rag:
        embed_fn = embeddings or build_embeddings()
        rerank = reranker or build_reranker(top_n=max(fetch_k, DEFAULT_RERANK_TOP_N))

    vectorstore_path = None
    if embed_fn is not None and index_dir is not None:
        try:
            vectorstore_path = corpus_index_dir(index_dir, corpus_path)
        except Exception:  # noqa: BLE001 - 缓存目录不可用时退化为不缓存
            vectorstore_path = None

    return Retriever(
        corpus,
        embed_fn=embed_fn,
        reranker=rerank,
        min_score=min_score,
        relevance_floor=relevance_floor if rerank is not None else None,
        fetch_k=fetch_k,
        vectorstore_path=vectorstore_path,
        **kwargs,
    )
