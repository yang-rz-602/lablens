"""约束检索（constrained retrieval）——基于 **LangChain** 的混合检索 + 重排 + 拒答。

架构
----
::

    canonical_name（规则引擎查表得到，不是模型生成的）
        │
        ├─ 硬过滤：把候选集限制在该指标的语料上（FAISS metadata filter + BM25 子集）
        │
        ├─ 召回：LangChain EnsembleRetriever
        │     ├─ BM25Retriever        （词法，零依赖，离线可用）
        │     └─ FAISS 向量检索        （语义，DashScope text-embedding-v4）
        │     两条腿各自只能在"本指标"的候选集里取文档 → 结构上不可能串指标
        │
        ├─ 重排：LangChain ContextualCompressionRetriever
        │     ├─ 在线：DashScopeRerank（gte-rerank-v2 交叉编码器）
        │     └─ 离线：DeterministicLexicalReranker（查询-文档词面交互，确定性）
        │
        ├─ 维度先验（业务规则，非相似度）：偏高就优先取 high_meaning
        │
        └─ 得分低于阈值 → **拒答**，返回"暂无可靠依据"而不是硬编一段话

为什么不是"把 query 扔进向量库取 top-k"
--------------------------------------
1. **可能捞错项目**。查"肌酐"却召回"尿素"的区间或意义，模型会照着错的依据
   写出一段自洽的解释，用户完全看不出来。→ 所以过滤是**硬**的，且在召回之前。
2. **表达不了优先级**。"报告区间 > 知识库区间"、"问升高意义不该返回定义"
   这类业务规则，纯相似度排序表达不了。→ 所以维度先验是独立的一层。
3. **没有"不知道"这个选项**。top-k 永远返回 k 条，哪怕全都无关。
   → 所以有拒答阈值，并且重排器提供绝对相关度时可以据此拒答。

离线可跑
--------
没有 API Key 时，向量腿与重排腿自动降级为 BM25 + 确定性重排，
整条链路依然完整可用、可单测、可评测。这是刻意的设计：
**产品的可验证性不应该绑定在"有没有额度"上。**
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

__all__ = [
    "Chunk",
    "RetrievalHit",
    "RetrievalResult",
    "Retriever",
    "DeterministicLexicalReranker",
    "load_corpus",
    "tokenize",
    "DIMENSION_LABELS",
    "DIMENSION_PRIORITY",
    "DEFAULT_MIN_SCORE",
]

DIMENSION_LABELS = {
    "definition": "指标含义",
    "high_meaning": "升高相关因素",
    "low_meaning": "降低相关因素",
    "factors": "非疾病影响因素",
    "specimen": "标本与检测注意",
    "clinical_note": "临床解读要点",
}

# 默认维度优先级：按判定结果决定最该检索哪一类依据
DIMENSION_PRIORITY: dict[str, tuple[str, ...]] = {
    "H": ("high_meaning", "factors", "definition", "clinical_note"),
    "CH": ("high_meaning", "clinical_note", "factors", "definition"),
    "L": ("low_meaning", "factors", "definition", "clinical_note"),
    "CL": ("low_meaning", "clinical_note", "factors", "definition"),
    "N": ("definition", "clinical_note", "factors"),
    "?": ("definition", "specimen", "clinical_note"),
}

# 维度先验权重：位于优先级越靠前的维度权重越高。
#
# 为什么维度先验能压过重排分：候选集已经按指标名**硬过滤**过，组内每一条讲的都是
# 同一个指标，此时"取哪一类依据"（定义 / 升高意义 / 影响因素）比"哪一条字面更像"
# 重要得多。若让相关度主导，就会出现"查升高意义却返回定义"这类看起来有依据、
# 实际答非所问的结果。
#
# 带与带之间刻意不重叠（跨度 0.19 < 带间距 0.2），于是：
#   · 跨维度 → 由维度先验决定（业务规则）
#   · 同维度 → 由重排器决定（语义相关度）
# 各司其职，不会互相污染。
_DIM_WEIGHTS = (1.0, 0.8, 0.6, 0.4)
_DIM_WEIGHT_OTHER = 0.2
_RERANK_SPAN = 0.19

DEFAULT_MIN_SCORE = 0.20
# 重排器给出绝对相关度时使用：最高相关度低于它就直接拒答
DEFAULT_RELEVANCE_FLOOR = 0.15
# 送进重排器的候选数（先多召回，再精排）
DEFAULT_FETCH_K = 12

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 语料
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    id: str
    canonical_name: str
    dimension: str
    text: str
    source: str | None = None
    year: int | None = None
    tags: list[str] = field(default_factory=list)


def load_corpus(path: str | Path) -> list[Chunk]:
    """载入 JSONL 语料。文件不存在或单行损坏都不应让服务崩溃。"""
    p = Path(path)
    if not p.is_file():
        return []
    chunks: list[Chunk] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not row.get("id") or not row.get("text"):
            continue
        chunks.append(
            Chunk(
                id=row["id"],
                canonical_name=row.get("canonical_name", ""),
                dimension=row.get("dimension", "definition"),
                text=row["text"],
                source=row.get("source"),
                year=row.get("year"),
                tags=list(row.get("tags") or []),
            )
        )
    return chunks


# --------------------------------------------------------------------------- #
# 检索结果
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float
    matched_by: str  # metadata_filter | hybrid | lexical
    relevance: float | None = None  # 重排器给出的原始相关度（可追溯）

    @property
    def citation(self) -> str:
        parts = [self.chunk.source or "来源未标注"]
        if self.chunk.year:
            parts.append(str(self.chunk.year))
        return "，".join(parts)


@dataclass
class RetrievalResult:
    """一次检索的完整结果，**包含拒答信息**。"""

    canonical_name: str
    hits: list[RetrievalHit] = field(default_factory=list)
    abstained: bool = False
    reason: str | None = None

    @property
    def texts(self) -> list[str]:
        return [h.chunk.text for h in self.hits]

    def as_context(self, max_chars: int = 1800) -> str:
        """拼成可直接注入 prompt 的上下文，每条都带编号与出处。"""
        if not self.hits:
            return ""
        lines: list[str] = []
        used = 0
        for i, h in enumerate(self.hits, 1):
            block = f"[{i}] （{DIMENSION_LABELS.get(h.chunk.dimension, h.chunk.dimension)}）{h.chunk.text}\n    出处：{h.citation}"
            if used + len(block) > max_chars:
                break
            lines.append(block)
            used += len(block)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 分词
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[\s，。、；：（）()\[\]【】,.;:!?！？\"'“”‘’\-—_/\\%]+")


def tokenize(text: str) -> list[str]:
    """中文无需外部分词器的降级方案：汉字二元组 + 英文/数字词 + 单字。

    二元组已能较好地区分"肌酐"与"尿素"这类医学术语，且零依赖。
    同时被 BM25 召回腿与确定性重排腿复用。
    """
    t = _PUNCT.sub(" ", str(text).lower())
    tokens: list[str] = []
    for part in t.split():
        if re.fullmatch(r"[a-z0-9^./]+", part):
            tokens.append(part)
            continue
        chars = [c for c in part if c.strip()]
        tokens.extend(chars)
        tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return tokens


# --------------------------------------------------------------------------- #
# 把普通可调用对象适配成 LangChain Embeddings
# --------------------------------------------------------------------------- #
class CallableEmbeddings(Embeddings):
    """把 ``list[str] -> list[list[float]]`` 的普通函数包装成 LangChain Embeddings。

    存在的意义是让调用方可以注入任意向量后端（本地模型、测试用的玩具向量、
    甚至已经算好的缓存），而不必依赖某一个具体的 Embeddings 类。
    """

    def __init__(self, fn: Callable[[list[str]], list[list[float]]]) -> None:
        self._fn = fn

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._fn(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._fn([text])[0]


# --------------------------------------------------------------------------- #
# 离线重排器
# --------------------------------------------------------------------------- #
class DeterministicLexicalReranker(BaseDocumentCompressor):
    """无外部重排服务时的确定性重排器（查询 × 文档的词面交互）。

    和向量检索的本质区别：向量检索把 query 与 doc **各自**编码后比距离，
    两者从不"见面"；交叉编码器让 query 与 doc **一起**过模型。
    这里用词面重叠近似后者——虽然弱，但同样具备"query 与 doc 交互"的性质，
    因此能纠正一部分向量检索的排序错误，且完全确定性、可单测。

    它给出的是**相对**分数（按候选集内最大值归一），所以不参与绝对相关度拒答，
    拒答交给外层阈值处理。``provides_absolute_relevance`` 显式声明了这一点。
    """

    # ClassVar 而不是普通字段：它是一份契约声明，不该变成 pydantic 的实例字段
    provides_absolute_relevance: ClassVar[bool] = False

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        q_tokens = set(tokenize(query))
        scored: list[tuple[float, int, Document]] = []
        for i, doc in enumerate(documents):
            # 用原始正文参与打分：page_content 里带着指标名前缀（为了 BM25 命中），
            # 那部分对"这段话与本问题有多相关"没有信息量。
            body = str(doc.metadata.get("raw_text") or doc.page_content)
            d_tokens = set(tokenize(body))
            if not q_tokens or not d_tokens:
                overlap = 0.0
            else:
                inter = len(q_tokens & d_tokens)
                # 用 query 侧覆盖率而不是 Jaccard：长文档不该因为"长"而被惩罚
                overlap = inter / len(q_tokens)
            scored.append((overlap, i, doc))

        best = max((s for s, _, _ in scored), default=0.0)
        out: list[Document] = []
        for score, _, doc in sorted(scored, key=lambda t: (-t[0], t[1])):
            copy = Document(doc.page_content, metadata=dict(doc.metadata))
            copy.metadata["relevance_score"] = round(score / best, 4) if best > 0 else 0.0
            copy.metadata["relevance_absolute"] = False
            out.append(copy)
        return out


# --------------------------------------------------------------------------- #
# FAISS 路径兼容
# --------------------------------------------------------------------------- #
def _faiss_safe_path(path: Path) -> str | None:
    """把索引目录转成能安全交给 FAISS C++ 层的形式。

    为什么要做这件事
    ----------------
    FAISS 内部用 ``std::string`` 收路径、再交给 ``fopen``。在 Windows 上这条路径按
    **ANSI 代码页**解释，于是**绝对路径里只要出现中文就会变成乱码**，报出
    ``could not open ... for writing: No such file or directory``——
    而目录其实就在那儿。中文用户目录（``C:\\Users\\<中文名>``）因此必踩。

    相对路径在这一步之前就已经由 Python 解析成"相对当前目录"的 ASCII 字符串，
    真正的目录定位交给内核用当前目录句柄完成，所以不受影响。

    Returns
    -------
    str | None
        可安全使用的路径字符串；``None`` 表示这个位置无论如何都不安全，
        调用方应当**放弃磁盘缓存**（内存检索不受影响），而不是冒险写坏数据。
    """
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
    except (ValueError, OSError):
        rel = path
    text = str(rel)
    return text if text.isascii() else None


# --------------------------------------------------------------------------- #
# 检索器
# --------------------------------------------------------------------------- #
class Retriever:
    """约束检索器。

    Parameters
    ----------
    chunks:
        语料分块。
    embed_fn:
        可选的向量后端：既可以是 ``list[str] -> list[list[float]]`` 的普通函数，
        也可以是 LangChain ``Embeddings`` 实例。提供时启用混合检索
        （FAISS 向量 + BM25），否则退化为纯词法检索。**没有它系统依然完整可用**。
    reranker:
        LangChain ``BaseDocumentCompressor``。``None`` 时使用内置的
        ``DeterministicLexicalReranker``（离线可用）。
    min_score:
        综合得分的拒答阈值。低于它宁可说"暂无依据"。
    relevance_floor:
        重排器提供**绝对**相关度时的拒答阈值；``None`` 表示用默认值。
        确定性重排器不提供绝对相关度，因此不触发这条。
    vectorstore_path:
        可选。给定时 FAISS 索引落盘，重启后直接复用，避免重复调用 embedding。
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        embed_fn: Callable[[list[str]], list[list[float]]] | Embeddings | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
        alpha: float = 0.5,
        reranker: BaseDocumentCompressor | None = None,
        relevance_floor: float | None = DEFAULT_RELEVANCE_FLOOR,
        fetch_k: int = DEFAULT_FETCH_K,
        vectorstore_path: str | Path | None = None,
    ) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.min_score = min_score
        self.alpha = alpha  # 混合检索中向量腿的权重
        self.embed_fn = embed_fn
        self.fetch_k = fetch_k
        self.relevance_floor = relevance_floor
        self.vectorstore_path = Path(vectorstore_path) if vectorstore_path else None

        self.reranker: BaseDocumentCompressor = reranker or DeterministicLexicalReranker()
        # 重排器是否提供**绝对**相关度。显式声明优先，其次回退到类型判断。
        declared = getattr(self.reranker, "provides_absolute_relevance", None)
        self.reranker_absolute = bool(
            declared if declared is not None
            else not isinstance(self.reranker, DeterministicLexicalReranker)
        )

        self._docs: list[Document] = [self._to_document(c) for c in self.chunks]
        self._by_id: dict[str, Chunk] = {c.id: c for c in self.chunks}
        self._by_name: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self._by_name.setdefault(c.canonical_name, []).append(i)

        self._vectorstore: FAISS | None = None
        if embed_fn is not None and self._docs:
            self._build_vectorstore()

        # 每个指标一条检索链，懒构建后缓存（82 个指标 × 个位数文档，开销可忽略）
        self._ensembles: dict[str, BaseRetriever] = {}

    # ------------------------------------------------------------------ #
    # 构建
    # ------------------------------------------------------------------ #
    def _to_document(self, c: Chunk) -> Document:
        # 指标名与维度标签一并入索引：语料正文里不一定写得出英文缩写，
        # 但"按指标名硬过滤 / 按指标名查询"这一步必须能稳定命中。
        # 原始正文放 metadata["raw_text"]，重排与引用都用它。
        searchable = f"{c.canonical_name} {DIMENSION_LABELS.get(c.dimension, '')} {c.text}"
        return Document(
            page_content=searchable,
            metadata={
                "chunk_id": c.id,
                "canonical_name": c.canonical_name,
                "dimension": c.dimension,
                "raw_text": c.text,
            },
        )

    @staticmethod
    def _as_embeddings(embed_fn: Any) -> Embeddings:
        if isinstance(embed_fn, Embeddings):
            return embed_fn
        return CallableEmbeddings(embed_fn)

    def _build_vectorstore(self) -> None:
        """构建或加载 FAISS 索引；任何失败都退回词法检索，不抛给上层。

        注意：降级是**静默的功能损失**，所以每一处失败都写 warning 日志。
        把"语义检索没生效"这件事留在日志里，比让它悄悄退化成词法检索要好排查得多。
        """
        try:
            embeddings = self._as_embeddings(self.embed_fn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding 后端不可用，退化为词法检索：%s", exc)
            return

        # 先尝试从磁盘复用，避免每次启动都重新调用 embedding 接口
        load_target = _faiss_safe_path(self.vectorstore_path) if self.vectorstore_path else None
        if load_target is not None and (self.vectorstore_path / "index.faiss").is_file():
            try:
                self._vectorstore = FAISS.load_local(
                    load_target,
                    embeddings,
                    # 索引是我们自己在 data/ 下生成的，非同源 pickle 不读取
                    allow_dangerous_deserialization=True,
                )
                return
            except Exception as exc:  # noqa: BLE001 - 缓存损坏就重建
                logger.warning("FAISS 索引缓存读取失败，改为重建：%s", exc)
                self._vectorstore = None

        try:
            self._vectorstore = FAISS.from_documents(self._docs, embeddings)
        except Exception as exc:  # noqa: BLE001 - embedding 不可用则退化为纯词法
            logger.warning("FAISS 索引构建失败，退化为词法检索：%s", exc)
            self._vectorstore = None
            return

        if self.vectorstore_path is None:
            return
        save_target = _faiss_safe_path(self.vectorstore_path)
        if save_target is None:
            logger.warning(
                "索引目录路径包含非 ASCII 字符且不在当前工作目录下，FAISS 无法写入，"
                "本次跳过磁盘缓存（内存检索不受影响）：%s",
                self.vectorstore_path,
            )
            return
        try:
            self.vectorstore_path.mkdir(parents=True, exist_ok=True)
            self._vectorstore.save_local(save_target)
        except Exception as exc:  # noqa: BLE001 - 落盘失败不影响内存检索
            logger.warning("FAISS 索引落盘失败（不影响本次检索）：%s", exc)

    def _retriever_for(self, canonical_name: str) -> BaseRetriever | None:
        """取得（并缓存）该指标的混合召回器。候选集在构造期就锁死为本指标。"""
        cached = self._ensembles.get(canonical_name)
        if cached is not None:
            return cached

        idxs = self._by_name.get(canonical_name) or []
        docs = [self._docs[i] for i in idxs]
        if not docs:
            return None

        legs: list[BaseRetriever] = []
        weights: list[float] = []

        # 词法腿：BM25Retriever 不支持 metadata filter，所以直接喂**子集**构建，
        # 这在结构上等价于"硬过滤"，而且比先召回再过滤更可靠（不会漏召回）。
        legs.append(BM25Retriever.from_documents(docs, k=self.fetch_k, preprocess_func=tokenize))
        weights.append(1.0 - self.alpha)

        # 语义腿：FAISS 原生支持 metadata filter
        if self._vectorstore is not None:
            legs.append(
                self._vectorstore.as_retriever(
                    search_kwargs={"k": self.fetch_k, "filter": {"canonical_name": canonical_name}}
                )
            )
            weights.append(self.alpha)

        retriever: BaseRetriever
        if len(legs) == 1:
            retriever = legs[0]
        else:
            # EnsembleRetriever 用 RRF（倒数排名融合）合并两条腿，不需要分数可比
            retriever = EnsembleRetriever(retrievers=legs, weights=weights)

        if self.reranker is not None:
            retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker, base_retriever=retriever
            )

        self._ensembles[canonical_name] = retriever
        return retriever

    # ------------------------------------------------------------------ #
    # 只读视图
    # ------------------------------------------------------------------ #
    @property
    def mode(self) -> str:
        return "hybrid" if self._vectorstore is not None else "lexical"

    def stats(self) -> dict:
        names = sorted(self._by_name)
        return {
            "chunks": len(self.chunks),
            "indicators": len(names),
            "mode": self.mode,
            "min_score": self.min_score,
            "dimensions": sorted({c.dimension for c in self.chunks}),
            "reranker": type(self.reranker).__name__,
            "vectorstore": "FAISS" if self._vectorstore is not None else None,
            "fetch_k": self.fetch_k,
        }

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        canonical_name: str,
        query: str | None = None,
        flag: str = "?",
        top_k: int = 4,
        min_score: float | None = None,
    ) -> RetrievalResult:
        """按指标名硬过滤后检索该指标的解释性依据。

        Parameters
        ----------
        canonical_name:
            **必须来自规则引擎的查表结果**，不能是模型自由生成的文本。
            这是"约束"二字的落点。
        flag:
            判定结果（H/L/CH/CL/N/?），决定维度优先级。
        """
        threshold = self.min_score if min_score is None else min_score
        if not self.chunks:
            return RetrievalResult(canonical_name, abstained=True, reason="知识库语料为空。")

        if not self._by_name.get(canonical_name):
            return RetrievalResult(
                canonical_name,
                abstained=True,
                reason=f"知识库中没有「{canonical_name}」的解释依据。",
            )

        retriever = self._retriever_for(canonical_name)
        if retriever is None:
            return RetrievalResult(
                canonical_name,
                abstained=True,
                reason=f"知识库中没有「{canonical_name}」的解释依据。",
            )

        q = query or canonical_name
        try:
            docs = retriever.invoke(q)
        except Exception as exc:  # noqa: BLE001 - 外部服务异常一律降级为"无依据"
            return RetrievalResult(
                canonical_name,
                abstained=True,
                reason=f"检索失败（{type(exc).__name__}），为避免无依据推测，本次不提供解释。",
            )

        if not docs:
            return RetrievalResult(
                canonical_name,
                abstained=True,
                reason="检索未产生有效结果。",
            )

        relevances = [self._relevance_of(d) for d in docs]
        max_rel = max(relevances, default=0.0)

        # 重排器若给出绝对相关度，它本身就是最诚实的一道拒答闸门：
        # "这条查询在本指标的语料里根本没有像样的依据"。
        #
        # 但 relevance_floor 是按 [0,1] 标定的，而"绝对相关度"的取值范围是**第三方
        # 服务的实现细节**（有的重排接口返回 logits，有的返回百分制）。
        # 所以这里加一道防线：max_rel > 1 说明量纲与假设不符，
        # 宁可**放弃这道闸门**（仍有综合得分阈值兜底），也不要按错误的量纲误拒答。
        absolute_scale = self.reranker_absolute and max_rel <= 1.0
        if self.reranker_absolute and not absolute_scale:
            logger.warning(
                "重排相关度出现 >1 的取值（max=%.3f），与 [0,1] 标定不符，"
                "本次跳过绝对相关度闸门，仅用综合得分阈值拒答。",
                max_rel,
            )
        if (
            absolute_scale
            and self.relevance_floor is not None
            and max_rel < self.relevance_floor
        ):
            return RetrievalResult(
                canonical_name,
                abstained=True,
                reason=f"最高重排相关度 {max_rel:.3f} 低于门槛 {self.relevance_floor:.2f}，"
                       "为避免无依据推测，本次不提供解释。",
            )

        priority = DIMENSION_PRIORITY.get(flag, DIMENSION_PRIORITY["?"])
        hits: list[RetrievalHit] = []
        for doc, rel in zip(docs, relevances, strict=False):
            chunk = self._by_id.get(str(doc.metadata.get("chunk_id")))
            if chunk is None:  # pragma: no cover - 索引与语料不一致时的兜底
                continue
            dim = chunk.dimension
            if dim in priority:
                weight = _DIM_WEIGHTS[min(priority.index(dim), len(_DIM_WEIGHTS) - 1)]
            else:
                weight = _DIM_WEIGHT_OTHER
            # 维度先验决定"落在哪个带"，重排分决定"带内排序"
            norm = (rel / max_rel) if max_rel > 0 else 0.0
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    score=round(weight + _RERANK_SPAN * norm, 4),
                    matched_by=self.mode,
                    relevance=rel,
                )
            )

        hits.sort(key=lambda h: (-h.score, h.chunk.dimension))

        # 去重：同一 dimension 最多保留 2 条，避免上下文被同类内容占满
        picked: list[RetrievalHit] = []
        per_dim: Counter = Counter()
        for h in hits:
            if per_dim[h.chunk.dimension] >= 2:
                continue
            if h.score < threshold and picked:
                continue
            picked.append(h)
            per_dim[h.chunk.dimension] += 1
            if len(picked) >= top_k:
                break

        if not picked or max(h.score for h in picked) < threshold:
            best = max((h.score for h in hits), default=0.0)
            return RetrievalResult(
                canonical_name,
                abstained=True,
                reason=f"检索到的最相关依据得分 {best:.3f} 低于阈值 {threshold:.2f}，"
                       "为避免无依据推测，本次不提供解释。",
            )

        return RetrievalResult(canonical_name, hits=picked)

    @staticmethod
    def _relevance_of(doc: Document) -> float:
        try:
            return float(doc.metadata.get("relevance_score") or 0.0)
        except (TypeError, ValueError):
            return 0.0
