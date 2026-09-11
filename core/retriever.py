"""约束检索（constrained retrieval）与拒答机制。

为什么不是普通的向量 RAG
-------------------------
普通 RAG 是"把 query 扔进向量库取 top-k"。在医疗场景这有三个致命问题：

1. **可能捞错项目**。查"肌酐"却召回"尿素"的区间或意义，模型会照着错的依据
   写出一段自洽的解释，用户完全看不出来。
2. **无法表达优先级**。参考区间必须"报告自带 > 知识库"，纯相似度排序表达不了
   这种业务规则。
3. **没有"不知道"这个选项**。top-k 永远会返回 k 条，哪怕全都无关。

本模块的做法是**先用确定性元数据过滤，再做语义排序**：

    规则引擎给出 canonical_name（这是查表得到的，不是模型猜的）
        ↓  硬过滤：只保留该指标的 chunk
        ↓  维度加权：偏高就优先取 high_meaning，偏低就取 low_meaning
        ↓  语义/词法打分排序
        ↓  得分低于阈值 → **拒答**，返回"暂无可靠依据"而不是硬编一段话

这个"过滤在前、检索在后"的顺序是本项目与普通医疗 RAG demo 的核心差别，
也是能写进简历和面试讲解的技术点。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Chunk",
    "RetrievalHit",
    "RetrievalResult",
    "Retriever",
    "load_corpus",
    "DIMENSION_LABELS",
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

# 维度权重：位于优先级越靠前的维度权重越高。
# 词法/语义相关度在这里只做**微调**而不是主导，原因见 retrieve() 中的说明。
_DIM_WEIGHTS = (1.0, 0.8, 0.6, 0.4)
_DIM_WEIGHT_OTHER = 0.15
_LEXICAL_TIEBREAK = 0.3

DEFAULT_MIN_SCORE = 0.20


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
# 分词与打分
# --------------------------------------------------------------------------- #
_PUNCT = re.compile(r"[\s，。、；：（）()\[\]【】,.;:!?！？\"'“”‘’\-—_/\\%]+")


def tokenize(text: str) -> list[str]:
    """中文无需外部分词器的降级方案：汉字二元组 + 英文/数字词 + 单字。

    二元组已能较好地区分"肌酐"与"尿素"这类医学术语，且零依赖。
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


class _BM25:
    """极简 BM25，够用于几百到几万条语料，省掉一个向量库依赖。"""

    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.tf = [Counter(d) for d in self.docs]
        self.len = [len(d) for d in self.docs]
        self.avg_len = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs) or 1
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def score(self, query: str, idx: int) -> float:
        q = tokenize(query)
        if not q:
            return 0.0
        tf, dl, total = self.tf[idx], self.len[idx] or 1, 0.0
        for term in q:
            if term not in tf:
                continue
            f = tf[term]
            total += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / (self.avg_len or 1))
            )
        return total

    def max_score(self, query: str) -> float:
        return max((self.score(query, i) for i in range(len(self.docs))), default=0.0)


# --------------------------------------------------------------------------- #
# 检索器
# --------------------------------------------------------------------------- #
EmbedFn = Callable[[list[str]], list[list[float]]]


class Retriever:
    """约束检索器。

    Parameters
    ----------
    chunks:
        语料分块。
    embed_fn:
        可选的 embedding 函数（``list[str] -> list[list[float]]``）。提供时启用
        混合检索（BM25 + 向量），否则纯词法检索。**没有它系统依然完整可用**——
        这是刻意的设计，避免把产品可用性绑在一个外部服务上。
    min_score:
        拒答阈值。低于它宁可说"暂无依据"。
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        embed_fn: EmbedFn | None = None,
        min_score: float = DEFAULT_MIN_SCORE,
        alpha: float = 0.5,
    ) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.min_score = min_score
        self.alpha = alpha  # 混合检索中向量分的权重
        self.embed_fn = embed_fn
        self._bm25 = _BM25([self._searchable_text(c) for c in self.chunks]) if self.chunks else None
        self._by_name: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self._by_name.setdefault(c.canonical_name, []).append(i)
        self._embeddings: list[list[float]] | None = None
        if self.embed_fn is not None and self.chunks:
            try:
                self._embeddings = self.embed_fn([self._searchable_text(c) for c in self.chunks])
            except Exception:  # noqa: BLE001 - embedding 不可用时自动降级为词法检索
                self._embeddings = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _searchable_text(c: Chunk) -> str:
        # canonical_name 一并入索引：即使某条语料的正文没写出指标英文名，
        # 也能被"按指标名硬过滤"这一步稳定命中，避免检索结果随机为空。
        return f"{c.canonical_name} {c.text} {' '.join(c.tags)} {DIMENSION_LABELS.get(c.dimension, '')}"

    @property
    def mode(self) -> str:
        return "hybrid" if self._embeddings is not None else "lexical"

    def stats(self) -> dict:
        names = sorted(self._by_name)
        return {
            "chunks": len(self.chunks),
            "indicators": len(names),
            "mode": self.mode,
            "min_score": self.min_score,
            "dimensions": sorted({c.dimension for c in self.chunks}),
        }

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

        idxs = self._by_name.get(canonical_name)
        if not idxs:
            return RetrievalResult(
                canonical_name,
                abstained=True,
                reason=f"知识库中没有「{canonical_name}」的解释依据。",
            )

        priority = DIMENSION_PRIORITY.get(flag, DIMENSION_PRIORITY["?"])
        q = query or canonical_name

        raw_scores = self._score_all(q, idxs)
        if not raw_scores:
            return RetrievalResult(
                canonical_name,
                abstained=True,
                reason="检索未产生有效得分。",
            )

        best = max(raw_scores.values()) or 0.0
        hits: list[RetrievalHit] = []
        for i, s in raw_scores.items():
            norm = (s / best) if best > 0 else 0.0
            dim = self.chunks[i].dimension
            # 维度优先，词法/语义相关度只做微调。
            #
            # 理由：候选集已经按指标名**硬过滤**过，组内每一条讲的都是同一个指标，
            # 此时"取哪一类依据"（定义 / 升高意义 / 影响因素）比"哪一条字面更像"
            # 重要得多。若让 BM25 主导，就会出现"查升高意义却返回定义"这类
            # 看起来有依据、实际答非所问的结果。
            if dim in priority:
                weight = _DIM_WEIGHTS[min(priority.index(dim), len(_DIM_WEIGHTS) - 1)]
            else:
                weight = _DIM_WEIGHT_OTHER
            final = weight + _LEXICAL_TIEBREAK * norm
            hits.append(RetrievalHit(self.chunks[i], round(final, 4), self.mode))

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
            return RetrievalResult(
                canonical_name,
                abstained=True,
                reason=f"检索到的最相关依据得分 {max((h.score for h in hits), default=0):.3f} 低于阈值 {threshold:.2f}，"
                       "为避免无依据推测，本次不提供解释。",
            )

        return RetrievalResult(canonical_name, hits=picked)

    # ------------------------------------------------------------------ #
    def _score_all(self, query: str, idxs: Sequence[int]) -> dict[int, float]:
        """对候选集合打分，返回 {索引: 得分}。"""
        scores: dict[int, float] = {}
        bm25_best = 0.0
        for i in idxs:
            s = self._bm25.score(query, i) if self._bm25 else 0.0
            scores[i] = s
            bm25_best = max(bm25_best, s)

        if self._embeddings is None or self.embed_fn is None:
            # 词法分归一化：把"指标名本身"当成基础相关度，避免同指标内所有条目同分
            if bm25_best <= 0:
                return {i: 1.0 / (1 + k) for k, i in enumerate(idxs)}
            return scores

        try:
            qvec = self.embed_fn([query])[0]
        except Exception:  # noqa: BLE001
            return scores

        combined: dict[int, float] = {}
        for i in idxs:
            cos = _cosine(qvec, self._embeddings[i])
            bm = (scores[i] / bm25_best) if bm25_best > 0 else 0.0
            combined[i] = (1 - self.alpha) * bm + self.alpha * max(0.0, cos)
        return combined


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
