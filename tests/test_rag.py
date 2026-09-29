"""单元测试：LangChain RAG 装配层（FAISS 向量库 + 混合召回 + 重排 + 拒答）。

这些用例全部**离线可跑**：需要向量能力时注入 ``DeterministicFakeEmbedding``
（LangChain 自带的确定性假向量），需要重排时用自建的打分压缩器。
所以 CI 不需要任何 API Key，也不依赖网络。

真正需要外部服务的路径（DashScope embedding / gte-rerank）只验证到"对象被正确构造"
这一层——它的行为属于第三方，不该由本仓库的测试来断言。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_classic.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from core.rag import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_MODEL,
    build_embeddings,
    build_reranker,
    build_retriever,
    corpus_index_dir,
    dashscope_api_key,
)
from core.retriever import DeterministicLexicalReranker, Retriever, _faiss_safe_path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS = DATA_DIR / "corpus.jsonl"


@pytest.fixture()
def no_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("LABLENS_EMBED_MODEL", raising=False)
    monkeypatch.delenv("LABLENS_RERANK_MODEL", raising=False)
    return monkeypatch


@pytest.fixture()
def small_corpus(local_tmp) -> Path:
    """一份小语料 + 落盘路径，用于验证索引缓存与过滤。"""
    p = local_tmp / "corpus.jsonl"
    rows = [
        {"id": "wbc__definition__01", "canonical_name": "wbc", "dimension": "definition",
         "text": "白细胞计数是血液中白细胞的总数，参与免疫防御。", "source": "教材A"},
        {"id": "wbc__high_meaning__01", "canonical_name": "wbc", "dimension": "high_meaning",
         "text": "白细胞升高常见于细菌感染、应激与炎症反应。", "source": "教材A"},
        {"id": "wbc__high_meaning__02", "canonical_name": "wbc", "dimension": "high_meaning",
         "text": "白细胞升高也可能由剧烈运动与情绪激动引起。", "source": "教材B"},
        {"id": "crea__definition__01", "canonical_name": "crea", "dimension": "definition",
         "text": "肌酐是肌肉代谢产物，主要由肾小球滤过排出。", "source": "教材C"},
        {"id": "crea__high_meaning__01", "canonical_name": "crea", "dimension": "high_meaning",
         "text": "肌酐升高常见于肾功能下降与高蛋白饮食。", "source": "教材C"},
    ]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


class _FixedScoreReranker(BaseDocumentCompressor):
    """把所有候选打成同一个绝对相关度，用来检验绝对相关度拒答闸门。"""

    score: float = 0.9

    def compress_documents(
        self, documents: list[Document], query: str, callbacks: Callbacks | None = None
    ) -> list[Document]:
        return [
            Document(d.page_content, metadata={**d.metadata, "relevance_score": self.score})
            for d in documents
        ]


# --------------------------------------------------------------------------- #
# 凭证与工厂
# --------------------------------------------------------------------------- #
def test_dashscope_api_key_reads_env(no_key) -> None:
    assert dashscope_api_key() is None
    no_key.setenv("DASHSCOPE_API_KEY", "sk-test")
    assert dashscope_api_key() == "sk-test"


def test_build_embeddings_without_key_is_none(no_key) -> None:
    assert build_embeddings() is None


def test_build_reranker_without_key_is_none(no_key) -> None:
    assert build_reranker() is None


def test_build_embeddings_builds_dashscope_object(no_key) -> None:
    emb = build_embeddings(api_key="sk-fake")
    assert isinstance(emb, Embeddings)
    assert emb.model == DEFAULT_EMBED_MODEL


def test_build_embeddings_honours_model_override(no_key) -> None:
    assert build_embeddings(api_key="sk-fake", model="text-embedding-v3").model == "text-embedding-v3"


def test_build_embeddings_reads_model_from_env(no_key) -> None:
    no_key.setenv("LABLENS_EMBED_MODEL", "text-embedding-v3")
    assert build_embeddings(api_key="sk-fake").model == "text-embedding-v3"


def test_build_reranker_builds_compressor_object(no_key) -> None:
    r = build_reranker(api_key="sk-fake")
    assert isinstance(r, BaseDocumentCompressor)
    # 必须显式传 client，否则 langchain_community 会把 model 覆盖成默认 gte-rerank
    assert r.model == DEFAULT_RERANK_MODEL


def test_build_reranker_honours_model_override(no_key) -> None:
    assert build_reranker(api_key="sk-fake", model="qwen3-rerank").model == "qwen3-rerank"


# --------------------------------------------------------------------------- #
# 索引缓存
# --------------------------------------------------------------------------- #
def test_corpus_index_dir_is_stable_for_same_content(small_corpus, local_tmp) -> None:
    assert corpus_index_dir(local_tmp, small_corpus) == corpus_index_dir(local_tmp, small_corpus)


def test_corpus_index_dir_changes_when_corpus_changes(small_corpus, local_tmp) -> None:
    """改了语料就必须换目录，否则会静默复用旧向量索引。"""
    before = corpus_index_dir(local_tmp, small_corpus)
    small_corpus.write_text(
        small_corpus.read_text(encoding="utf-8") + "\n"
        + json.dumps({"id": "hgb__definition__01", "canonical_name": "hgb",
                      "dimension": "definition", "text": "血红蛋白负责运输氧气。"}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert corpus_index_dir(local_tmp, small_corpus) != before


def test_corpus_index_dir_missing_file_is_safe(local_tmp) -> None:
    assert corpus_index_dir(local_tmp, local_tmp / "nope.jsonl").name == "empty"


# --------------------------------------------------------------------------- #
# 装配：离线路径
# --------------------------------------------------------------------------- #
def test_offline_retriever_is_lexical_with_deterministic_reranker(small_corpus) -> None:
    r = build_retriever(small_corpus, use_rag=False)
    s = r.stats()
    assert s["mode"] == "lexical"
    assert s["vectorstore"] is None
    assert s["reranker"] == "DeterministicLexicalReranker"


def test_offline_retriever_still_answers(small_corpus) -> None:
    r = build_retriever(small_corpus, use_rag=False)
    res = r.retrieve("wbc", flag="H")
    assert not res.abstained
    assert res.hits[0].chunk.dimension == "high_meaning"


def test_use_rag_defaults_off_without_key(small_corpus, no_key) -> None:
    """没有 DashScope 凭证时，默认不启用向量检索——这是刻意的安全默认值。"""
    assert build_retriever(small_corpus).stats()["mode"] == "lexical"


# --------------------------------------------------------------------------- #
# 装配：向量路径（注入假向量，离线）
# --------------------------------------------------------------------------- #
def test_retriever_with_embeddings_is_hybrid_with_faiss(small_corpus) -> None:
    r = build_retriever(small_corpus, use_rag=True, embeddings=DeterministicFakeEmbedding(size=32))
    s = r.stats()
    assert s["mode"] == "hybrid"
    assert s["vectorstore"] == "FAISS"


def test_vector_leg_never_crosses_indicators_on_real_corpus(no_key) -> None:
    """最危险的一类静默错误：查 A 指标却召回 B 指标的语料。

    这条性质必须由**向量腿的 metadata filter** 保证，而不是靠排序碰巧对。
    这里用真实语料 + 假向量，把每个指标的检索结果都验一遍。
    """
    r = build_retriever(CORPUS, use_rag=True, embeddings=DeterministicFakeEmbedding(size=32))
    assert r.stats()["mode"] == "hybrid"

    names = sorted({c.canonical_name for c in r.chunks})
    assert len(names) > 50, "语料规模异常，用例失去意义"

    for name in names:
        res = r.retrieve(name, flag="H", top_k=6)
        assert all(h.chunk.canonical_name == name for h in res.hits), f"{name} 的检索结果串到了别的指标"


def test_ensemble_combines_both_legs(small_corpus) -> None:
    """两条腿都在时结果不应为空，且仍被硬过滤限制在本指标内。"""
    r = build_retriever(small_corpus, use_rag=True, embeddings=DeterministicFakeEmbedding(size=32))
    res = r.retrieve("wbc", query="细菌感染", flag="H", top_k=4)
    assert not res.abstained
    assert {h.chunk.canonical_name for h in res.hits} == {"wbc"}
    assert all(h.matched_by == "hybrid" for h in res.hits)


def test_faiss_index_is_cached_and_reloaded(small_corpus, local_tmp) -> None:
    """第一次构建落盘，第二次直接加载——避免每次启动重复调用 embedding。

    这条用例同时是一条**回归测试**：FAISS 的 C++ 层用 ANSI ``fopen`` 打开路径，
    在 Windows 上遇到含中文的绝对路径会报 "No such file or directory"。
    本项目的根目录（``C:\\Users\\<中文名>\\Desktop\\简历\\lablens``）正好命中这个坑，
    所以索引落盘必须把相对路径交给 FAISS（见 ``_faiss_safe_path``）。
    """
    index_root = local_tmp / "idx"
    first = build_retriever(
        small_corpus, use_rag=True, embeddings=DeterministicFakeEmbedding(size=32), index_dir=index_root
    )
    assert first.stats()["mode"] == "hybrid"
    index_files = list(index_root.rglob("index.faiss"))
    assert index_files, "FAISS 索引没有落盘"

    # 第二次构建应当直接复用磁盘索引，且检索能力不变
    second = build_retriever(
        small_corpus, use_rag=True, embeddings=DeterministicFakeEmbedding(size=32), index_dir=index_root
    )
    assert second.stats()["mode"] == "hybrid"
    assert not second.retrieve("wbc", flag="H").abstained


def test_faiss_safe_path_makes_workspace_paths_usable() -> None:
    """回归：交给 FAISS 的路径必须是 ASCII 相对路径，否则中文根目录下写不了索引。"""
    target = (Path.cwd() / ".index" / "deadbeef").resolve()
    safe = _faiss_safe_path(target)
    assert safe is not None
    assert safe.isascii(), "含中文的绝对路径会让 FAISS 的 fopen 打开失败"
    assert not Path(safe).is_absolute()


def test_faiss_safe_path_refuses_unsafe_location() -> None:
    """既不在工作目录下、又含非 ASCII 的位置必须放弃缓存，而不是写入坏数据。"""
    assert _faiss_safe_path(Path("C:/其他位置/index")) is None


def test_broken_embeddings_degrade_to_lexical(small_corpus) -> None:
    class Broken(Embeddings):
        def embed_documents(self, texts):
            raise RuntimeError("embedding 服务不可用")

        def embed_query(self, text):
            raise RuntimeError("embedding 服务不可用")

    r = build_retriever(small_corpus, use_rag=True, embeddings=Broken())
    assert r.stats()["mode"] == "lexical"
    assert not r.retrieve("wbc", flag="H").abstained


# --------------------------------------------------------------------------- #
# 重排行为
# --------------------------------------------------------------------------- #
def test_reranker_decides_order_within_same_dimension(small_corpus) -> None:
    """两条候选同属 high_meaning（同一个先验带），排序必须由重排分决定。"""
    r = build_retriever(small_corpus, use_rag=False)
    res = r.retrieve("wbc", query="细菌感染", flag="H", top_k=2)
    assert not res.abstained
    assert res.hits[0].chunk.dimension == "high_meaning"
    assert "细菌感染" in res.hits[0].chunk.text, "重排没有把与查询最相关的那条排在前面"
    assert res.hits[0].relevance is not None


def test_dimension_prior_outranks_rerank_score(small_corpus) -> None:
    """业务规则不可被相似度推翻：查"升高意义"时 high_meaning 必须压过 definition。"""
    r = build_retriever(small_corpus, use_rag=False)
    res = r.retrieve("wbc", query="白细胞计数是什么", flag="H", top_k=4)
    assert res.hits[0].chunk.dimension == "high_meaning"


def test_absolute_relevance_below_floor_abstains(small_corpus) -> None:
    """重排器给出绝对相关度且全都极低时必须拒答，而不是硬编一段解释。"""
    chunks = build_retriever(small_corpus, use_rag=False).chunks
    r = Retriever(chunks, reranker=_FixedScoreReranker(score=0.01), relevance_floor=0.5)
    res = r.retrieve("wbc", flag="H")
    assert res.abstained
    assert res.hits == []
    assert "相关度" in (res.reason or "")


def test_absolute_relevance_above_floor_is_kept(small_corpus) -> None:
    chunks = build_retriever(small_corpus, use_rag=False).chunks
    r = Retriever(chunks, reranker=_FixedScoreReranker(score=0.9), relevance_floor=0.5)
    assert not r.retrieve("wbc", flag="H").abstained


def test_absolute_gate_skipped_when_scale_is_not_unit_interval(small_corpus) -> None:
    """第三方重排接口的分数范围是实现细节，不一定是 [0,1]。

    遇到 >1 的量纲时必须**放弃闸门**（还有综合得分阈值兜底），
    否则会按错误的量纲把所有查询都误拒答。
    """
    chunks = build_retriever(small_corpus, use_rag=False).chunks
    # 分数是百分制，却按 [0,1] 的 floor 判断 → 若不设防就会全部拒答
    r = Retriever(chunks, reranker=_FixedScoreReranker(score=87.0), relevance_floor=0.5)
    res = r.retrieve("wbc", flag="H")
    assert not res.abstained
    assert res.hits


def test_deterministic_reranker_declares_relative_scores() -> None:
    """确定性重排器给的是相对分，不能拿去和绝对门槛比较——这个契约要显式写死。"""
    assert DeterministicLexicalReranker.provides_absolute_relevance is False
    r = Retriever([], reranker=DeterministicLexicalReranker())
    assert r.reranker_absolute is False


def test_deterministic_reranker_survives_empty_documents() -> None:
    assert DeterministicLexicalReranker().compress_documents([], "查询") == []


# --------------------------------------------------------------------------- #
# 拒答仍然有效
# --------------------------------------------------------------------------- #
def test_unknown_indicator_abstains_in_both_modes(small_corpus) -> None:
    for kwargs in ({}, {"use_rag": True, "embeddings": DeterministicFakeEmbedding(size=32)}):
        r = build_retriever(small_corpus, **kwargs)
        res = r.retrieve("interleukin_6", flag="H")
        assert res.abstained
        assert res.hits == []


def test_retrieval_returns_citations_in_both_modes(small_corpus) -> None:
    for kwargs in ({}, {"use_rag": True, "embeddings": DeterministicFakeEmbedding(size=32)}):
        r = build_retriever(small_corpus, **kwargs)
        ctx = r.retrieve("wbc", flag="H").as_context()
        assert "出处" in ctx
        assert "教材" in ctx


# --------------------------------------------------------------------------- #
# Stage 3 端到端（检索 → 重排 → LLM 解释 → 护栏）
# --------------------------------------------------------------------------- #
def test_stage3_end_to_end_with_langchain_fake_client() -> None:
    """把 LangChain 传输层、检索层、解释层与护栏串起来跑一遍。

    单测各自覆盖了每一层的内部逻辑，但"换掉 LLM 传输层之后整条链还能不能接上"
    只有集成用例能答。这里用真实知识库 + 真实语料，但走离线检索 + 假模型，
    所以不联网也不需要 Key。
    """
    from core.explain import interpret
    from core.providers import FakeLLMClient
    from core.rules import build_report, load_knowledge_base
    from core.schema import RawLabItem, RawLabReport

    kb = load_knowledge_base(DATA_DIR)
    report = build_report(
        RawLabReport(
            items=[
                RawLabItem(
                    raw_name="白细胞计数", result_raw="13.2",
                    unit_raw="10^9/L", ref_raw="3.5-9.5",
                )
            ]
        ),
        kb,
    )
    assert report.items[0].canonical_name == "wbc"
    assert report.items[0].flag.value == "H"

    payload = {
        "overview": "本次共 1 项，偏高 1 项。",
        "items": [{
            "canonical_name": "wbc",
            "one_liner": "白细胞计数反映免疫防御水平。",
            "what_it_means": "本次结果高于参考区间，可能与感染等因素有关。",
            "possible_factors": ["细菌感染", "剧烈运动"],
        }],
        "questions_for_doctor": ["需要复查吗？"],
        "next_steps": ["携带原始报告就诊。"],
    }
    result = interpret(
        report,
        client=FakeLLMClient(extraction=payload),
        retriever=build_retriever(CORPUS, use_rag=False),
    )

    assert result.used_llm is True
    assert result.model_used == "fake"
    assert result.interpretation.overview == payload["overview"]
    assert result.interpretation.items[0].one_liner.startswith("白细胞计数")
    # 检索确实取到了依据，而不是拒答
    assert not result.retrieval["wbc"].abstained
    # 护栏对合规输出不应误伤
    assert result.guard_hits == []
    # 免责声明与局限性说明必须始终存在
    assert result.interpretation.disclaimer
    assert result.interpretation.limitations


def test_stage3_guard_still_blocks_on_llm_path() -> None:
    """换成 LangChain 传输层之后，护栏必须在真实调用路径上依然生效。"""
    from core.explain import interpret
    from core.providers import FakeLLMClient
    from core.rules import build_report, load_knowledge_base
    from core.schema import RawLabItem, RawLabReport

    kb = load_knowledge_base(DATA_DIR)
    report = build_report(
        RawLabReport(items=[RawLabItem(raw_name="白细胞计数", result_raw="13.2",
                                       unit_raw="10^9/L", ref_raw="3.5-9.5")]),
        kb,
    )
    payload = {
        "overview": "可以确诊为细菌感染。",
        "items": [{
            "canonical_name": "wbc",
            "one_liner": "白细胞计数。",
            "what_it_means": "建议服用阿司匹林100mg。",
            "possible_factors": [],
        }],
        "questions_for_doctor": [],
        "next_steps": [],
    }
    result = interpret(
        report,
        client=FakeLLMClient(extraction=payload),
        retriever=build_retriever(CORPUS, use_rag=False),
    )
    assert result.guard_hits, "诊断结论与用药建议必须被护栏拦下"


# --------------------------------------------------------------------------- #
# 不变式：护栏不得误删知识库原文
# --------------------------------------------------------------------------- #
def test_offline_extractive_path_never_redacts_corpus_text() -> None:
    """离线拼装路径的输出**逐字来自知识库语料**，所以护栏不该删掉任何东西。

    语料是经过校对的检验医学原文，讲的都是"哪些因素、哪些药会影响这个指标"，
    属于必须保留的科普内容。护栏在这里一旦删句子，就说明规则把合法医学说明
    误判成了用药建议。

    这条不变式专门用来抓**误删**：只断言"最终输出里没有违规内容"的用例抓不到它，
    因为句子被删掉之后，输出当然就是干净的——它分不清"删对了"和"误删了"。
    """
    from core.explain import interpret
    from core.rules import build_report, load_knowledge_base
    from core.schema import RawLabItem, RawLabReport

    cases = json.loads(
        (ROOT / "evals/synthetic/ground_truth.json").read_text(encoding="utf-8")
    )["cases"]
    kb = load_knowledge_base(DATA_DIR)
    retriever = build_retriever(CORPUS, use_rag=False)

    offenders: list[tuple[str, str, str]] = []
    for case in cases:
        patient = case.get("patient") or {}
        items = []
        for it in case["items"]:
            value = it.get("value")
            items.append(
                RawLabItem(
                    raw_name=it.get("name_zh") or it["canonical_name"],
                    result_raw=f"{value:g}" if value is not None else str(it.get("value_text") or ""),
                    unit_raw=it.get("unit"),
                    ref_raw=it.get("ref_text"),
                    page=1,
                )
            )
        report = build_report(
            RawLabReport(
                report_type=case.get("report_type"),
                patient_sex=patient.get("sex"),
                patient_age=patient.get("age"),
                items=items,
            ),
            kb,
            model_used="synthetic",
        )
        result = interpret(report, client=None, retriever=retriever)
        offenders += [
            (case["case_id"], h.rule, h.matched) for h in result.guard_hits
        ]

    assert offenders == [], f"离线路径误删了知识库原文：{offenders}"
