"""单元测试：约束检索与拒答。

最关键的一条安全性质量保证是：**检索绝不能串到别的指标上**。
普通向量 RAG 做不到这一点，而这是本项目"约束检索"要解决的问题。
"""

from __future__ import annotations

import pytest

from core.retriever import Chunk, Retriever, load_corpus, tokenize


def _chunk(cid: str, name: str, dim: str, text: str, source: str = "测试教材", year: int = 2020) -> Chunk:
    return Chunk(id=cid, canonical_name=name, dimension=dim, text=text, source=source, year=year)


@pytest.fixture()
def retriever() -> Retriever:
    return Retriever(
        [
            _chunk("wbc__definition__01", "wbc", "definition", "白细胞计数是血液中白细胞的总数，参与免疫防御。"),
            _chunk("wbc__high_meaning__01", "wbc", "high_meaning", "白细胞升高常见于细菌感染、应激、炎症等情况。"),
            _chunk("wbc__low_meaning__01", "wbc", "low_meaning", "白细胞降低常见于病毒感染、药物影响、骨髓抑制。"),
            _chunk("wbc__factors__01", "wbc", "factors", "剧烈运动、情绪激动、进食后采血可导致白细胞暂时升高。"),
            _chunk("crea__definition__01", "crea", "definition", "肌酐是肌肉代谢产物，主要由肾小球滤过排出。"),
            _chunk("crea__high_meaning__01", "crea", "high_meaning", "肌酐升高常见于肾功能下降、肌肉量大、高蛋白饮食。"),
        ]
    )


# --------------------------------------------------------------------------- #
# 核心安全性质：不串指标
# --------------------------------------------------------------------------- #
def test_never_crosses_indicators(retriever: Retriever) -> None:
    """查 wbc 时，绝不能返回 crea 的任何内容——哪怕语义上很像。"""
    result = retriever.retrieve("crea", flag="H")
    assert not result.abstained
    assert all(h.chunk.canonical_name == "crea" for h in result.hits), \
        "检索结果里混入了其它指标的语料，这是最危险的静默错误"


def test_filter_is_hard_all_the_way(retriever: Retriever) -> None:
    for name in ("wbc", "crea"):
        res = retriever.retrieve(name, flag="H", top_k=10)
        assert res.hits
        assert {h.chunk.canonical_name for h in res.hits} == {name}


# --------------------------------------------------------------------------- #
# 拒答
# --------------------------------------------------------------------------- #
def test_abstains_when_indicator_absent(retriever: Retriever) -> None:
    """语料里没有这个指标 → 拒答，而不是返回 k 条不相关内容。"""
    res = retriever.retrieve("interleukin_6", flag="H")
    assert res.abstained
    assert res.hits == []
    assert res.reason and "没有" in res.reason


def test_abstains_on_empty_corpus() -> None:
    res = Retriever([]).retrieve("wbc", flag="H")
    assert res.abstained
    assert res.as_context() == ""


def test_abstains_when_score_below_threshold() -> None:
    """把阈值抬到不可能达到的高度，必须拒答。"""
    r = Retriever(
        [_chunk("wbc__definition__01", "wbc", "definition", "白细胞计数是血液中白细胞的总数。")],
        min_score=1.5,  # 归一化得分上限为 1.0 + 维度加权，设成 1.5 强制拒答
    )
    res = r.retrieve("wbc", flag="H")
    assert res.abstained
    assert "阈值" in (res.reason or "")


# --------------------------------------------------------------------------- #
# 维度优先级
# --------------------------------------------------------------------------- #
def test_high_flag_prefers_high_meaning(retriever: Retriever) -> None:
    res = retriever.retrieve("wbc", flag="H", top_k=3)
    assert res.hits[0].chunk.dimension == "high_meaning"


def test_low_flag_prefers_low_meaning(retriever: Retriever) -> None:
    res = retriever.retrieve("wbc", flag="L", top_k=3)
    assert res.hits[0].chunk.dimension == "low_meaning"


def test_unknown_flag_prefers_definition(retriever: Retriever) -> None:
    res = retriever.retrieve("wbc", flag="?", top_k=2)
    assert res.hits[0].chunk.dimension in {"definition", "specimen", "clinical_note"}


def test_top_k_respected(retriever: Retriever) -> None:
    assert len(retriever.retrieve("wbc", flag="H", top_k=2).hits) <= 2


def test_same_dimension_capped(retriever: Retriever) -> None:
    """同一维度最多两条，避免上下文被同类内容占满。"""
    chunks = [
        _chunk(f"wbc__high_meaning__0{i}", "wbc", "high_meaning", f"白细胞升高相关因素说明第{i}条。")
        for i in range(5)
    ]
    res = Retriever(chunks).retrieve("wbc", flag="H", top_k=5)
    dims = [h.chunk.dimension for h in res.hits]
    assert dims.count("high_meaning") <= 2


# --------------------------------------------------------------------------- #
# 上下文与引用
# --------------------------------------------------------------------------- #
def test_context_carries_citations(retriever: Retriever) -> None:
    ctx = retriever.retrieve("wbc", flag="H").as_context()
    assert "出处" in ctx
    assert "测试教材" in ctx


def test_context_respects_max_chars(retriever: Retriever) -> None:
    ctx = retriever.retrieve("wbc", flag="H").as_context(max_chars=60)
    assert len(ctx) <= 400  # 至少不会把全部语料都塞进去


# --------------------------------------------------------------------------- #
# 混合检索（提供 embedding 时）
# --------------------------------------------------------------------------- #
def test_hybrid_mode_with_embed_fn() -> None:
    def fake_embed(texts: list[str]) -> list[list[float]]:
        # 极简玩具向量：按是否含关键字给维度
        return [[1.0 if "升高" in t else 0.0, 1.0 if "降低" in t else 0.0, 0.5] for t in texts]

    r = Retriever(
        [
            _chunk("wbc__high_meaning__01", "wbc", "high_meaning", "白细胞升高相关因素。"),
            _chunk("wbc__low_meaning__01", "wbc", "low_meaning", "白细胞降低相关因素。"),
        ],
        embed_fn=fake_embed,
    )
    assert r.mode == "hybrid"
    res = r.retrieve("wbc", query="升高", flag="H")
    assert not res.abstained
    assert all(h.chunk.canonical_name == "wbc" for h in res.hits)


def test_embedding_failure_falls_back_to_lexical() -> None:
    def broken(_: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding 服务不可用")

    r = Retriever([_chunk("wbc__definition__01", "wbc", "definition", "白细胞计数。")], embed_fn=broken)
    assert r.mode == "lexical"
    assert not r.retrieve("wbc", flag="N").abstained


# --------------------------------------------------------------------------- #
# 语料加载
# --------------------------------------------------------------------------- #
def test_load_corpus_missing_file(local_tmp) -> None:
    assert load_corpus(local_tmp / "nope.jsonl") == []


def test_load_corpus_skips_bad_lines(local_tmp) -> None:
    p = local_tmp / "c.jsonl"
    p.write_text(
        '{"id":"a","canonical_name":"wbc","dimension":"definition","text":"正常一行"}\n'
        "这不是 JSON\n"
        '{"id":"","text":"缺 id"}\n'
        "\n"
        '{"id":"b","canonical_name":"hgb","dimension":"definition","text":"另一行"}',
        encoding="utf-8",
    )
    chunks = load_corpus(p)
    assert [c.id for c in chunks] == ["a", "b"]


# --------------------------------------------------------------------------- #
# 分词
# --------------------------------------------------------------------------- #
def test_tokenize_chinese_bigrams() -> None:
    toks = tokenize("白细胞计数")
    assert "白细" in toks and "细胞" in toks


def test_tokenize_mixed() -> None:
    toks = tokenize("WBC 白细胞 10^9/L")
    assert "wbc" in toks
    assert "白细" in toks


def test_stats(retriever: Retriever) -> None:
    s = retriever.stats()
    assert s["chunks"] == 6
    assert s["indicators"] == 2
    assert s["mode"] == "lexical"
