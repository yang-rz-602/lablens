"""输出护栏：**代码级**的合规兜底，不依赖提示词自觉。

为什么必须有这一层
------------------
《互联网诊疗监管细则（试行）》（2022）第 13 条明确"人工智能软件等不得冒用、
替代医师本人提供诊疗服务"，第 21 条明确"严禁使用人工智能等自动生成处方"。

只靠提示词写"请不要给出用药建议"是不够的——模型仍可能偶发输出，而这类内容一旦
出现在面向公众的产品里就是监管风险。所以这里做**确定性正则拦截**：

- 命中即从输出中剔除该句，并记录一条 ``GuardHit`` 供审计
- 剔除不是"静默失败"，被剔除的内容会在界面上以"已拦截"提示体现

这也回应了 FDA 对 WHOOP 的警告信（2025-07-14）里的立场：**免责声明不能替代
对输出内容本身的克制**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["GuardHit", "GUARD_RULES", "scan_text", "sanitize_text"]

# 句子切分（中英文句末标点）
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;\n])")


@dataclass
class GuardHit:
    rule: str
    category: str
    matched: str
    sentence: str


# (规则名, 类别, 正则)
GUARD_RULES: list[tuple[str, str, re.Pattern]] = [
    (
        "prescription_drug_dose",
        "处方与用药",
        re.compile(
            r"(建议|可以|应当|应该|需要|推荐)?\s*(服用|口服|注射|静滴|静注|外用|吸入|含服)"
            r"[^。；\n]{0,24}?(\d+(\.\d+)?\s*(mg|g|μg|ug|ml|mL|IU|iu|片|粒|袋|支|单位|丸))",
            re.I,
        ),
    ),
    (
        "drug_name_plus_action",
        "处方与用药",
        re.compile(
            r"(服用|口服|使用|应用|加用|改用|换用)\s*[^。；\n]{0,10}?"
            r"(阿司匹林|他汀|二甲双胍|抗生素|头孢|青霉素|阿莫西林|布洛芬|对乙酰氨基酚|"
            r"左甲状腺素|优甲乐|华法林|氯吡格雷|胰岛素|激素|泼尼松)",
            re.I,
        ),
    ),
    (
        "dose_adjust",
        "处方与用药",
        re.compile(r"(停药|停用|加量|减量|调整剂量|加大剂量|减少剂量|自行服药|自行用药)"),
    ),
    (
        "prescription_word",
        "处方与用药",
        re.compile(r"(处方|开药|用药方案|治疗方案|给药方案)"),
    ),
    (
        "definitive_diagnosis",
        "诊断断言",
        re.compile(
            r"(确诊为?|诊断为|已患|患有|得了|您患|提示您?患|可以确诊|基本可确诊|"
            r"明确诊断|即是|就是)\s*[^。；\n]{0,12}?"
            r"(癌|肿瘤|糖尿病|高血压|肝炎|肾炎|贫血|白血病|甲亢|甲减|冠心病|肝硬化|痛风)",
        ),
    ),
    (
        "certainty_overreach",
        "绝对化表述",
        re.compile(
            r"(一定|肯定|必然|100%|毫无疑问|绝对)(是|会|说明|表明|提示|有|代表|意味|属于)"
        ),
    ),
    (
        "reassurance_overreach",
        "绝对化表述",
        re.compile(r"(完全正常|没有任何问题|不用担心|无需就医|可以放心|绝对健康)"),
    ),
]


def scan_text(text: str | None) -> list[GuardHit]:
    """扫描文本，返回所有被拦截的命中项（不修改文本）。"""
    if not text:
        return []
    hits: list[GuardHit] = []
    for name, category, pattern in GUARD_RULES:
        for m in pattern.finditer(text):
            hits.append(
                GuardHit(
                    rule=name,
                    category=category,
                    matched=m.group(0),
                    sentence=_enclosing_sentence(text, m.start(), m.end()),
                )
            )
    return hits


def sanitize_text(text: str | None) -> tuple[str, list[GuardHit]]:
    """剔除命中护栏的句子。

    Returns
    -------
    (清洗后的文本, 命中列表)
    若整段都被剔除，返回空串，由上层决定如何向用户说明。
    """
    if not text:
        return "", []

    sentences = [s for s in _SENT_SPLIT.split(text) if s and s.strip()]
    kept: list[str] = []
    hits: list[GuardHit] = []

    for sent in sentences:
        sent_hits = scan_text(sent)
        if sent_hits:
            hits.extend(sent_hits)
            continue
        kept.append(sent)

    return "".join(kept).strip(), hits


def _enclosing_sentence(text: str, start: int, end: int) -> str:
    left = max(text.rfind(p, 0, start) for p in "。！？；\n") if any(
        p in text[:start] for p in "。！？；\n"
    ) else -1
    right_candidates = [text.find(p, end) for p in "。！？；\n" if text.find(p, end) != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : right + 1].strip()
