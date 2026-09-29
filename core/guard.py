"""输出护栏：**代码级**的合规兜底，不依赖提示词自觉。

为什么必须有这一层
-----------------
《互联网诊疗监管细则（试行）》（2022）第 13 条明确"人工智能软件等不得冒用、
替代医师本人提供诊疗服务"，第 21 条明确"严禁使用人工智能等自动生成处方"。

只靠提示词写"请不要给出用药建议"是不够的——模型仍可能偶发输出，而这类内容一旦
出现在面向公众的产品里就是监管风险。所以这里做**确定性正则拦截**：

- 命中即从输出中剔除该句，并记录一条 ``GuardHit`` 供审计
- 剔除不是"静默失败"，被剔除的内容会在界面上以"已拦截"提示体现

这也回应了 FDA 对 WHOOP 的警告信（2025-07-14）里的立场：**免责声明不能替代
对输出内容本身的克制**。

误伤也是缺陷
------------
护栏的方向是"宁枉勿纵"，但**误删合法医学说明同样是缺陷**：知识库里"哪些药会
干扰这个检验项目"的说明是必须保留的科普内容，不是用药建议。所以药名类规则带一条
**语境豁免**（见 ``_descriptive_not_prescriptive``）——只在句子确实带指导语气时拦截。
剂量、停药、处方这三类信号更硬，不豁免。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

__all__ = ["GuardHit", "GUARD_RULES", "scan_text", "sanitize_text"]

# 句子切分（中英文句末标点）
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;\n])")


@dataclass
class GuardHit:
    rule: str
    category: str
    matched: str
    sentence: str


# --------------------------------------------------------------------------- #
# 语境判据：区分"解释成因"与"指导用药"
# --------------------------------------------------------------------------- #
#: 祈使 / 指导语气线索：句子在**告诉用户该怎么做**。
#:
#: 刻意收窄：只保留真正"对读者下指令"的信号。
#: 反例说明——``需要``（"需要说明的是…"）、``不宜``（"不宜单独据此判断"，属克制表述）
#: 都是检验医学文本里的**话语标记**，不是用药指导，放进来会把大量合法说明误判成建议。
#: ``您/你`` 是最强信号：面向第二人称说话基本等于在给建议。
_PRESCRIPTIVE_CUE = re.compile(r"(建议|推荐|请|您|你|务必|最好|应当|应该|可以|遵医嘱)")
#: 描述性语境线索：句子在**解释这是怎么发生的、哪些因素会影响**。
#:
#: 覆盖检验医学文本的实际写法：``非疾病因素``、``可使…升高``、``亦可使其降低``、
#: ``属于``、``有助于``、``用于`` 等都表明这是因果说明而非指令。
_DESCRIPTIVE_CUE = re.compile(
    r"(可见于|见于|常见于|相关因素|影响因素|非疾病因素|影响|干扰|导致|引起|所致|"
    r"可使|亦可|包括|例如|属于|有助于|用于|状态|升高|降低|下降|上升)"
)


def _descriptive_not_prescriptive(sentence: str) -> bool:
    """判断句子是在**解释成因**，而不是在**指导用户用药**。

    为什么需要这条判据
    ------------------
    ``drug_name_plus_action`` 只有"动词 + 药名"两个要素，在医学表述里天然容易撞上。
    真实案例（离线拼装路径，文字逐字来自知识库语料）::

        也可见于……以及使用部分利尿剂和抗利尿激素分泌异常状态。

    这里的「抗利尿激素」是 SIADH 这个**疾病状态**的一部分，不是在给用户开药；
    整句讲的是低钠血症的病因。这类"哪些药或哪些状态会影响这个指标"的说明
    正是知识库该说的话，**不该被拦**。

    判据刻意保守：**有描述性线索、且没有祈使线索**才豁免。
    只要句子里出现"建议/需要/可以/请"这类指导语气，仍然按原强度拦截。
    """
    return bool(_DESCRIPTIVE_CUE.search(sentence)) and not _PRESCRIPTIVE_CUE.search(sentence)


class GuardRule(NamedTuple):
    name: str
    category: str
    pattern: re.Pattern[str]
    #: 返回 True 表示"这句属于描述性表述，本次豁免该规则"；None 表示不豁免
    exempt: Callable[[str], bool] | None = None


# --------------------------------------------------------------------------- #
# 规则表
# --------------------------------------------------------------------------- #
GUARD_RULES: list[GuardRule] = [
    GuardRule(
        "prescription_drug_dose",
        "处方与用药",
        re.compile(
            r"(建议|可以|应当|应该|需要|推荐)?\s*(服用|口服|注射|静滴|静注|外用|吸入|含服)"
            r"[^。；\n]{0,24}?(\d+(\.\d+)?\s*(mg|g|μg|ug|ml|mL|IU|iu|片|粒|袋|支|单位|丸))",
            re.I,
        ),
    ),
    GuardRule(
        "drug_name_plus_action",
        "处方与用药",
        # 动词表与 prescription_drug_dose 对齐：原先少了注射/静滴/静注/外用/吸入/含服，
        # 导致"静滴青霉素""吸入激素"这类无剂量的给药指令整条漏放。
        #
        # 动词与药名之间排除「的」：``注射胰岛素`` 是祈使句，而 ``注射的胰岛素制剂``
        # 是名词短语（"被注射的胰岛素"），后者是生理/制剂描述，不是给药指令。
        re.compile(
            r"(服用|口服|注射|静滴|静注|外用|吸入|含服|使用|应用|加用|改用|换用)"
            r"\s*[^。；\n的]{0,10}?"
            r"(阿司匹林|他汀|二甲双胍|抗生素|头孢|青霉素|阿莫西林|布洛芬|对乙酰氨基酚|"
            r"左甲状腺素|优甲乐|华法林|氯吡格雷|胰岛素|激素|泼尼松)",
            re.I,
        ),
        # 药名类规则最容易误伤"药物干扰检验结果"的合法说明，所以只对它开口子。
        exempt=_descriptive_not_prescriptive,
    ),
    GuardRule(
        "dose_adjust",
        "处方与用药",
        re.compile(r"(停药|停用|加量|减量|调整剂量|加大剂量|减少剂量|自行服药|自行用药)"),
        # "也可见于误服或自行加量的情形"是在讲**病因**（患者行为导致结果异常），
        # 不是让用户去加量；带建议语气的"建议自行加量"仍会被拦住。
        exempt=_descriptive_not_prescriptive,
    ),
    GuardRule(
        "prescription_word",
        "处方与用药",
        re.compile(r"(处方|开药|用药方案|治疗方案|给药方案)"),
        # "治疗方案调整"出现在罗列临床用途的句子里是合法说明；
        # 真正的建议（"以下是您的用药方案""给您开个处方"）带第二人称，仍会被拦住。
        exempt=_descriptive_not_prescriptive,
    ),
    GuardRule(
        "definitive_diagnosis",
        "诊断断言",
        re.compile(
            r"(确诊为?|诊断为|已患|患有|得了|您患|提示您?患|可以确诊|基本可确诊|"
            r"明确诊断|即是|就是)\s*[^。；\n]{0,12}?"
            r"(癌|肿瘤|糖尿病|高血压|肝炎|肾炎|贫血|白血病|甲亢|甲减|冠心病|肝硬化|痛风)"
        ),
    ),
    GuardRule(
        "certainty_overreach",
        "绝对化表述",
        # 前面加否定前瞻：``不一定代表`` ``未必说明`` ``并非绝对`` 这类**恰恰是克制表述**，
        # 不加这道防线会把"极高值不一定代表保护作用增强"误判成绝对化断言。
        re.compile(
            r"(?<![不未非])(一定|肯定|必然|100%|毫无疑问|绝对)(是|会|说明|表明|提示|有|代表|意味|属于)"
        ),
    ),
    GuardRule(
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
    for rule in GUARD_RULES:
        for m in rule.pattern.finditer(text):
            sentence = _enclosing_sentence(text, m.start(), m.end())
            if rule.exempt is not None and rule.exempt(sentence):
                continue
            hits.append(
                GuardHit(
                    rule=rule.name,
                    category=rule.category,
                    matched=m.group(0),
                    sentence=sentence,
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
