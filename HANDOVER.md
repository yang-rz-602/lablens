# LabLens 交接文档

> 最后更新：2026-09-11　｜　状态：**已上线、已交付、两个远端均已同步**

---

## 1. 30 秒速览

**LabLens 是一个中文检验报告解读智能体。** 它把检验报告上的数值与**报告自己印的参考区间**逐项对照，
判定偏高 / 偏低 / 危急值，再把结果翻译成普通人能读懂的话。

它和市面上十几个同类 Demo 的区别只有一句话：

> **数值判定 100% 交给确定性代码，语言模型只负责把结果翻译成人话；没有参考区间时，系统拒绝判断。**

这不是洁癖，是这个项目存在的全部理由——见第 6 节。

| | |
|---|---|
| **线上地址** | <https://modelscope.cn/studios/yangrz2222/lablens>（公开、国内直连、免费） |
| **直连地址** | <https://yangrz2222-lablens.ms.show> |
| **代码仓库** | <https://github.com/yang-rz-602/lablens> |
| **技术栈** | Python 3.10+ · **LangChain 1.x** · **FAISS** · DashScope（`text-embedding-v4` + `gte-rerank-v2`）· Pydantic v2 · Streamlit |
| **代码规模** | 7,076 行 Python（core 3055 / tests 1717 / root 956 / ui 622 / evals 468 / scripts 258） |
| **测试** | **292 个，全绿**（含 20 个端到端界面测试、61 个 RAG/LLM 层测试） |
| **评测** | 生产路径准确率 **100.0%**，危急值漏报 **0** 项 |
| **知识库** | 95 项指标字典 + 574 条解释语料 |

> **检索与 LLM 层已重构为 LangChain**（2026-09 之后）：
> `EnsembleRetriever`（BM25 ∥ FAISS）+ `ContextualCompressionRetriever`（gte-rerank-v2 重排），
> LLM 传输层从手写 HTTP 换成 LangChain `ChatOpenAI`。
> **没有 API Key 时自动降级为 BM25 + 确定性重排**，CI 全程跑在降级路径上，不需要任何凭证。

---

## 2. 当前状态（事实）

### 2.1 代码与部署

```
分支            main
未提交改动      无
远端 origin     https://github.com/yang-rz-602/lablens.git
远端 modelscope https://modelscope.cn/studios/yangrz2222/lablens.git
同步状态        两个远端均已同步（git push origin main / git push modelscope HEAD:master）
```

**创空间状态**（查 OpenAPI 实测）：

```
id          = yangrz2222/lablens
status      = Running
visibility  = public
license     = mit
sdk_type    = streamlit
hardware    = platform/2v-cpu-16g-mem      ← 免费规格，无付费选项
base_image  = ubuntu22.04-py311-torch2.9.1-modelscope1.35.0
host        = https://yangrz2222-lablens.ms.show
```

### 2.2 已解决 / 待办

| # | 事项 | 状态 |
|---|---|---|
| 1 | GitHub 提交同步 | ✅ **已解决**。曾因本机到 `github.com` 超时积压 3 个提交，后已全部推送（`fa6a53c..ffb5bc7`）。**注意这台机器到 GitHub 不稳定，推送失败重试几次即可** |
| 2 | **README 缺界面截图** | ⚠️ **仍未完成**。面试官在 GitHub 上只能看文字。见下方"待办" |

### 2.3 ⚠️ 关于截图：为什么一直没做上

不是没试，是**环境不允许**，三条路都被同一堵墙挡住（记录在此避免重复踩）：

1. **Playwright** → 它通过**命名管道**和 Node 驱动通信，沙箱禁止（`couldn't create signal pipe`）
2. **Chrome headless** → 进程间通信走 `mojo::platform_channel`，同样是命名管道
3. **申请更宽权限后 Chrome 能起来** → 但 Streamlit 靠 websocket 拉内容，
   Chrome 的一次性 `--screenshot` 等不到渲染完成，**截出来是空白页**（已实测确认）

**结论：截图必须人工做。** 打开 <https://modelscope.cn/studios/yangrz2222/lablens>，
选 `case_012`（血钾 6.8）→ 点「开始分析」→ 截 2 张：
②「指标与判读依据」页（含危急值提示与判据表）、③「解读」页。放进 `docs/images/`。

### 2.3 本地环境版本

```
Python       3.14.4   （⚠️ 创空间跑的是 3.11，新代码需兼容 3.11）
pydantic     2.13.5
PyYAML       6.0.3
streamlit    1.63.0
pandas       3.0.5
duckdb       1.5.5
pytest       9.1.1
ruff         0.16.7

# LangChain 栈（重构后新增）
langchain          1.4.3
langchain-core     1.6.5
langchain-classic  1.0.8     ← EnsembleRetriever / ContextualCompressionRetriever 在这里
langchain-community 0.4.2    ← BM25Retriever / DashScopeEmbeddings / DashScopeRerank
langchain-openai   1.6.6
faiss-cpu          1.15.1
rank-bm25          0.2.2
dashscope          1.27.7
```

> 注意 `httpx` 已不再是直接依赖：手写的 HTTP 客户端已被 LangChain 的
> `ChatOpenAI` 取代（`httpx` 现在只是 `langchain-openai` 的传递依赖）。

---

## 3. 快速上手

```bash
cd lablens

# 1) 建环境（用 uv 最快；普通 venv 也行）
uv venv .venv
# 依赖清单已含 LangChain 栈与 FAISS，直接按 pyproject 装即可
uv pip install --python .venv -e ".[app,dev]"

# 2) 跑测试（292 个，约 1 分钟）
.venv/Scripts/python -m pytest -q

# 3) 跑评测（不需要网络、不需要 API Key）
.venv/Scripts/python evals/run_eval.py

# 4) 起界面
.venv/Scripts/python -m streamlit run app.py
```

**不需要任何 API Key 就能跑通全部功能。** 没有 Key 时系统自动降级到
「离线抽取式解读」+「BM25 词法检索 + 确定性重排」——判读、检索、拒答、护栏全部照常工作，
只是解读文案由知识库原文拼装、检索没有语义能力，不如配了 Key 时流畅准确。
这是刻意的设计：**把可验证性从"有没有额度"里解耦出来**。

配置 `DASHSCOPE_API_KEY` 后会自动启用完整 RAG 链路
（FAISS 向量库 + `text-embedding-v4` + `gte-rerank-v2` 交叉编码器重排）。

### Windows 沙箱下的两个坑（如果在这个环境里开发）

```powershell
# 1) uv 的默认缓存在工作区外会被拒绝 → 指到项目内
$env:UV_CACHE_DIR="<项目路径>\.uv-cache"

# 2) pytest 的 tmp_path 在系统临时目录会被拒绝 → 指到项目内
$env:TEMP = "$PWD\.tmp"; $env:TMP = $env:TEMP
```

---

## 4. 凭据与账号（**交接重点**）

| 项 | 位置 | 说明 |
|---|---|---|
| ModelScope 账号 | 用户名 `yangrz2222` | 手机号 19555107954 注册 |
| ModelScope Access Token | 项目根目录 `.ms_token`（**已 gitignore**） | 也曾在对话中明文出现过，**建议轮换**：<https://modelscope.cn/my/myaccesstoken> |
| GitHub 账号 | `yang-rz-602` | 已通过 `gh auth login` 登录 |
| DashScope / 其他模型 Key | **未配置** | 线上刻意不配（见下） |

### 关于"线上要不要配模型 Key"

**当前选择：不配。** 理由：

- 内置 20 份合成示例，**访客不配 Key 也能完整体验**判读、溯源、危急值、护栏全部功能
- 公开创空间 + 你的 Key = **任何访客都在消耗你的额度**，被刷没有感知

如果要配（让访客直接体验 LLM 解读）：

```powershell
$env:MODELSCOPE_API_KEY = (Get-Content .ms_token -Raw).Trim()
.venv/Scripts/python scripts/deploy_modelscope.py --owner yangrz2222 --dashscope-key "sk-xxxx" --skip-deploy
```

代码会自动读取服务端环境变量（`app.py` 的 `make_client`：**界面输入 > 服务端环境变量 > 离线模式**）。

### 提交身份

Git 提交当前使用 `Runze Yang <y18195226719@gmail.com>`。
部署脚本支持用环境变量覆盖：

```powershell
$env:GIT_AUTHOR_NAME="别的名字"; $env:GIT_AUTHOR_EMAIL="other@example.com"
```

---

## 5. 架构：数据怎么流动

```
检验报告图片 / PDF / 文本
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 1  抽取层（LLM，多模态）        core/extract.py      │
│ 只抄不算：参考区间原样照抄，抄不到就 null，不许自己补区间     │
│ 输出受 JSON Schema + Pydantic 双重校验                     │
└──────────────────────────────────────────────────────────┘
        │  RawLabReport
        ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 2  规则引擎（100% 确定性，无 LLM）  core/rules.py ★  │
│  · 指标名解析（95 项字典 + 别名索引 + 冲突检测与消歧）       │
│  · 参考区间选择：报告单 > 知识库 > 拒判                     │
│  · 单位归一与换算（未登记因子则拒判，绝不猜系数）            │
│  · 定性结果判定（阴性/-/neg/(-) 记号集求交）               │
│  · H / L / 危急值 / 临界 判定，计算偏离度                   │
│  · 生成 decision_trace（用了哪条区间、来自哪里、跨过哪个界）  │
└──────────────────────────────────────────────────────────┘
        │  LabReport
        ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 3  解释层（LLM + LangChain 约束检索 + 重排 + 拒答）   │
│                                      core/explain.py      │
│  · 先按 canonical_name 硬过滤 —— 绝不跨指标检索             │
│  · EnsembleRetriever：BM25 ∥ FAISS 向量（metadata filter）│
│  · ContextualCompressionRetriever：gte-rerank-v2 交叉编码器│
│  · 再按判定结果排维度优先级（偏高→升高意义，偏低→降低意义）   │
│  · 得分低于阈值 → 返回"暂无可靠依据"，不硬编                  │
│  · 无 Key 时：BM25 + 确定性词面重排，功能完整只是弱一些       │
└──────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 4  护栏与溯源                    core/guard.py ★    │
│ 正则拦截诊断/用药/剂量 → 字段级溯源 → 免责声明与局限性说明    │
└──────────────────────────────────────────────────────────┘
```

**一个关键点**：`canonical_name` 是规则引擎**查表**得到的，不是模型生成的。
约束检索用它做硬过滤，这是"不串指标"这条安全性质的根基。

---

## 6. 三条硬约束（设计的全部价值）

### 6.1 数值判定完全去 LLM 化

**依据不是直觉，是文献：**

- **Meyer 等**（*Frontiers in AI* 2025，[DOI](https://doi.org/10.3389/frai.2025.1681979)）在 **246,842 个参考区间**、726,000 次请求上测得：大模型自报的参考区间**下限平均变异系数 26.5%**
- **Lab-AI**（[arXiv:2409.18986](https://arxiv.org/abs/2409.18986)）实测 GPT-4-turbo **无检索时参考区间检索准确率仅 42.9%**，加检索后 0.995

### 6.2 没有参考区间就拒绝判断

优先级：**报告单自带区间 > 内置知识库 > 拒绝判定**。

Meyer 那篇的结论原文：在用户未提供参考区间时，**AI 应被训练为拒绝进行实验室解读**。

性别分层指标缺少性别信息时**同样拒判**——用男性区间判女性会误报"偏低"，
用男女并集判男性会漏报真实的偏低，两种都是静默错误。

### 6.3 输出护栏是代码，不是提示词

诊断结论、用药与剂量、处方表述由**正则确定性拦截**（`core/guard.py`），
依据《互联网诊疗监管细则（试行）》第 13、21 条。

反面教材：FDA 2025-07-14 给 WHOOP 的警告信明确表示，
产品标签上的"非医疗器械、不能诊断"**不足以推翻器械认定**。
→ **免责声明不能替代对输出内容本身的克制。**

#### 误伤同样是缺陷：药名规则的语境豁免

护栏方向是"宁枉勿纵"，但**误删合法医学说明也是缺陷**——"哪些药会影响这个指标"
是知识库必须保留的科普内容，不是用药建议。

原先的 `drug_name_plus_action` 只有"动词 + 药名"两个要素，在检验医学文本里天然撞车。
用不变式测试扫全量语料后一次性暴露了 15 处误伤，例如：

| 被误删的原文（逐字来自 `corpus.jsonl`） | 命中原因 |
|---|---|
| ……以及使用部分利尿剂和**抗利尿激素**分泌异常状态。 | 「抗利尿激素」是 SIADH 疾病状态，不是被开出的药 |
| 饮酒、服用部分**抗生素**、抗癫痫药……也可使AST上升。 | 讲的是"什么会升高AST"，是病因说明 |
| 但极高值**不一定代表**保护作用增强。 | `certainty_overreach` 忽略了否定词，把克制表述判成绝对化 |
| ……长期血糖控制评估与**治疗方案**调整。 | `prescription_word` 在罗列临床用途时命中 |

修法是给**药名类**规则加一条保守的语境豁免（`_descriptive_not_prescriptive`）：
**有描述性线索、且没有祈使线索**才放行。三条要点：

1. **祈使线索刻意收窄**：`需要`（"需要说明的是…"）、`不宜`（"不宜单独据此判断"，属克制表述）
   都是话语标记而非用药指导，放进判据会把大量合法说明误判成建议；`您/你` 才是最强信号。
2. **剂量类不豁免**：`prescription_drug_dose` 的"剂量 + 单位"是最硬信号，维持原强度。
3. **补上原有漏放**：`drug_name_plus_action` 的动词表原先缺 `注射|静滴|静注|外用|吸入|含服`，
   导致"静滴青霉素""吸入激素"这类不带剂量的给药指令**整条漏放**（已用 HEAD 版本对照确认是原有缺陷）。
   补动词的同时排除了动词与药名之间的「的」——`注射胰岛素`是祈使句，
   `注射的胰岛素制剂`是名词短语，两者必须区分。

**守这条性质的是不变式而非样例**：离线拼装路径的输出逐字来自语料，因此
`test_offline_extractive_path_never_redacts_corpus_text` 断言"护栏在这条路径上不得删掉任何东西"。
只断言"最终输出里没有违规内容"的用例抓不到误删——句子被删掉之后输出当然干净。

---

## 7. 目录与文件职责

```
lablens/
├── core/                       ★ 后端
│   ├── schema.py               三层数据契约 + DecisionTrace（Pydantic 强校验）
│   ├── units.py                单位归一与解析（纯函数，写法规格化）
│   ├── rules.py                ★ 唯一的数值判定层，可穷举单测（不含任何 LLM）
│   ├── retriever.py            ★ 约束检索（LangChain：硬过滤 → 混合召回 → 重排 → 拒答）
│   ├── rag.py                  LangChain RAG 装配（embedding / FAISS / reranker 工厂）
│   ├── providers/__init__.py   LLM 传输层（LangChain ChatOpenAI + 6 个供应商预设 + 离线假模型）
│   ├── extract.py              Stage 1 抽取（含防幻觉提示词）
│   ├── explain.py              Stage 3 解释（含抽取式降级路径）
│   ├── guard.py                ★ 输出护栏（正则确定性拦截）
│   └── store.py                趋势存储（默认关闭）
│
├── ui/                         界面层
│   ├── theme.py                设计令牌 + 全局 CSS
│   └── components.py           纯函数组件（数据 → HTML），可单测
│
├── data/                       知识库
│   ├── indicators_cbc_urine.yaml   血常规 + 尿常规（28 项）
│   ├── indicators_biochem.yaml     生化 + 免疫（67 项）
│   └── corpus.jsonl                解释语料 574 条 / 82 指标（396 KB）
│
├── .index/                     FAISS 向量索引缓存（按语料指纹分目录，可随时删）
│
├── evals/                      评测
│   ├── synthetic/ground_truth.json 20 份合成报告 / 276 项
│   ├── render.py                   把 ground truth 渲染成仿真报告（文本 + 图片）
│   ├── run_eval.py                 三条路径评测 + CI 安全闸门
│   └── RESULTS.md                  自动生成的评测报告
│
├── tests/                      292 个测试
│   ├── test_rules.py           48  判定逻辑（含拒判、性别区间、别名冲突消歧）
│   ├── test_units.py           41  单位归一与解析
│   ├── test_guard.py           41  输出护栏（含误伤检查）
│   ├── test_providers.py       29  LangChain LLM 传输层（消息转换 / JSON 修复重试）
│   ├── test_rag.py             32  FAISS + 混合召回 + 重排 + 拒答 + Stage 3 集成（全离线可跑）
│   ├── test_mcp_server.py      26  MCP 协议 + 工具
│   ├── test_app.py             20  端到端界面（AppTest，无需浏览器）
│   ├── test_retriever.py       19  约束检索与拒答（对外 API 兼容性）
│   └── test_ui.py              35  界面组件纯函数
│
├── scripts/deploy_modelscope.py  一键部署
├── docs/DEPLOY_MODELSCOPE.md     部署文档
├── mcp_server.py                 MCP 封装（手写 JSON-RPC 2.0，判读工具本身不依赖 LLM）
├── app.py                        Streamlit 界面入口
├── HANDOVER.md                   ← 本文档
├── README.md                     对外门面
├── DISCLAIMER.md                 合规定位与依据
└── requirements.txt              创空间依赖清单
```

---

## 8. 关键设计决策（FAQ）

### Q：为什么用 LangChain + FAISS？医疗场景不是更适合纯词法检索吗？

**这是当前架构，也是踩过坑之后的选择。** 有一点必须先说清楚：
**向量检索在本项目里只是"召回的一条腿"，不是判定的依据**——
数值判定 100% 仍由 `core/rules.py` 的确定性代码完成，参考区间也仍然走查表，
不是语义检索出来的。换掉检索层不会动摇 §6 那三条硬约束。

当前链路（`core/retriever.py`）：

```
canonical_name（规则引擎查表得到，不是模型生成）
   → 硬过滤：候选集锁死在本指标的语料上
   → EnsembleRetriever   BM25Retriever ∥ FAISS(metadata filter)
   → ContextualCompressionRetriever   DashScope gte-rerank-v2
   → 维度先验（业务规则）→ 阈值拒答
```

为什么敢上向量检索，以及怎么防它"串指标"：

1. **过滤在召回之前，而且是硬的。** FAISS 那条腿靠 metadata `filter={"canonical_name": ...}`
   限制；BM25 那条腿直接喂**子集**构建（`BM25Retriever` 不支持 metadata filter，
   所以不靠"先召回再过滤"——那会漏召回）。两条腿都在构造期就被锁死，
   所以**结构上不可能**把肌酐的依据写给尿素。`test_rag.py` 里有一条用例
   把真实语料的 82 个指标逐个跑一遍来守这条性质。
2. **语义能力确实有用。** 词法检索对"剧烈运动会不会影响这个指标"这类
   措辞差异大的查询召回很差；交叉编码器重排在这类查询上明显更稳。
3. **不确定性被隔离在可降级的位置。** 没有 Key、embedding 挂掉、重排服务超时，
   都会自动退回 BM25 + 确定性重排，功能完整只是弱一些（`mode` 字段从
   `hybrid` 变 `lexical`，界面上如实标注）。CI 全程跑在降级路径上，不需要任何 Key。

**维度先验压过相似度，是刻意的。** 候选集已经硬过滤到同一指标，
此时"取哪一类依据"（定义 / 升高意义 / 影响因素）比"哪一句字面更像"重要得多。
所以打分成两个不重叠的带：维度先验决定**跨维度**优先级，重排分只决定**同维度内**排序
（`_DIM_WEIGHTS` + `_RERANK_SPAN=0.19`）。否则就会出现"查升高意义却返回定义"
这种看起来有依据、实际答非所问的结果——那是老版本评测抓出来的真实缺陷。

### Q：为什么索引缓存目录要按语料哈希命名？

因为 FAISS 索引是**不可增量更新**的，改了语料却复用旧索引会静默返回错误的候选。
把语料内容哈希编进目录名（`corpus_index_dir`），"语料变了 → 目录变了 → 自动重建"
就是结构性保证，不需要额外的失效逻辑。

### Q：`.index/` 里为什么会有 FAISS 的坑？

FAISS 的 C++ 层用 ANSI `fopen` 打开路径，**Windows 上含中文的绝对路径会打开失败**
（报 `could not open ... No such file or directory`，而目录其实存在）。
本仓库根目录 `C:\Users\<中文名>\Desktop\简历\lablens` 正好命中。
`_faiss_safe_path()` 的做法是把路径转成**相对当前工作目录的 ASCII 路径**再交给 FAISS，
内核用当前目录句柄解析因而不受编码影响。`test_rag.py` 有两条回归用例守着它。

### Q：为什么 `data/` 与代码分离？

方便按卫生行业标准版本替换参考区间而**不用改一行逻辑**。
每条指标都带 `source` 字段标注到具体标准（WS/T 405-2012、WS/T 404 系列等），
`note` 字段记录方法学陷阱（`ckmb` 质量法 vs 活性法不可比、`ddimer` 的 DDU/FEU 差约 2 倍等），
这些说明会随判定结果透传给用户。

### Q：别名冲突怎么处理的？

两阶段索引：**本体名（canonical_name / name_zh）优先，别名只填空位，撞名记录下来而不是悄悄覆盖**。

真实事故：`crp` 曾抢注了 `超敏C反应蛋白`，导致 `hscrp` 永远解析不到，
**系统拿着 crp 的参考区间去判 hscrp 的结果**——数字看着正常，结论是错的。

跨类别同名（`白细胞` 在血常规指 WBC，在尿常规指尿白细胞酯酶）用**报告类型收窄作用域**消歧。

### Q：`decision_trace` 是什么？

每个判定的结构化依据：`rule` / `interval_source` / `interval_standard` / `interval_used` / `crossed` / `outcome`。

回答的是审计问题"**这个结论是怎么来的**"。只给结论不给依据的系统在医疗场景不可用——
用户无法判断该不该信，也无法在结论可疑时定位问题。

### Q：趋势存储为什么默认关闭？

检验报告属《个人信息保护法》第 28 条的敏感个人信息；
GB/T 39725-2020 要求健康医疗数据"不宜存储在境外服务器"；
《促进和规范数据跨境流动规定》第 5 条的豁免**不含敏感个人信息**。

**默认选择是"不处理"**：进程退出即忘。开启后仅写本机文件，只存数值与日期，
不含姓名、医院、报告原文。创空间磁盘每次重启都会丢，线上保持关闭即可。

---

## 9. 评测：怎么跑，怎么看

```bash
python evals/run_eval.py            # 三条路径评测
python evals/run_eval.py --write    # 同时写 evals/RESULTS.md
python evals/run_eval.py --gate     # 安全闸门：危急值漏报 > 0 则非零退出（CI 用）
python evals/run_eval.py --extract  # 额外评测 LLM 抽取层（需 API Key）
```

### 当前结果

| 路径 | 样本 | 总体准确率 | **危急值漏报率** | 异常总漏报率 | 未判定 |
|---|---|---|---|---|---|
| A1 · 仅报告自带区间（空知识库） | 276 | 98.6% | 100.0%※ | 0.0% | 1 |
| A2 · 仅内置知识库 | 276 | 99.3% | **0.0%** | 0.9% | 1 |
| A3 · 生产路径 | 276 | **100.0%** | **0.0%** | **0.0%** | 0 |

※ **A1 那一列的 100% 不是缺陷，是设计使然**：该路径刻意清空知识库，
此时系统里不存在任何危急值阈值，四项危急值只能被判成普通偏高/偏低。
这一列的作用是确认"**没有参考数据时系统不会凭空捏造危急值**"。

### 为什么头条指标是「危急值漏报率」而不是平均准确率

**Ramaswamy 等**（*Nature Medicine* 2026;32:1671）：加入客观检验数据后，
**总体分诊准确率从 54.6% 升到 77.9%，但急症漏判率反而升到 56.2%**。
Zayed 等（*CCLM* 2025）测出大模型检验开单精确率 68–82%、**召回仅 41–51%**。

把危急值判成正常、和把正常判成偏高，代价完全不对称。**只报平均准确率的评测是在掩盖尾部风险。**

### 评测跑出来并已修复的真实缺陷

| 缺陷 | 后果 | 修复 |
|---|---|---|
| 别名冲突被静默吞掉 | `hscrp` 用错 crp 的区间 | 两阶段索引 + 冲突记录 |
| 危急值量纲错配 | 报告用 `%`、阈值用 `L/L` → 正常值判成**危急偏高** | 阈值只在单位兼容时套用 |
| 定性记号组合漏判 | "阴性(-)" 误判为阳性 | 括号拆分为记号集求交 |
| `mL/min/1.73m^2` vs `m2` | eGFR 被误拒判 | 单位别名补全 |
| ALT/AST 缺危急值阈值 | 685 U/L 只判成"偏高" | 补 500 U/L（标注各院阈值不同） |
| 检索维度被词法分压制 | 查"升高意义"却返回"定义" | 维度权重高于词法相关度 |

**两项刻意保留、没有"优化掉"的行为**：
`apoa1` 是知识库（WS/T 404.3-2012）与评测集的区间定义分歧；
`ckmb` 是系统**正确拒判**——报告用活性法 U/L、知识库按质量法 μg/L 存区间，
两者不可换算，宁可放弃判定也不硬套阈值。

---

## 10. 部署到魔搭创空间

完整清单见 [`docs/DEPLOY_MODELSCOPE.md`](docs/DEPLOY_MODELSCOPE.md)。要点：

```powershell
$env:MODELSCOPE_API_KEY = (Get-Content .ms_token -Raw).Trim()
python scripts/deploy_modelscope.py --owner yangrz2222 --tail-logs
```

脚本流程：验令牌 → 查可用硬件（**自动跳过付费规格**）→ 创建/复用空间 →
推送代码到 `master` → 写 secret → 触发部署 → 轮询日志。
推完在 `finally` 里无条件把 remote 还原成不含令牌的干净地址。

### 四个必须知道的坑（都是实跑撞出来的）

1. **创空间只读 `requirements.txt`，不读 `pyproject.toml`** —— 漏了就是启动时 `ModuleNotFoundError`
2. **新建的创空间里已有平台初始化内容**，直接 push 会被 `non-fast-forward` 拒绝，
   必须先 `fetch` + `merge --allow-unrelated-histories`（脚本已处理，冲突时保留本地）
3. **默认分支是 `master` 不是 `main`，且禁止 force push**
4. **创空间无法通过 API 删除**，只能在网页控制台删；程序侧只能 `stop`
5. **首次构建约 5 分钟**（Building → Deploying → Running），日志接口返回的是分页对象
   `{logs, total_count}` 而不是数组

### 为什么脚本用 httpx 而不是 PowerShell

本机 `Invoke-RestMethod` 存在 TLS 问题（连 `api.github.com` 都失败），
而 `httpx` 访问 `modelscope.cn` 完全正常。

### 为什么令牌内联到 Git URL

本机沙箱会拦掉 git 凭证管理器的命名管道通信
（报 `couldn't create signal pipe, Win32 error 5`）。URL 内联是唯一稳定可行的方式。

---

## 11. 已知问题与待办

### 阻塞中

- 无。

### 建议补上（按性价比排序）

- [ ] **README 截图**（最高性价比）—— 面试官在 GitHub 上不会去点链接跑代码。
      见 2.3 节的说明与具体步骤
- [ ] **轮换 ModelScope Token** —— 它曾在对话中明文出现过
- [ ] **补长尾语料** —— 字典里的 13 项还没有解释语料
      （`neut_abs` / `lymph_abs` / `u_wbc` / `u_rbc` / `u_pro` / `u_glu` / `u_ket` /
      `u_bil` / `u_blo` / `u_le` / `u_nit` / `u_ph` / `u_sg`）
- [ ] **真实报告验证** —— 目前评测集全部是合成数据，尚未用真实脱敏报告跑过

### 可选的增强方向

- [ ] 危急值阈值做成**按机构可配置**（现在写死在知识库里，是常见默认值）
- [ ] 儿童 / 妊娠期参考区间分档（现在只有成人 + 性别分层）
- [ ] 抽出层加入 OCR 置信度（参考 `blood-test-explainer` 的字段级评测范式）
- [ ] FHIR Observation 出口（B 端集成）
- [ ] **重排器换成可离线跑的本地模型**（BGE-reranker-base）——创空间基础镜像已带
      `torch2.9.1`，所以这条路是通的；现在是"有 Key 用 DashScope API，无 Key 用确定性词面重排"
- [ ] 用真实脱敏报告验证检索质量（现在只能验证"不串指标"这类结构性性质）

### 技术债 / 需要留意的上游变化

- ⚠️ **`langchain-community` 已被官方标记为 sunset**（安装时会打 DeprecationWarning）。
  本项目只从它取三样东西：`BM25Retriever`、`DashScopeEmbeddings`、`DashScopeRerank`。
  官方迁移方向是"独立的集成包"，但 DashScope 的 embedding / rerank **目前没有**独立包
  （`langchain-qwq` 只有 chat model），所以暂时无法迁走。
  `pyproject.toml` 里已把它钉在 `>=0.4,<0.5`（LangChain 1.x 兼容线），
  上游一旦提供独立包，只需改 `core/rag.py` 的两个工厂函数。
- ⚠️ **FAISS 在 Windows 上不支持含中文的绝对路径**（见 §8 最后一条 FAQ）。
  现在的做法是转相对路径；如果将来索引目录必须放在工作目录之外，需要改用
  `faiss.serialize_index` + Python 自己写文件来绕开 C++ 的文件 I/O。
- ⚠️ `langchain.retrievers` 在 LangChain 1.x **已不存在**，
  `EnsembleRetriever` / `ContextualCompressionRetriever` 都在 `langchain_classic.retrievers`。
  升级 LangChain 大版本时这里最容易踩，`tests/test_rag.py` 会在导入期就报出来。

### 明确不做（设计边界，别当成待办）

- ❌ 不出诊断结论
- ❌ 不给用药建议、不写剂量、不做处方相关表述（法规红线，代码强制）
- ❌ 不做综合风险评分或"红黄绿灯"分级（易被认定为辅助决策）
- ❌ 不做生活方式、饮食、运动建议

---

## 12. 常用操作速查

```bash
# 测试 / 质量
pytest -q                                   # 全部 292 个
pytest tests/test_rules.py -q               # 只跑判定逻辑
pytest tests/test_rag.py -q                 # 只跑检索 / 向量库 / 重排
pytest tests/test_providers.py -q           # 只跑 LLM 传输层
ruff check .
python evals/run_eval.py --gate             # 安全闸门

# 界面
streamlit run app.py

# MCP（stdio，供 Claude Desktop 等接入）
python mcp_server.py

# 部署
python scripts/deploy_modelscope.py --owner yangrz2222 --tail-logs
python scripts/deploy_modelscope.py --owner yangrz2222 --visibility public --skip-deploy
python scripts/deploy_modelscope.py --owner yangrz2222 --dashscope-key "sk-xxx"

# 查线上状态 / 日志
python -c "import httpx;print(httpx.get('https://modelscope.cn/openapi/v1/studios/yangrz2222/lablens',headers={'Authorization':'Bearer '+open('.ms_token').read().strip()},timeout=30).json()['data']['status'])"

# 推送
git push origin main                        # GitHub
git push modelscope HEAD:master             # 创空间（默认分支是 master）
```

---

## 13. 资料索引

### 项目内

| 文件 | 内容 |
|---|---|
| `README.md` | 对外门面：定位、架构、评测、快速开始 |
| `DISCLAIMER.md` | 合规定位与法规依据（药监局 47 号通告、互联网诊疗细则） |
| `docs/DEPLOY_MODELSCOPE.md` | 魔搭部署完整清单 |
| `evals/RESULTS.md` | 自动生成的评测报告 |

### 项目外（在上一级目录 `简历/`）

这四份是立项前做的调研，**是理解"为什么这样设计"的关键背景**：

| 文件 | 大小 | 内容 |
|---|---|---|
| `AI检验报告解读_合规与竞品调研.md` | 102 KB | 52 章节：NMPA 分类界定、互联网诊疗 AI 条款、PIPL 与数据出境、国内外竞品对照、25 条学术证据、"最小合规 5 件事" |
| `health-tech-competitive-report.md` | 101 KB | 国外竞品详表（11 家 DTC + 12 项 FDA 记录 + 35 项未核实清单） |
| `体检报告AI解读_竞品调研报告.md` | 43 KB | 国内竞品逐产品详表 |
| `lab-report-ai-opensource-research.md` | 22 KB | GitHub 开源生态（红海 / 空白 / 最值得复用的 5 个库） |

**核心结论一句话**：红海是"上传报告让 LLM 解释"（十几个 0–6★ 同质项目，国内 C 端已全面免费
且被蚂蚁阿福等覆盖）；空白是**中文检验报告的确定性判读层 + 字段级溯源 + 评测集 + MCP 封装**。
本项目做的就是这四条空白。

### 关键文献

1. Meyer A, et al. *ChatGPT and reference intervals: a comparative analysis of repeatability.* Frontiers in AI, 2025. [DOI](https://doi.org/10.3389/frai.2025.1681979)
2. Ramaswamy, et al. *Nature Medicine*, 2026;32(5):1671-1675. [DOI](https://doi.org/10.1038/s41591-026-04297-7)
3. Girton MR, et al. *Clinical Chemistry*, 2024;70(9):1122-1139. [DOI](https://doi.org/10.1093/clinchem/hvae093)
4. 国家药监局.《人工智能医用软件产品分类界定指导原则》(2021 年第 47 号通告)
5. 《互联网诊疗监管细则（试行）》(2022) 第 13、21 条
6. WS/T 405-2012《血细胞分析参考区间》；WS/T 404 系列《临床常用生化检验项目参考区间》

---

## 14. 如果接手后要继续做，我建议的顺序

1. **先补 README 截图**（10 分钟，性价比最高）
2. **把 GitHub 补齐**（网络恢复后一条命令）
3. **轮换 Token**
4. 想加功能的话，从"**危急值阈值按机构可配置**"开始——
   这是现在最实际的产品缺口（知识库里的阈值是常见默认值，各院不同）
5. 面试讲解时，把叙事锚在**三条硬约束**和**评测跑出来的 6 个真实缺陷**上，
   而不是"我做了个界面"——前者体现判断力，后者任何人都能做
