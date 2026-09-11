# LabLens 交接文档

> 最后更新：2026-09-11　｜　当前提交：`19ab105`　｜　状态：**已上线，可交付**

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
| **代码规模** | 7,215 行 Python（core 3011 / tests 1670 / root 1077 / ui 600 / evals 553 / scripts 304） |
| **测试** | **218 个，全绿**（含 20 个端到端界面测试） |
| **评测** | 生产路径准确率 **100.0%**，危急值漏报 **0** 项 |
| **知识库** | 95 项指标字典 + 574 条解释语料 |

---

## 2. 当前状态（事实）

### 2.1 代码与部署

```
分支            main
提交数          11
HEAD            19ab105  chore: sync for ModelScope deployment
未提交改动      无
远端 origin     https://github.com/yang-rz-602/lablens.git
远端 modelscope https://modelscope.cn/studios/yangrz2222/lablens.git
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

### 2.2 ⚠️ 两件未完成的事

| # | 事项 | 影响 | 怎么处理 |
|---|---|---|---|
| 1 | **GitHub 有 2 个提交未推送**（`ff29820`、`19ab105`） | GitHub 上的代码落后于 ModelScope。含全部界面美化改动（1446 行） | 这台机器到 `github.com` 持续超时，重试 5 次均失败。网络恢复后 `git push origin main` 即可。**ModelScope 不受影响，已是最新代码** |
| 2 | **README 缺界面截图** | 面试官在 GitHub 上只能看文字 | 打开线上地址截 2 张（②指标与判读依据页、③解读页）放进 `docs/images/`。注意：沙箱禁止命名管道，Playwright 与 Chrome 都跑不起来，**必须人工截** |

### 2.3 本地环境版本

```
Python       3.14.4   （⚠️ 创空间跑的是 3.11，新代码需兼容 3.11）
pydantic     2.13.5
httpx        0.28.1
PyYAML       6.0.3
streamlit    1.63.0
pandas       3.0.5
duckdb       1.5.5
pytest       9.1.1
ruff         0.16.7
```

---

## 3. 快速上手

```bash
cd lablens

# 1) 建环境（用 uv 最快；普通 venv 也行）
uv venv .venv
uv pip install --python .venv pydantic httpx PyYAML pytest duckdb streamlit

# 2) 跑测试（218 个，约 1 分钟）
.venv/Scripts/python -m pytest -q

# 3) 跑评测（不需要网络、不需要 API Key）
.venv/Scripts/python evals/run_eval.py

# 4) 起界面
.venv/Scripts/python -m streamlit run app.py
```

**不需要任何 API Key 就能跑通全部功能。** 没有 Key 时系统自动降级到
「离线抽取式解读」——判读、检索、拒答、护栏全部照常工作，只是解读文案由知识库原文拼装，
不如语言模型流畅。这是刻意的设计：**把可验证性从"有没有额度"里解耦出来**。

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
│ Stage 3  解释层（LLM + 约束检索 + 拒答） core/explain.py   │
│  · 先按 canonical_name 硬过滤 —— 绝不跨指标检索             │
│  · 再按判定结果排维度优先级（偏高→升高意义，偏低→降低意义）   │
│  · 得分低于阈值 → 返回"暂无可靠依据"，不硬编                  │
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

---

## 7. 目录与文件职责

```
lablens/
├── core/                       ★ 后端（3011 行）
│   ├── schema.py               三层数据契约 + DecisionTrace（Pydantic 强校验）
│   ├── units.py                单位归一与解析（纯函数，写法规格化）
│   ├── rules.py                ★ 唯一的数值判定层，可穷举单测
│   ├── retriever.py            约束检索（BM25 + 元数据硬过滤 + 维度优先级 + 拒答）
│   ├── providers/__init__.py   LLM 传输层（OpenAI 兼容 + 6 个供应商预设 + 离线夹具）
│   ├── extract.py              Stage 1 抽取（含防幻觉提示词）
│   ├── explain.py              Stage 3 解释（含抽取式降级路径）
│   ├── guard.py                ★ 输出护栏（正则确定性拦截）
│   └── store.py                趋势存储（默认关闭）
│
├── ui/                         界面层（600 行）
│   ├── theme.py                设计令牌 + 全局 CSS
│   └── components.py           纯函数组件（数据 → HTML），可单测
│
├── data/                       知识库
│   ├── indicators_cbc_urine.yaml   血常规 + 尿常规（28 项）
│   ├── indicators_biochem.yaml     生化 + 免疫（67 项）
│   └── corpus.jsonl                解释语料 574 条 / 82 指标（396 KB）
│
├── evals/                      评测（553 行）
│   ├── synthetic/ground_truth.json 20 份合成报告 / 276 项
│   ├── render.py                   把 ground truth 渲染成仿真报告（文本 + 图片）
│   ├── run_eval.py                 三条路径评测 + CI 安全闸门
│   └── RESULTS.md                  自动生成的评测报告
│
├── tests/                      218 个测试（1670 行）
│   ├── test_rules.py           48  判定逻辑（含拒判、性别区间、别名冲突消歧）
│   ├── test_units.py           41  单位归一与解析
│   ├── test_guard.py           29  输出护栏（含误伤检查）
│   ├── test_mcp_server.py      26  MCP 协议 + 工具
│   ├── test_app.py             20  端到端界面（AppTest，无需浏览器）
│   ├── test_retriever.py       19  约束检索与拒答
│   └── test_ui.py              35  界面组件纯函数
│
├── scripts/deploy_modelscope.py  一键部署（304 行）
├── docs/DEPLOY_MODELSCOPE.md     部署文档
├── mcp_server.py                 MCP 封装（零依赖手写 JSON-RPC）
├── app.py                        Streamlit 界面入口
├── HANDOVER.md                   ← 本文档
├── README.md                     对外门面
├── DISCLAIMER.md                 合规定位与依据
└── requirements.txt              创空间依赖清单
```

---

## 8. 关键设计决策（FAQ）

### Q：为什么不用向量数据库？是不是没做 RAG？

**没有向量库是刻意的，不是遗漏。** 项目依赖只有 `pydantic / httpx / PyYAML`，
没有 Chroma、FAISS、Milvus、Qdrant。

三个理由：

1. **候选集已经被硬过滤掉了。** `canonical_name` 由规则引擎查表得到，用它先过滤，
   组内只剩几条到十几条候选。向量检索擅长"从百万文档里捞相关"，这里没有这个问题。
2. **医疗场景下向量检索反而更危险。** 肌酐 / 尿素 / 胱抑素C 在语义空间里挨得极近——
   向量检索最容易做的事，恰好就是"串指标"，而这正是本项目最不能犯的错。
3. **可复现、零依赖。** 词法打分确定性、可单测；向量检索要引入 embedding 服务，
   多一个失败点和一个不确定性来源。

**升级接口已预留**：`Retriever(embed_fn=...)` 传入 embedding 函数即自动切换混合检索
（`mode` 从 `lexical` 变 `hybrid`，BM25 + 余弦加权）。有测试覆盖，且 embedding 挂掉时**自动降级回词法**。

**什么时候该上**：语料超过 2000 条，或要支持跨指标的语义查询（如"哪些指标和肾功能有关"）。现在是 574 条。

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

- [ ] **GitHub 有 2 个提交未推送**（`ff29820`、`19ab105`）—— 本机到 github.com 超时。
      网络恢复后 `git push origin main` 即可

### 建议补上（按性价比排序）

- [ ] **README 截图**（最高性价比）—— 面试官在 GitHub 上不会去点链接跑代码。
      打开线上地址截 2 张放进 `docs/images/`
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
- [ ] 语料超过 2000 条时接入 embedding，启用混合检索

### 明确不做（设计边界，别当成待办）

- ❌ 不出诊断结论
- ❌ 不给用药建议、不写剂量、不做处方相关表述（法规红线，代码强制）
- ❌ 不做综合风险评分或"红黄绿灯"分级（易被认定为辅助决策）
- ❌ 不做生活方式、饮食、运动建议

---

## 12. 常用操作速查

```bash
# 测试 / 质量
pytest -q                                   # 全部 218 个
pytest tests/test_rules.py -q               # 只跑判定逻辑
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
