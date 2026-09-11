# 部署到魔搭社区（ModelScope）创空间

> 本文依据魔搭官方 [`ms-studio-deploy` Skill](https://github.com/modelscope/modelscope-skills/blob/main/skills/ms-studio-deploy/SKILL.md) 整理，
> 该文档标注为 **verified live against ModelScope OpenAPI ｜ modelscope 1.37.1 / modelscope_hub 0.1.2（2026-06-29）**。
> 平台规则会变，执行前建议再核对一次官方文档。

## 一句话结论

LabLens 是 **Streamlit 应用**，选 `streamlit` SDK 类型即可：

- ✅ **不需要**实名认证
- ✅ **不需要** Docker（实名认证是 Docker 类型的前置要求）
- ✅ **不需要**关心端口（7860 的限制只针对 Docker 类型）
- ✅ **不需要**选规格——Streamlit 类型目前只有一个规格，且是免费

## 实测的可用规格（2026-09 查询 OpenAPI）

```
GET /openapi/v1/studios/hardware?sdk_type=streamlit
→ 仅 1 项：platform/2v-cpu-16g-mem   resource_type=free   （2 vCPU / 16GB）
```

**只有免费规格，不存在付费选项**，所以部署不会产生任何费用。

`GET /studios/sdk-versions?sdk_type=streamlit` 返回空——Streamlit 类型无需指定 SDK 版本，
依赖由 `requirements.txt` 安装。基础镜像共 24 个（Python 3.11 / 3.12）。

## 你需要准备什么

| 项 | 说明 | 谁来做 |
|---|---|---|
| **魔搭账号 + Access Token** | <https://modelscope.cn/my/myaccesstoken> | **只有你能做** |
| 本地 Git | 已具备 | ✅ |
| `app.py` 入口 | 已具备（Streamlit SDK 要求入口是根目录的 `app.py`） | ✅ |
| `requirements.txt` | 已具备（**创空间不读 pyproject.toml**，这是最常见的启动失败原因） | ✅ |
| Dockerfile | **不需要**（那是 Docker 类型才用的） | — |

### 两个容易踩的坑

1. **站点与令牌不互通。** 国内站 `modelscope.cn` 与国际站 `modelscope.ai` 是两套独立账号、独立令牌、独立内容。
   面向国内面试官就用国内站；拿错站点的令牌会直接 401。
2. **默认分支是 `master`，不是 `main`。** 仓库本地分支叫什么都不影响，
   用 `git push modelscope HEAD:master` 显式指定即可。**禁止 force push。**

## 一键部署

```powershell
$env:MODELSCOPE_API_KEY = "你的令牌"
python scripts/deploy_modelscope.py --owner <你的魔搭用户名>

# 顺便把通义千问的 Key 配成创空间 secret，访客打开即可直接用模型：
python scripts/deploy_modelscope.py --owner <你的魔搭用户名> --dashscope-key "sk-xxxx"

# 部署并跟踪启动日志：
python scripts/deploy_modelscope.py --owner <你的魔搭用户名> --tail-logs
```

脚本做这些事：验令牌 → 查可用硬件（**自动跳过一切付费规格**）→ 创建或复用创空间（默认私有）→
推送代码到 `master` → 写入 secret → 触发部署 → 可选轮询日志。
推完在 `finally` 里无条件把 remote 还原成不含令牌的干净地址。

> **为什么脚本用 Python 而不是 PowerShell**：本机 `Invoke-RestMethod` 存在 TLS 问题
> （连 `api.github.com` 都失败），而 `httpx`（项目已有依赖）访问 `modelscope.cn` 完全正常。
>
> **为什么把令牌内联到 Git URL**：本机沙箱会拦掉 git 凭证管理器的命名管道通信
> （报 `couldn't create signal pipe, Win32 error 5`）。URL 内联是唯一稳定可行的方式。

## 手动流程（对照官方 10 步）

如果想自己走一遍，官方流程是：

```bash
pip install modelscope                     # 提供 ms CLI
export MODELSCOPE_ENDPOINT="https://modelscope.cn"
export MODELSCOPE_API_KEY="<token>"

# 1-3. 查配置 → 创建空间（sdk_type 选 streamlit，visibility 建议先 private）
ms create <owner>/lablens --repo-type studio --sdk-type streamlit --private

# 5. 同步代码（唯一非 API 的一步）
git remote add modelscope https://oauth2:${MODELSCOPE_API_KEY}@modelscope.cn/studios/<owner>/lablens.git
git push -u modelscope HEAD:master

# 6. 配置密钥（secrets 只回显 key，不回显 value）
ms secret add <owner>/lablens DASHSCOPE_API_KEY sk-xxxx

# 7-8. 部署并看日志
ms deploy <owner>/lablens --repo-type studio
ms logs <owner>/lablens --log-type run

# 10. 改公开
ms settings <owner>/lablens --repo-type studio private=false
```

## LabLens 特有的三个注意点

### 1. API Key 用 secret，不要写进代码

代码已支持从环境变量读取（`app.py` 的 `make_client`：界面输入 > 服务端环境变量）。
配成 secret 后访客打开即可直接用模型：

```bash
ms secret add <owner>/lablens DASHSCOPE_API_KEY sk-xxxx
```

⚠️ **但要权衡额度**：公开创空间 + 你的 Key = 任何访客都在消耗你的额度。
更稳妥的做法是**不配 secret**，让 demo 跑离线模式（内置合成示例本来就能完整体验），
访客想看模型效果就自己填 Key。

### 2. 不要依赖本地存储

创空间的磁盘**每次重启都会丢**，持久化目录是 `/mnt/workspace`。
LabLens 的"趋势存储"默认就是关闭的，正好契合——**在线上保持关闭即可**，不用改代码。

### 3. 上传真实报告前想清楚

公开 demo 让访客上传检验报告图片，若走云端模型，图片会发到模型服务商。
控制措施已经内建：

- 内置合成示例，**访客可以不传任何数据就体验完整流程**
- 上传前有脱敏提示（遮挡姓名、就诊号、身份证号、条码）
- 三处 AI 生成内容标识 + 非医疗器械声明

## 平台已知限制（会影响后续维护）

| 限制 | 影响 |
|---|---|
| 创空间**无法通过 API 删除** | 只能在 <https://modelscope.cn> 网页控制台删；程序侧只能 `stop` |
| 免费额度**有时间限制** | 长期展示需留意额度，或转付费规格（会扣绑定的阿里云账号） |
| 仓库 / 文件删除同样限制在网页控制台 | 删错文件得去网页上删 |
| 单个文件 > 100MB 必须用 Git LFS | 本项目最大文件约 400KB，无影响 |
| 默认分支 `master`，禁 force push | 已在上文处理 |

## 验收清单

- [ ] `$Endpoint/studios/<owner>/lablens` 能打开
- [ ] 侧栏显示"指标字典 95 项 · 解释语料 574 条"
- [ ] 选一份内置示例 → 点"开始分析" → ②指标与判读依据页出表格
- [ ] 挑危急值那几份示例（`case_005` 血小板、`case_007` ALT、`case_012` 血钾、`case_017` 肌钙蛋白），确认出现红色危急值提示
- [ ] ③解读页能看到"本次解读的局限性"
- [ ] 未登录窗口也能打开（说明已公开）
- [ ] 手机宽度下可用
