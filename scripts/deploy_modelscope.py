"""把 LabLens 部署到魔搭社区（ModelScope）创空间。

为什么用 Python 而不是 PowerShell / curl：
    本机 PowerShell 的 Invoke-RestMethod 存在 TLS 问题（连 api.github.com 也失败），
    而 httpx（项目已有依赖）访问 modelscope.cn 完全正常。用 httpx 既可靠又跨平台。

为什么不用 modelscope SDK：
    官方 OpenAPI 是 source of truth，直接调 API 少一层依赖，
    也避免 `ms` / `modelscope` 两个 CLI 入口互相覆盖的坑（官方文档明确提到这个问题）。

用法::

    set MODELSCOPE_API_KEY=你的令牌        # Windows PowerShell: $env:MODELSCOPE_API_KEY="..."
    python scripts/deploy_modelscope.py --owner <你的魔搭用户名>

    # 顺手把通义千问 Key 配成创空间 secret（访客打开即可直接用模型）
    python scripts/deploy_modelscope.py --owner <名> --dashscope-key sk-xxxx

    # 部署并跟踪启动日志
    python scripts/deploy_modelscope.py --owner <名> --tail-logs

令牌获取：https://modelscope.cn/my/myaccesstoken
（国内站 modelscope.cn 与国际站 modelscope.ai 的账号与令牌互不通用。）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

# 查证于 2026-09：Streamlit 类型当前只有一个免费规格，不存在付费选项
DEFAULT_HARDWARE = "platform/2v-cpu-16g-mem"
DEFAULT_ENDPOINT = "https://modelscope.cn"


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"    {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"    ⚠️  {msg}", flush=True)


def run_git(args: list[str], cwd: Path, check: bool = True) -> int:
    """跑 git 命令。

    刻意**不捕获输出**（不设 capture_output）：受限环境下管道可能被拒绝，
    让 git 直接继承终端的 stdio 最稳妥，用户也能实时看到进度。
    """
    result = subprocess.run(["git", *args], cwd=cwd, check=False)
    if check and result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败（退出码 {result.returncode}）")
    return result.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="部署 LabLens 到魔搭创空间")
    ap.add_argument("--owner", required=True, help="魔搭用户名或组织名（链接里的拥有者）")
    ap.add_argument("--repo", default="lablens", help="创空间名称，默认 lablens")
    ap.add_argument("--endpoint", default=os.environ.get("MODELSCOPE_ENDPOINT", DEFAULT_ENDPOINT),
                    help="站点地址，默认国内站 https://modelscope.cn")
    ap.add_argument("--visibility", choices=["private", "public"], default="private")
    ap.add_argument("--hardware", default=DEFAULT_HARDWARE)
    ap.add_argument("--display-name", default="LabLens · 中文检验报告解读（技术演示）")
    ap.add_argument("--dashscope-key", default="", help="可选：写入创空间 secret，供线上直接调用模型")
    ap.add_argument("--skip-deploy", action="store_true", help="只推代码，不触发部署")
    ap.add_argument("--tail-logs", action="store_true", help="部署后轮询运行日志")
    ap.add_argument("--timeout", type=int, default=600, help="--tail-logs 的最长等待秒数，默认 600")
    args = ap.parse_args()

    api = args.endpoint.rstrip("/") + "/openapi/v1"
    studio_url = f"{args.endpoint.rstrip('/')}/studios/{args.owner}/{args.repo}"

    # ----------------------------------------------------------------- #
    step("检查前置条件")

    token = os.environ.get("MODELSCOPE_API_KEY", "").strip()
    if not token:
        raise SystemExit(
            "缺少环境变量 MODELSCOPE_API_KEY。\n"
            f"    请到 {args.endpoint}/my/myaccesstoken 获取后设置：\n"
            '    PowerShell:  $env:MODELSCOPE_API_KEY = "..."\n'
            '    bash:        export MODELSCOPE_API_KEY="..."'
        )
    ok(f"已读取 MODELSCOPE_API_KEY（长度 {len(token)}）")

    if not (ROOT / "app.py").is_file():
        raise SystemExit("未找到 app.py —— Streamlit SDK 要求入口文件是根目录下的 app.py")
    if not (ROOT / "requirements.txt").is_file():
        warn("缺少 requirements.txt —— 创空间不读 pyproject.toml，依赖会装不上")
    else:
        ok("本地项目检查通过")

    client = httpx.Client(
        base_url=api,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60.0,
    )

    # ----------------------------------------------------------------- #
    step("验证令牌并获取账号信息")
    try:
        me = client.get("/users/me").json()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"无法连接 {api}：{exc}") from exc
    if not me.get("success"):
        raise SystemExit(
            f"令牌验证失败：{me.get('code')} {me.get('message')}\n"
            "    提示：确认令牌所属站点与 --endpoint 一致（国内站/国际站令牌不互通）"
        )
    data = me.get("data") or {}
    ok(f"登录身份：{data.get('name') or data.get('username')}")

    # ----------------------------------------------------------------- #
    step("查询可用硬件规格")
    try:
        hw = client.get("/studios/hardware", params={"sdk_type": "streamlit"}).json()
        items = ((hw.get("data") or {}).get("hardware")) or []
        free = [h for h in items if h.get("resource_type") != "paid"]
        paid = [h for h in items if h.get("resource_type") == "paid"]
        for h in free:
            ok(f"免费规格：{h.get('name')}")
        if paid:
            warn(f"存在 {len(paid)} 项付费规格；本脚本不会选用，避免扣费")
        if free and args.hardware not in {h.get("name") for h in free}:
            warn(f"指定的 {args.hardware} 不在免费列表里，已回退到 {free[0].get('name')}")
            args.hardware = str(free[0].get("name"))
    except Exception as exc:  # noqa: BLE001
        warn(f"查询硬件失败（{exc}），沿用 {args.hardware}")

    # ----------------------------------------------------------------- #
    step(f"创建或复用创空间 {args.owner}/{args.repo}")
    detail = client.get(f"/studios/{args.owner}/{args.repo}").json()
    if detail.get("success"):
        ok("已存在，更新设置")
        r = client.patch(
            f"/studios/{args.owner}/{args.repo}/settings",
            json={"sdk_type": "streamlit", "display_name": args.display_name},
        ).json()
        if not r.get("success"):
            warn(f"更新设置失败：{r.get('message')}")
    else:
        r = client.post(
            "/studios",
            json={
                "owner": args.owner,
                "repo_name": args.repo,
                "sdk_type": "streamlit",
                "visibility": args.visibility,
                "hardware": args.hardware,
                "display_name": args.display_name,
            },
        ).json()
        if not r.get("success"):
            raise SystemExit(f"创建失败：{r.get('code')} {r.get('message')}")
        ok(f"创建成功（可见性：{args.visibility}，规格：{args.hardware}）")

    # ----------------------------------------------------------------- #
    step("推送代码（创空间默认分支是 master，禁止 force push）")

    host = args.endpoint.rstrip("/").replace("https://", "").replace("http://", "")
    clean_remote = f"https://{host}/studios/{args.owner}/{args.repo}.git"

    init_rc = run_git(["rev-parse", "--git-dir"], ROOT, check=False)
    if init_rc != 0:
        run_git(["init"], ROOT)

    remote_url = f"https://oauth2:{token}@{host}/studios/{args.owner}/{args.repo}.git"
    ident = [
        "-c", f"user.name={os.environ.get('GIT_AUTHOR_NAME', 'Runze Yang')}",
        "-c", f"user.email={os.environ.get('GIT_AUTHOR_EMAIL', 'y18195226719@gmail.com')}",
    ]
    try:
        run_git(["remote", "remove", "modelscope"], ROOT, check=False)
        run_git(["remote", "add", "modelscope", remote_url], ROOT)
        run_git(["add", "-A"], ROOT, check=False)
        # 没有改动时 commit 会失败，这是正常的，所以 check=False
        run_git([*ident, "commit", "-q", "-m", "chore: sync for ModelScope deployment"], ROOT, check=False)

        # 新建的创空间里已经有平台初始化的一份内容（README.md 等）。
        # 直接 push 会被以 non-fast-forward 拒绝，所以必须先 fetch 再 merge 一次。
        if run_git(["fetch", "modelscope", "master"], ROOT, check=False) == 0:
            merged = run_git(
                [*ident, "merge", "modelscope/master", "--allow-unrelated-histories", "--no-edit"],
                ROOT,
                check=False,
            )
            if merged != 0:
                # 冲突时一律保留本地版本：平台初始化的是占位文件，没有保留价值
                warn("与创空间初始内容有冲突，保留本地版本")
                run_git(["checkout", "--ours", "."], ROOT, check=False)
                run_git(["add", "-A"], ROOT, check=False)
                run_git(
                    [*ident, "commit", "-q", "-m", "Merge ModelScope init content, keep local"],
                    ROOT,
                    check=False,
                )

        # 本地分支可能叫 main，创空间只认 master，用 refspec 显式指定
        run_git(["push", "modelscope", "HEAD:master"], ROOT)
        ok("已推送到 master")
    finally:
        # 无论成败都把 remote 还原成不含令牌的干净地址
        run_git(["remote", "set-url", "modelscope", clean_remote], ROOT, check=False)

    # ----------------------------------------------------------------- #
    if args.dashscope_key:
        step("写入模型密钥（走 secrets，接口不会回显 value）")
        body = {"key": "DASHSCOPE_API_KEY", "value": args.dashscope_key}
        r = client.post(f"/studios/{args.owner}/{args.repo}/secrets", json=body).json()
        if not r.get("success"):
            r = client.put(f"/studios/{args.owner}/{args.repo}/secrets", json=body).json()
        if r.get("success"):
            ok("secret DASHSCOPE_API_KEY 已配置")
        else:
            warn(f"写入 secret 失败：{r.get('message')}")

    # ----------------------------------------------------------------- #
    if not args.skip_deploy:
        step("触发部署")
        r = client.post(f"/studios/{args.owner}/{args.repo}/deploy").json()
        if not r.get("success"):
            warn(f"部署请求返回：{r.get('code')} {r.get('message')}")
        else:
            ok("部署已触发")

        if args.tail_logs:
            step(f"轮询构建/运行状态与日志（首次构建实测约 5 分钟，上限 {args.timeout}s）")
            deadline = time.time() + args.timeout
            finished = False
            i = 0
            while time.time() < deadline and not finished:
                i += 1
                time.sleep(10)
                try:
                    info = (client.get(f"/studios/{args.owner}/{args.repo}").json().get("data")) or {}
                    status = str(info.get("status") or "?")

                    lg = client.get(f"/studios/{args.owner}/{args.repo}/logs/run").json()
                    data = lg.get("data")
                    if isinstance(data, dict):
                        lines = data.get("logs") or []
                    elif isinstance(data, list):
                        lines = data
                    else:
                        lines = [data] if data else []
                    text = "\n".join(
                        ln if isinstance(ln, str) else json.dumps(ln, ensure_ascii=False)
                        for ln in lines
                    )

                    last = next((ln for ln in reversed(text.splitlines()) if ln.strip()), "")
                    print(f"    [{i:02d}] status={status}  {last.strip()[:120]}", flush=True)

                    if status.lower() == "running":
                        # 状态为 Running 只说明进程起来了，再看一眼日志里有没有启动异常
                        if any(k in text for k in ("ModuleNotFoundError", "SyntaxError", "Traceback")):
                            warn("状态 Running，但日志中存在异常，请人工确认")
                            print(text[-3000:])
                        else:
                            ok("状态：Running，启动成功")
                        finished = True
                    elif status.lower() in {"failed", "error", "stopped", "crashed"}:
                        warn(f"状态：{status}")
                        if text:
                            print(text[-4000:])
                        finished = True
                    # Building / Deploying / Pending 都是中间态，继续等
                except Exception as exc:  # noqa: BLE001
                    warn(f"[{i:02d}] 查询失败：{exc}")
            if not finished:
                warn(f"等待超过 {args.timeout}s 仍未就绪。首次构建通常较慢，可稍后重新运行本脚本，"
                     "或到网页控制台的日志页查看")

    # ----------------------------------------------------------------- #
    print(f"\n{'=' * 64}")
    print(f"创空间地址：{studio_url}")
    print(f"{'=' * 64}")
    print(
        "\n后续手动步骤：\n"
        "  1. 打开上面的地址验收（内置合成示例可离线体验，无需上传任何真实报告）\n"
        "  2. 确认无误后改公开：\n"
        f"     python scripts/deploy_modelscope.py --owner {args.owner} --visibility public --skip-deploy\n"
        "     或直接在网页控制台改\n"
        "  3. 注意：创空间无法通过 API 删除，只能在网页控制台删（程序侧只能 stop）\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
