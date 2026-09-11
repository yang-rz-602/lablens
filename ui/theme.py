"""LabLens 界面设计系统：设计令牌 + 全局样式。

为什么把样式集中在一处
----------------------
界面美化最容易变成"到处塞 inline style"，改一个颜色要翻十个文件。
这里把所有颜色、圆角、阴影收敛成令牌，组件层（``ui/components.py``）只引用令牌，
不写字面量颜色。

视觉方向：**临床、克制、可信任**。
不追求花哨——面向医疗场景，用户需要的是"这个数字我敢信"，
所以主色用一个沉稳的深青（而非鲜艳的蓝紫），状态色严格对应严重度，
危急值用红色但不过度渲染，正常值用低饱和的绿。
"""

from __future__ import annotations

import streamlit as st

__all__ = ["COLORS", "SEVERITY_STYLE", "FLAG_STYLE", "inject_css", "HTML_HEAD"]

# --------------------------------------------------------------------------- #
# 设计令牌
# --------------------------------------------------------------------------- #
COLORS = {
    "ink": "#202522",          # 主文字
    "ink_soft": "#59635E",     # 次级文字
    "muted": "#87908B",        # 弱化文字
    "line": "#DDD9D0",         # 分隔线
    "line_soft": "#F1EFE9",
    "surface": "#FFFDF9",      # 卡片
    "canvas": "#F5F2EC",       # 页面底色
    "brand": "#176B65",        # 主色：沉稳的深青
    "brand_dark": "#10534F",
    "brand_soft": "#EDF5F2",
    "brand_line": "#BBD8D1",
}

# 严重度 → 视觉（组件层统一从这里取色）
SEVERITY_STYLE = {
    "critical": {"fg": "#B91C1C", "bg": "#FEF2F2", "bd": "#FECACA", "dot": "#DC2626"},
    "abnormal": {"fg": "#C2410C", "bg": "#FFF7ED", "bd": "#FED7AA", "dot": "#EA580C"},
    "borderline": {"fg": "#A16207", "bg": "#FEFCE8", "bd": "#FEF08A", "dot": "#CA8A04"},
    "normal": {"fg": "#047857", "bg": "#ECFDF5", "bd": "#A7F3D0", "dot": "#059669"},
    "unknown": {"fg": "#475569", "bg": "#F8FAFC", "bd": "#E2E8F0", "dot": "#94A3B8"},
}

# 判定标记 → 展示文案与严重度
FLAG_STYLE = {
    "CH": ("危急偏高", "critical"),
    "H": ("偏高", "abnormal"),
    "N": ("正常", "normal"),
    "L": ("偏低", "abnormal"),
    "CL": ("危急偏低", "critical"),
    "?": ("未判定", "unknown"),
}

HTML_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
"""


def _css() -> str:
    c = COLORS
    return f"""
<style>
/* ===================== 基础排版 ===================== */
html, body, [class*="css"], .stApp {{
    font-family: system-ui, -apple-system, "Segoe UI", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
    color: {c['ink']};
}}
.stApp {{ background: {c['canvas']}; }}

/* 收敛 Streamlit 默认留白，让内容更紧凑 */
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }}
[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

h1, h2, h3, h4 {{ color: {c['ink']}; font-weight: 650; letter-spacing: -0.01em; }}
h1 {{ font-size: 1.9rem !important; }}
h2 {{ font-size: 1.3rem !important; }}
h3 {{ font-size: 1.05rem !important; }}
p, li, .stMarkdown {{ color: {c['ink_soft']}; line-height: 1.75; }}
code {{
    background: {c['line_soft']}; color: {c['brand_dark']};
    padding: .12em .4em; border-radius: 5px; font-size: .86em;
}}
hr {{ border-color: {c['line']}; margin: 1.4rem 0; }}

/* ===================== 侧栏 ===================== */
[data-testid="stSidebar"] {{
    background: {c['surface']};
    border-right: 1px solid {c['line']};
}}
[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
[data-testid="stSidebar"] hr {{ margin: .9rem 0; }}
.ll-side-brand {{ display: flex; align-items: center; gap: .55rem; margin-bottom: .15rem; }}
.ll-brand-mark {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.7rem; height: 1.7rem; border-radius: 6px;
    background: {c['brand']}; color: #fff; font-size: .88rem; font-weight: 750;
    letter-spacing: .02em;
}}
.ll-side-name {{ color: {c['ink']}; font-size: 1.1rem; font-weight: 720; letter-spacing: -.02em; }}
.ll-side-sub {{ color: {c['muted']}; font-size: .77rem; margin: 0 0 1rem 2.25rem; }}

/* ===================== 标签页 → 分段控件 ===================== */
.stTabs [data-baseweb="tab-list"] {{
    gap: .9rem; background: transparent;
    padding: 0; border-radius: 0; border: 0; border-bottom: 1px solid {c['line']};
}}
.stTabs [data-baseweb="tab-list"] button {{
    border-radius: 0; padding: .5rem .15rem .65rem; height: auto;
    font-weight: 600; color: {c['ink_soft']}; background: transparent;
}}
.stTabs [data-baseweb="tab-list"] button:hover {{ color: {c['brand_dark']}; background: transparent; }}
.stTabs [aria-selected="true"] {{
    background: transparent !important; color: {c['brand_dark']} !important;
    box-shadow: inset 0 -2px 0 {c['brand']};
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.4rem; }}

/* ===================== 按钮 ===================== */
.stButton > button, .stDownloadButton > button {{
    border-radius: 7px; font-weight: 600; border: 1px solid {c['line']};
    padding: .5rem 1.1rem; transition: all .15s ease;
}}
.stButton > button[kind="primary"] {{
    background: {c['brand']}; border-color: {c['brand']}; color: #fff;
    box-shadow: 0 1px 2px rgba(13,148,136,.25);
}}
.stButton > button[kind="primary"]:hover {{
    background: {c['brand_dark']}; border-color: {c['brand_dark']};
    box-shadow: 0 3px 10px rgba(23,107,101,.22); transform: translateY(-1px);
}}
.stButton > button:disabled {{ opacity: .45; }}

/* ===================== 表单控件 ===================== */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {{
    border-radius: 9px !important; border-color: {c['line']} !important;
}}
[data-baseweb="input"]:focus-within, [data-baseweb="textarea"]:focus-within {{
    border-color: {c['brand']} !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,.12) !important;
}}

/* ===================== 展开面板 → 卡片 ===================== */
[data-testid="stExpander"] {{
    border: 1px solid {c['line']}; border-radius: 8px;
    background: {c['surface']}; overflow: hidden;
}}
[data-testid="stExpander"] summary {{ font-weight: 550; padding: .35rem 0; }}
[data-testid="stExpander"] summary:hover {{ color: {c['brand_dark']}; }}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{ padding-top: .4rem; }}

/* ===================== 数据表 ===================== */
[data-testid="stDataFrame"] {{
    border: 1px solid {c['line']}; border-radius: 12px; overflow: hidden;
}}
[data-testid="stImage"] img {{
    border: 1px solid {c['line']}; border-radius: 8px;
    box-shadow: 0 7px 24px rgba(42, 48, 43, .06);
}}

/* ===================== 提示条 ===================== */
[data-testid="stAlert"] {{ border-radius: 11px; border-width: 1px; padding: .85rem 1rem; }}

/* ===================== 自定义组件 ===================== */
.ll-hero {{
    background: {c['surface']}; border: 1px solid {c['line']};
    border-left: 4px solid {c['brand']}; border-radius: 8px;
    padding: 1.35rem 1.55rem; margin-bottom: 1.1rem;
    box-shadow: 0 5px 18px rgba(42, 48, 43, .035);
}}
.ll-hero h1 {{
    margin: 0 0 .45rem 0 !important; font-size: 1.75rem !important;
    letter-spacing: -.02em;
}}
.ll-hero .ll-sub {{ color: {c['ink_soft']}; font-size: .95rem; margin: 0 0 .9rem 0; }}
.ll-chips {{ display: flex; flex-wrap: wrap; gap: .4rem; }}
.ll-chip {{
    display: inline-flex; align-items: center; gap: .3rem;
    background: {c['canvas']}; border: 1px solid {c['line']};
    color: {c['brand_dark']}; font-size: .78rem; font-weight: 550;
    padding: .22rem .6rem; border-radius: 4px;
}}

.ll-flow-divider {{
    display: flex; align-items: center; gap: .7rem;
    margin: 1.3rem 0 .1rem; color: {c['muted']};
}}
.ll-flow-label, .ll-flow-copy {{
    font-size: .7rem; font-weight: 700; letter-spacing: .08em; white-space: nowrap;
}}
.ll-flow-label {{ color: {c['brand']}; text-transform: uppercase; }}
.ll-flow-copy {{ color: {c['muted']}; letter-spacing: .04em; }}
.ll-flow-track {{
    position: relative; display: block; flex: 1; height: 1px;
    background: {c['line']}; overflow: hidden;
}}
.ll-flow-track::after {{
    content: ""; position: absolute; inset: 0; width: 26%;
    background: linear-gradient(90deg, transparent, {c['brand_line']}, transparent);
    animation: ll-flow-sweep 3.4s ease-in-out infinite;
}}
.ll-flow-dot {{
    position: absolute; z-index: 1; top: -3px; left: 0; width: 7px; height: 7px;
    border-radius: 50%; background: {c['brand']};
    box-shadow: 0 0 0 3px {c['brand_soft']};
    animation: ll-flow-travel 3.4s ease-in-out infinite;
}}
@keyframes ll-flow-sweep {{
    0% {{ transform: translateX(-120%); }}
    55%, 100% {{ transform: translateX(390%); }}
}}
@keyframes ll-flow-travel {{
    0% {{ left: 0; }}
    100% {{ left: calc(100% - 7px); }}
}}

.ll-disclaimer {{
    display: flex; gap: .7rem; align-items: flex-start;
    background: #FFFBEB; border: 1px solid #FDE68A;
    border-radius: 12px; padding: .85rem 1.05rem; margin-bottom: 1.2rem;
}}
.ll-disclaimer .ll-ic {{ flex: 0 0 auto; margin-top: .12rem; }}
.ll-disclaimer .ll-tx {{ font-size: .845rem; color: #78350F; line-height: 1.65; }}
.ll-disclaimer b {{ color: #92400E; }}

.ll-alert {{
    display: flex; gap: .8rem; align-items: flex-start;
    border-radius: 13px; padding: 1rem 1.15rem; margin: .3rem 0 1.1rem 0;
}}
.ll-alert.critical {{ background: #FEF2F2; border: 1px solid #FECACA; }}
.ll-alert .ll-tx {{ font-size: .9rem; line-height: 1.7; color: #7F1D1D; }}
.ll-alert .ll-title {{ font-weight: 700; color: #B91C1C; display: block; margin-bottom: .2rem; }}
.ll-alert code {{ background: rgba(185,28,28,.09); color: #B91C1C; }}

.ll-cards {{ display: grid; gap: .7rem; margin-bottom: 1.2rem; }}
.ll-cards.c3 {{ grid-template-columns: repeat(3, 1fr); }}
.ll-cards.c5 {{ grid-template-columns: repeat(5, 1fr); }}
.ll-card {{
    background: {c['surface']}; border: 1px solid {c['line']};
    border-radius: 8px; padding: .85rem 1rem; position: relative; overflow: hidden;
}}
.ll-card::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: var(--accent, {c['line']});
}}
.ll-card .ll-k {{
    font-size: .765rem; color: {c['muted']}; font-weight: 550;
    letter-spacing: .02em; margin-bottom: .3rem;
}}
.ll-card .ll-v {{
    font-size: 1.6rem; font-weight: 680; line-height: 1.1;
    font-variant-numeric: tabular-nums; color: var(--fg, {c['ink']});
}}
.ll-card .ll-u {{ font-size: .75rem; color: {c['muted']}; margin-left: .2rem; }}

.ll-sec {{
    display: flex; align-items: baseline; gap: .6rem;
    margin: 1.5rem 0 .7rem 0;
}}
.ll-sec .ll-sec-t {{ font-size: 1.02rem; font-weight: 650; color: {c['ink']}; }}
.ll-sec .ll-sec-h {{ font-size: .8rem; color: {c['muted']}; }}

.ll-tbl {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: .855rem; }}
.ll-tbl thead th {{
    background: {c['line_soft']}; color: {c['ink_soft']};
    font-weight: 600; font-size: .78rem; text-align: left;
    padding: .6rem .75rem; border-bottom: 1px solid {c['line']}; white-space: nowrap;
}}
.ll-tbl thead th:first-child {{ border-top-left-radius: 11px; }}
.ll-tbl thead th:last-child {{ border-top-right-radius: 11px; }}
.ll-tbl tbody td {{
    padding: .6rem .75rem; border-bottom: 1px solid {c['line_soft']};
    color: {c['ink_soft']}; vertical-align: middle;
}}
.ll-tbl tbody tr:hover td {{ background: {c['brand_soft']}; }}
.ll-tbl tbody tr:last-child td {{ border-bottom: none; }}
.ll-tbl .num {{
    font-variant-numeric: tabular-nums; font-weight: 620; color: {c['ink']};
    font-family: "SF Mono", "JetBrains Mono", Consolas, monospace;
}}
.ll-tbl .rn {{ color: {c['ink']}; font-weight: 550; }}
.ll-tbl-wrap {{
    border: 1px solid {c['line']}; border-radius: 12px;
    overflow: hidden; background: {c['surface']};
}}
.ll-src {{
    display: inline-block; font-size: .7rem; padding: .1rem .42rem;
    border-radius: 5px; background: {c['line_soft']}; color: {c['muted']};
}}
.ll-src.report {{ background: {c['brand_soft']}; color: {c['brand_dark']}; }}

.ll-badge {{
    display: inline-flex; align-items: center; gap: .28rem; white-space: nowrap;
    font-size: .755rem; font-weight: 620; padding: .17rem .52rem; border-radius: 6px;
}}
.ll-badge .dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}

.ll-item {{
    background: {c['surface']}; border: 1px solid {c['line']};
    border-left: 3px solid var(--accent, {c['line']});
    border-radius: 12px; padding: .95rem 1.15rem; margin-bottom: .7rem;
}}
.ll-item .ll-hd {{
    display: flex; flex-wrap: wrap; align-items: center; gap: .55rem;
    margin-bottom: .5rem;
}}
.ll-item .ll-nm {{ font-weight: 640; color: {c['ink']}; font-size: .96rem; }}
.ll-item .ll-val {{
    font-family: "SF Mono", "JetBrains Mono", Consolas, monospace;
    font-weight: 620; font-size: .95rem; color: var(--fg, {c['ink']});
}}
.ll-item .ll-ref {{ font-size: .78rem; color: {c['muted']}; }}
.ll-item .ll-blk {{ margin: .45rem 0 0 0; font-size: .865rem; line-height: 1.72; }}
.ll-item .ll-lb {{
    display: inline-block; font-size: .76rem; font-weight: 620;
    color: {c['muted']}; margin-right: .35rem;
}}
.ll-item ul {{ margin: .2rem 0 0 1.05rem; padding: 0; }}
.ll-item li {{ font-size: .85rem; line-height: 1.7; margin-bottom: .12rem; }}
.ll-trace {{
    margin-top: .6rem; padding: .55rem .7rem; background: {c['canvas']};
    border: 1px dashed {c['line']}; border-radius: 9px;
    font-size: .765rem; color: {c['ink_soft']}; line-height: 1.7;
}}
.ll-trace code {{ background: rgba(13,148,136,.09); }}
.ll-src-text {{
    margin-top: .5rem; font-family: "SF Mono", Consolas, monospace;
    font-size: .735rem; color: {c['muted']}; background: {c['line_soft']};
    padding: .4rem .6rem; border-radius: 7px; word-break: break-all;
}}

.ll-empty {{
    text-align: center; padding: 2.6rem 1rem; color: {c['muted']};
    background: {c['surface']}; border: 1px dashed {c['line']}; border-radius: 14px;
}}
.ll-empty .ll-ei {{ font-size: 1.9rem; display: block; margin-bottom: .5rem; opacity: .6; }}
.ll-empty .ll-et {{ font-size: .9rem; }}

.ll-note {{
    background: {c['brand_soft']}; border: 1px solid {c['brand_line']};
    border-radius: 11px; padding: .8rem 1rem; font-size: .845rem;
    color: {c['brand_dark']}; line-height: 1.7;
}}

/* ===================== 关于页 ===================== */
.ll-about-visual {{
    border: 1px solid {c['line']}; border-radius: 8px; overflow: hidden;
    background: {c['surface']}; box-shadow: 0 7px 24px rgba(42, 48, 43, .06);
}}
.ll-about-visual img {{ display: block; width: 100%; }}
.ll-about-copy {{ padding: .3rem .15rem .3rem .5rem; }}
.ll-about-kicker {{
    color: {c['brand']}; font-size: .72rem; font-weight: 750;
    letter-spacing: .12em; text-transform: uppercase; margin-bottom: .55rem;
}}
.ll-about-copy h2 {{ margin: 0 0 .55rem 0; font-size: 1.55rem !important; }}
.ll-about-copy p {{ margin: 0 0 .9rem 0; font-size: .92rem; }}
.ll-about-meta {{
    display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .55rem;
    margin-top: 1.1rem;
}}
.ll-about-fact {{ border-top: 1px solid {c['line']}; padding-top: .5rem; }}
.ll-about-fact b {{ display: block; color: {c['ink']}; font-size: 1.15rem; }}
.ll-about-fact span {{ color: {c['muted']}; font-size: .76rem; }}
.ll-principle-grid {{
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .7rem; margin: .2rem 0 1.2rem;
}}
.ll-principle {{
    background: {c['surface']}; border: 1px solid {c['line']}; border-radius: 8px;
    padding: .9rem 1rem; min-height: 7.5rem;
}}
.ll-principle .num {{ color: {c['brand']}; font-size: .72rem; font-weight: 750; letter-spacing: .08em; }}
.ll-principle b {{ display: block; color: {c['ink']}; margin: .35rem 0 .3rem; }}
.ll-principle span {{ color: {c['ink_soft']}; font-size: .82rem; line-height: 1.65; }}
.ll-proof {{
    display: flex; flex-wrap: wrap; gap: .65rem 1.4rem; align-items: center;
    background: {c['brand_soft']}; border: 1px solid {c['brand_line']}; border-radius: 8px;
    padding: .85rem 1rem; margin-bottom: .5rem;
}}
.ll-proof b {{ color: {c['brand_dark']}; font-size: 1.18rem; }}
.ll-proof span {{ color: {c['ink_soft']}; font-size: .82rem; }}

@media (max-width: 900px) {{
    .ll-cards.c5 {{ grid-template-columns: repeat(2, 1fr); }}
    .ll-cards.c3 {{ grid-template-columns: 1fr; }}
    .ll-hero {{ padding: 1.15rem 1.25rem; }}
    .ll-principle-grid {{ grid-template-columns: 1fr; }}
    .ll-about-copy {{ padding: 1rem 0 0; }}
}}
@media (prefers-reduced-motion: reduce) {{
    .ll-flow-track::after, .ll-flow-dot {{ animation: none; }}
    .ll-flow-dot {{ left: calc(50% - 3px); }}
}}
</style>
"""


def inject_css() -> None:
    """注入全局样式。每个 session 只需调用一次。"""
    st.markdown(_css(), unsafe_allow_html=True)
