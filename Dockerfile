# LabLens 容器镜像
#
# 构建：
#   docker build -t lablens .
#
# 运行界面（内置合成示例可直接体验，无需上传任何真实数据）：
#   docker run --rm -p 8501:8501 lablens
#   然后打开 http://localhost:8501
#
# 跑评测 / 测试：
#   docker run --rm lablens python evals/run_eval.py
#   docker run --rm lablens pytest -q
#
# 接本地模型（推荐用于真实数据，报告不出容器）：
#   docker run --rm -p 8501:8501 \
#     -e LABLENS_BASE_URL=http://host.docker.internal:8000/v1 \
#     -e LABLENS_MODEL=Qwen/Qwen3-VL-8B-Instruct \
#     -e LABLENS_VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct \
#     lablens

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# 依赖单独一层，改代码时不必重装
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[app]" \
    && pip install --no-cache-dir pytest ruff

COPY core/ ./core/
COPY data/ ./data/
COPY evals/ ./evals/
COPY tests/ ./tests/
COPY app.py mcp_server.py ./

# 以非 root 运行
RUN useradd --create-home --shell /bin/bash lablens \
    && chown -R lablens:lablens /app
USER lablens

EXPOSE 8501

# 默认启动界面；也可用 `docker run --rm -i lablens python mcp_server.py` 以 stdio 方式跑 MCP
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
