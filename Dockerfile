# 使用官方 python 镜像作为基础镜像
FROM python:3.11-slim AS base
LABEL maintainer="YuanJie" \
    description="An Acoustic Feature Detection Mirror Construction Project" \
    license="MIT" \
    email="wangjh0825@qq.com"

# 写入阿里云 Debian 12 源（deb822 格式）
RUN cat > /etc/apt/sources.list.d/debian.sources <<'EOF'
Types: deb
URIs: http://mirrors.aliyun.com/debian
Suites: bookworm bookworm-updates
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb
URIs: http://mirrors.aliyun.com/debian-security
Suites: bookworm-security
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
EOF

# 系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    procps \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /workspace/models/hf/datasets /workspace/logs

# 用 PyPI 国内源安装 uv（稳定）
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# 增加 uv 下载超时（单位秒，建议 300+）
ENV UV_HTTP_TIMEOUT=600
# 用 pip3 安装 uv
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -U uv -i ${UV_INDEX_URL} && \
    uv --version

ENV PATH="/root/.local/bin:${PATH}"

# Hugging Face 国内配置
# 模型缓存目录
ENV HF_ENDPOINT=https://hf-mirrors.com
ENV HF_HOME=/workspace/models/hf
ENV HF_DATASETS_CACHE=/workspace/models/hf/datasets
# ENV TRANSFORMERS_OFFLINE=1
# ENV HF_DATASETS_OFFLINE=1
# 设置工作目录
WORKDIR /workspace

# 拷贝依赖文件（先拷贝依赖有利于 Docker 层缓存）
COPY pyproject.toml .python-version ./

# 同步依赖，--active 强制使用当前 venv，避免重建
RUN uv sync --python 3.11 --active
RUN uv pip install transformers==4.57.6 qwen-asr
# 再拷贝项目代码
COPY . .

# 创建必要的日志目录并赋予 run.sh 执行权限
RUN chmod +x run.sh && mkdir -p logs

# uv 会创建 .venv，这里将其添加到 PATH
ENV PATH="/workspace/.venv/bin:${PATH}"

# 暴露 FastAPI 运行端口
EXPOSE 8001

# 容器启动命令
CMD ["bash", "run.sh", "start"]