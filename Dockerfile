FROM python:3.10-slim

WORKDIR /workspace

# 1️⃣ 先拷贝项目一
COPY jinsie-ai-agent-platform ./jinsie-ai-agent-platform

# ⭐ 关键：先装 requirements
RUN pip install --no-cache-dir -r ./jinsie-ai-agent-platform/requirements.txt

# 再 editable 安装平台
RUN pip install --no-cache-dir -e ./jinsie-ai-agent-platform

# 2️⃣ 再拷贝项目二
COPY jinsie-rag-support-system ./jinsie-rag-support-system
RUN pip install --no-cache-dir fastapi uvicorn

WORKDIR /workspace/jinsie-rag-support-system

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
