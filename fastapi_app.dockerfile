# 使用官方 Python 轻量版本作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制 FastAPI 应用代码
COPY fake_news_detection/ fake_news_detection/

# ✅ 复制模型文件（重点！）
COPY models/ models/

# 启动 FastAPI 应用
EXPOSE 8080
CMD ["uvicorn", "fake_news_detection.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
