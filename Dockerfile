FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app/src:/app PYTHONUNBUFFERED=1 HF_HOME=/app/.hf OLLAMA_URL=http://host.docker.internal:11434/api/chat
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY src ./src
COPY api ./api
COPY data/processed ./data/processed
RUN python -m rag.index data/processed/chunks_1000_150.jsonl
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
