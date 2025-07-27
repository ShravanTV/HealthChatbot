FROM python:3.11-slim
WORKDIR /app

# Install curl for healthcheck/model pull
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY ./api ./api
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Entrypoint script to pull model from Ollama before starting backend
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
