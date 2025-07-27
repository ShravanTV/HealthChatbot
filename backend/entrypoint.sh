#!/bin/bash
set -e

# Wait for Ollama to be up
until curl -s http://ollama:11434 > /dev/null; do
  echo "Waiting for Ollama to be available..."
  sleep 2
done

echo "Ollama is up. Pulling nomic-embed-text model..."
curl -X POST http://ollama:11434/api/pull -d '{"name": "nomic-embed-text"}' -H 'Content-Type: application/json'

echo "Starting backend..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
