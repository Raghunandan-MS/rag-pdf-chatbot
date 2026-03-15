#!/bin/bash

echo "Starting Ollama server..."
ollama serve &

echo "Waiting for Ollama to wake up..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 2
done

echo "Pulling models (phi and nomic-embed-text)..."
ollama pull phi
ollama pull nomic-embed-text

echo "Starting Streamlit app..."
/app/.venv/bin/python3 -m streamlit run src/chatbox.py --server.port=8501 --server.address=0.0.0.0