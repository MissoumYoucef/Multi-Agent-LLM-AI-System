#!/bin/bash
QUERIES=(
    "What is a Large Language Model?"
    "How do LLMs differ from traditional AI?"
    "What is the role of attention mechanism in LLMs?"
    "How are LLMs trained?"
    "What is fine-tuning in the context of LLMs?"
    "Explain the concept of zero-shot learning."
    "What are the common challenges in LLM deployment?"
    "How does RAG (Retrieval-Augmented Generation) work?"
    "What is the difference between an LLM and an embedding model?"
    "What are the ethical considerations of using LLMs?"
)

for query in "${QUERIES[@]}"; do
    echo "Sending query: $query"
    curl -s -X POST http://localhost:8000/chat \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query\", \"session_id\": \"batch-test\"}" \
        | jq -r '.solution' | head -n 5
    echo "-----------------------------------"
    sleep 2
done
