#!/bin/bash
echo "=== GET hotwords ==="
curl -s http://127.0.0.1:8003/transcribe/hotwords | jq .

echo "\n=== POST hotwords ==="
curl -X POST -s http://127.0.0.1:8003/transcribe/hotwords \
     -H "Content-Type: application/json" \
     -d '{"words": ["测试", "航空", "民航"]}' | jq .

echo "\n=== GET hotwords (after update) ==="
curl -s http://127.0.0.1:8003/transcribe/hotwords | jq .
