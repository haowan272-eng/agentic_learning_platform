#!/bin/bash
# ============================================================
# RAG 模型下载脚本
# 用法:
#   bash scripts/download-models.sh              # 下载到 ~/models/
#   bash scripts/download-models.sh /opt/models   # 下载到指定目录
#   MODEL_ROOT=/opt/models docker compose up -d   # Docker 中使用
# ============================================================

set -e

MIRROR="${HF_MIRROR:-https://hf-mirror.com}"
MODEL_ROOT="${1:-${MODEL_ROOT:-$HOME/models}}"

echo "模型下载目录: $MODEL_ROOT"
echo "镜像源: $MIRROR"
echo ""

download_model() {
    local repo="$1"
    local dir="$MODEL_ROOT/$2"
    shift 2
    local files=("$@")

    if [ -d "$dir" ] && [ -f "$dir/${files[0]}" ]; then
        echo "[跳过] $repo (已存在)"
        return 0
    fi

    echo "[下载] $repo → $dir"
    mkdir -p "$dir/1_Pooling"

    for f in "${files[@]}"; do
        echo "  $f"
        # Handle subdirectory files (e.g., "1_Pooling/config.json")
        if [[ "$f" == */* ]]; then
            mkdir -p "$(dirname "$dir/$f")"
        fi
        curl -sL --connect-timeout 15 --retry 3 \
            -o "$dir/$f" \
            "$MIRROR/$repo/resolve/main/$f"
    done
    echo "  完成!"
}

# ========== bge-large-zh-v1.5 (Embedding, ~1.3GB) ==========
download_model \
    "BAAI/bge-large-zh-v1.5" \
    "bge-large-zh-v1.5" \
    "config.json" \
    "config_sentence_transformers.json" \
    "tokenizer.json" \
    "tokenizer_config.json" \
    "special_tokens_map.json" \
    "sentence_bert_config.json" \
    "modules.json" \
    "vocab.txt" \
    "pytorch_model.bin" \
    "1_Pooling/config.json"

# ========== bge-reranker-v2-m3 (Reranker, ~2.2GB) ==========
download_model \
    "BAAI/bge-reranker-v2-m3" \
    "bge-reranker-v2-m3" \
    "config.json" \
    "model.safetensors" \
    "sentencepiece.bpe.model" \
    "special_tokens_map.json" \
    "tokenizer.json" \
    "tokenizer_config.json"

echo ""
echo "============================================"
echo "  下载完成! 模型路径: $MODEL_ROOT"
echo "============================================"
echo ""
echo "本地运行:"
echo "  MODEL_ROOT=$MODEL_ROOT uv run python run.py"
echo ""
echo "Docker 运行:"
echo "  MODEL_ROOT=$MODEL_ROOT docker compose up -d"
