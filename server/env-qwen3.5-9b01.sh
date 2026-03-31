#!/bin/bash
# 靈溪後端 v3.2 - RTX 5060 Ti 優化啟動腳本 (附前端服務)
# 完整版 - 保留所有核心功能
cd /mnt/e/AI/models/Qwen3.5-9B-Base/server
set -e
chmod +x env-qwen3.5-9b01.sh
chmod +x 注入腳本01.sh
clear
cd /mnt/e/AI/models/Qwen3.5-9B-Base/server

echo "═══════════════════════════════════════"
echo "🌸 靈溪後端 v3.2 - RTX 5060 Ti 完整版"
echo "═══════════════════════════════════════"

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 1. 檢測 GPU
echo -e "${BLUE}🔍 檢測 GPU 狀態...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ 未找到 nvidia-smi${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
USED_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
FREE_MEM=$((TOTAL_MEM - USED_MEM))

echo -e "${GREEN}📊 GPU 資訊：${NC}"
echo "   型號：$GPU_NAME"
echo "   總顯存：${TOTAL_MEM}MB ($(echo "scale=1; $TOTAL_MEM/1024" | bc)GB)"
echo "   可用顯存：${FREE_MEM}MB ($(echo "scale=1; $FREE_MEM/1024" | bc)GB)"

# 2. 智慧顯存分配
SAFE_MEM_MB=$((FREE_MEM * 85 / 100 - 3072))
if [ $SAFE_MEM_MB -lt 8192 ]; then SAFE_MEM_MB=8192; fi
if [ $SAFE_MEM_MB -gt 14336 ]; then SAFE_MEM_MB=14336; fi
MAX_MEMORY_GB=$((SAFE_MEM_MB / 1024))

if [ -n "$1" ]; then 
    MAX_MEMORY_GB=$1
    echo -e "${YELLOW}🔧 使用手動顯存：${MAX_MEMORY_GB}GB${NC}"
else
    echo -e "${GREEN}💡 自動分配顯存：${MAX_MEMORY_GB}GB${NC}"
fi

# 3. 檢查後端端口
echo -e "${BLUE}🔍 檢查後端端口 8000...${NC}"
EXISTING_PID=$(lsof -t -i:8000 2>/dev/null || true)
if [ -n "$EXISTING_PID" ]; then
    echo -e "${YELLOW}⚠️  端口 8000 被佔用 (PID: $EXISTING_PID)${NC}"
    read -p "是否終止該進程？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill -9 $EXISTING_PID
        sleep 2
        echo -e "${GREEN}✅ 進程已終止${NC}"
    else
        echo -e "${RED}❌ 請手動釋放端口後重試${NC}"
        exit 1
    fi
fi

# 3b. 檢查前端端口 3000
echo -e "${BLUE}🔍 檢查前端端口 3000...${NC}"
FRONTEND_PID=$(lsof -t -i:3000 2>/dev/null || true)
if [ -n "$FRONTEND_PID" ]; then
    echo -e "${YELLOW}⚠️  端口 3000 被佔用 (PID: $FRONTEND_PID)${NC}"
    read -p "是否終止該進程？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kill -9 $FRONTEND_PID
        sleep 1
        echo -e "${GREEN}✅ 進程已終止${NC}"
    else
        echo -e "${RED}❌ 請手動釋放端口後重試${NC}"
        exit 1
    fi
fi

# 4. 環境變數
echo -e "${BLUE}📝 配置環境變數...${NC}"

MODEL_PATH="/mnt/e/AI/models/Qwen3.5-9B-Base"
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}❌ 模型路徑不存在：$MODEL_PATH${NC}"
    exit 1
fi

export MODEL_PATH="$MODEL_PATH"
export MAX_MEMORY_GB="$MAX_MEMORY_GB"
export QUANT_BITS="4"
export PORT="8000"
export DEFAULT_MAX_TOKENS="8192"
export DEFAULT_MAX_CONTEXT="262144"
export DEFAULT_ENABLE_THINKING="false"

# PyTorch 優化
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export OMP_NUM_THREADS="8"
export TOKENIZERS_PARALLELISM="false"
# 向量檢索開關（true / false）
export ENABLE_VECTOR_RAG="true"
# 選擇嵌入模型（支援 HuggingFace 模型名稱）
export VECTOR_MODEL="BAAI/bge-small-zh-v1.5"
# 可選模型：
# - "BAAI/bge-large-zh-v1.5"  （更大，精度更高，需更多記憶體）
# - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" （輕量，多語言）

echo -e "${GREEN}✅ 環境配置完成：${NC}"
echo "   MODEL_PATH: $MODEL_PATH"
echo "   MAX_MEMORY_GB: $MAX_MEMORY_GB GB"
echo "   QUANT_BITS: $QUANT_BITS"
echo "   MAX_CONTEXT: $DEFAULT_MAX_CONTEXT tokens"

# 5. Python 環境
echo -e "${BLUE}🐍 檢查 Python 環境...${NC}"

VENV_PATH="/mnt/e/AI/models/Qwen3.5-9B-Base/server/.env"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}⚠️  創建虛擬環境...${NC}"
    python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

# 檢查套件
echo "檢查必要套件..."
python -c "import torch; import transformers; import fastapi" 2>/dev/null || {
    echo -e "${YELLOW}⚠️  安裝依賴...${NC}"
    pip install --upgrade pip -q
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
    pip install transformers accelerate bitsandbytes -q
    pip install fastapi uvicorn pydantic -q
    pip install numpy scipy -q
    echo -e "${GREEN}✅ 依賴安裝完成${NC}"
}

CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" == "True" ]; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    echo -e "${GREEN}✅ CUDA $CUDA_VERSION 可用${NC}"
else
    echo -e "${RED}❌ PyTorch CUDA 不可用${NC}"
    exit 1
fi

# 6. 啟動前端服務器（後台）
echo -e "${BLUE}🌐 啟動前端服務器 (端口 3000)...${NC}"
cd /mnt/e/AI/models/Qwen3.5-9B-Base/server/UI
python3 -m http.server 3000 > /dev/null 2>&1 &
FRONTEND_PID=$!
cd - > /dev/null
echo -e "${GREEN}✅ 前端服務器已啟動，PID: $FRONTEND_PID${NC}"
echo -e "${GREEN}🌐 前端地址：http://localhost:3000/001.html (對話) 或 /002.html (監控)${NC}"

# 設置清理函數（腳本退出時終止前端）
cleanup() {
    echo -e "${YELLOW}🛑 正在關閉前端服務器...${NC}"
    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
}
trap cleanup EXIT

# 7. 啟動後端服務（前台）
echo -e "${CYAN}"
echo "═══════════════════════════════════════"
echo "✨ 靈溪後端 v3.2 啟動中..."
echo "═══════════════════════════════════════"
echo -e "${NC}"

echo "📝 日誌文件：9b.log"
echo "🌐 API 地址：http://localhost:8000"
echo "🔍 健康檢查：http://localhost:8000/health"
echo "🔧 模型列表：http://localhost:8000/v1/models"
echo ""

> 9b.log

python backend01.py 2>&1 | tee -a 9b.log
# 自動驗證（後台運行，不阻塞主進程）
( sleep 5 && ./verify.sh ) &
# 注意：後端結束後會自動觸發 cleanup 關閉前端