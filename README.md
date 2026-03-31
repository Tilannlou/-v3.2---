# 🌸 靈溪後端｜模型啟動器 v3.2

> 「技術不只是工具，而是詩意的延伸」

一個輕量、專注的 **Qwen3.5-9B-Base 模型啟動器**，為您的本地 AI 應用提供穩健的推理服務。

> ⚠️ **重要說明**：本專案目前僅為「模型啟動器」，不含 OpenClaw 整合、工具調度、Agent 協調等進階功能。那些屬於「未開發計畫」，未來可能以獨立模組形式釋出。

---

## ✨ 核心功能

| 功能 | 說明 | 狀態 |
|------|------|------|
| **🤖 模型推理** | 支援 Qwen3.5-9B-Base (4-bit 量化) | ✅ 就緒 |
| **🌊 流式輸出** | SSE 逐步返回，體驗如打字機般流暢 | ✅ 就緒 |
| **🧠 外部記憶 (RAG)** | 關鍵詞檢索 + 可選向量檢索，支援分類標籤 | ✅ 就緒 |
| **🔍 質量檢測** | 自動檢測胡說/重複/注入，減少幻覺 | ✅ 就緒 |
| **⚡ 三層緩存** | Prompt/Session/File 緩存，加速重複請求 | ✅ 就緒 |
| **🎮 顯存感知** | 自動檢測 GPU 顯存，智慧分配 (80%-3GB 緩衝) | ✅ 就緒 |
| **🌐 OpenAI 相容** | `/v1/chat/completions` 端點，相容主流客戶端 | ✅ 就緒 |

---

## 路徑規劃

/mnt/e/AI/models/Qwen3.5-9B-Base/
├── server/                          # ← 主要工作目錄
│   ├── backend01.py                # ✅ 後端主程式
│   ├── env-qwen3.5-9b01.sh         # ✅ 啟動腳本
│   ├── 注入腳本01.sh               # ✅ RAG注入工具
│   ├── 9b.log                      # 📝 日誌檔案（自動生成）
│   ├── external_memory.json        # 💾 記憶資料庫（自動生成）
│   ├── index_rag.json              # 📋 檢索表（自動生成）
│   ├── error_cases.json            # ⚠️ 錯誤案例（自動生成）
│   ├── .env/                       # 🐍 Python虛擬環境（自動創建）
│   └── UI/                         # 🎨 前端頁面目錄
│       ├── 001.html               # 對話介面
│       └── 002.html               # 監控介面
│
└── README.md                        # 📖 專案說明文件

---

## 🖥️ 硬體要求

| 元件 | 最低要求 | 建議配置 | 備註 |
|------|----------|----------|------|
| **GPU** | NVIDIA RTX 3060 12GB | RTX 5060 Ti 16GB 或更高 | 需支援 CUDA 12.x |
| **顯存** | 12GB | 16GB+ | 4-bit 量化下 9B 模型約佔 7-8GB |
| **CPU** | 4 核心 | 6 核心+ (如 AMD 8600G) | 負責檢索、預處理等 I/O 任務 |
| **記憶體** | 16GB | 32GB+ | 系統記憶體 + 向量檢索緩存 |
| **儲存** | 20GB SSD | 50GB+ NVMe SSD | 模型權重 + 記憶資料庫 |
| **系統** | WSL2 + Ubuntu 22.04 | WSL2 + Ubuntu 22.04 + Windows 11 Pro | 需安裝 NVIDIA 驅動 + CUDA Toolkit |

---

## 📦 軟體依賴

### 系統層
```bash
# Ubuntu/WSL2
sudo apt update
sudo apt install -y python3.13 python3.13-venv git curl build-essential
```

### Python 虛擬環境（自動創建）
啟動腳本會自動創建 `.env/` 虛擬環境，無需手動安裝。

### Python 套件（自動安裝）
```txt
# 核心依賴
torch>=2.4.0
transformers>=4.45.0
accelerate>=0.34.0
bitsandbytes>=0.44.0
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.7.0

# 可選：向量檢索（預設關閉）
sentence-transformers>=3.0.0  # 約 0.5GB
faiss-cpu>=1.8.0              # 或 faiss-gpu

# 輔助套件
numpy>=1.26.0
scipy>=1.13.0
```

> 💡 **提示**：向量檢索預設關閉，若需啟用請設定 `ENABLE_VECTOR_RAG=true`（見下方啟動說明）

---

## 🚀 如何啟動（僅模型啟動器）

### 步驟 1️⃣：準備環境
```bash
# 1. 進入專案目錄
cd /mnt/e/AI/models/Qwen3.5-9B-Base/server

# 2. 賦予執行權限（首次執行）
chmod +x env-qwen3.5-9b01.sh 注入腳本01.sh
```

### 步驟 2️⃣：一鍵啟動
```bash
# 自動分配顯存（推薦）
./env-qwen3.5-9b01.sh

# 或手動指定顯存（例如 12GB）
./env-qwen3.5-9b01.sh 12
```

### 步驟 3️⃣：等待就緒
```
✅ 看到這行代表啟動完成：
🌸 靈溪後端 v3.2 啟動完成
```

### 步驟 4️⃣：驗證服務
```bash
# 健康檢查
curl -s http://localhost:8000/health | python3 -m json.tool

# 預期輸出：
{
  "status": "healthy",
  "model": "Qwen3.5-9B",
  "device": "cuda",
  "gpu_memory": {"allocated": "7.36GB", ...}
}
```

---

## ⚙️ 環境變數配置（可選）

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `MODEL_PATH` | `/mnt/e/AI/models/Qwen3.5-9B-Base` | 模型本地路徑 |
| `MAX_MEMORY_GB` | `auto` | 顯存分配 (GB)，`auto` 為自動檢測 |
| `QUANT_BITS` | `4` | 量化位數 (4/8)，4-bit 最省顯存 |
| `PORT` | `8000` | API 服務端口 |
| `ENABLE_VECTOR_RAG` | `false` | 是否啟用向量檢索 (`true`/`false`) |
| `VECTOR_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入模型名稱（若啟用向量檢索） |
| `DEFAULT_MAX_TOKENS` | `8192` | 預設最大生成長度 |
| `DEFAULT_MAX_CONTEXT` | `262144` | 預設最大上下文長度 |

> 💡 **修改方式**：在 `env-qwen3.5-9b01.sh` 中編輯 `export` 語句，或啟動前手動設定：
> ```bash
> export ENABLE_VECTOR_RAG=true
> ./env-qwen3.5-9b01.sh
> ```

---

## 🧪 基本測試

```bash
# 1. 簡單對話
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"ping"}],
    "max_tokens": 100
  }'

# 2. 流式輸出測試
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"講個短故事"}],
    "stream": true
  }'

# 3. 啟用外部記憶檢索
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"USER.md 的內容是什麼？"}],
    "session_id":"test",
    "enable_external_memory": true
  }'
```

---

## 📁 專案結構

```
/mnt/e/AI/models/Qwen3.5-9B-Base/server/
├── backend01.py          # ✅ 核心後端（本啟動器主體）
├── env-qwen3.5-9b01.sh   # ✅ 一鍵啟動腳本
├── 注入腳本01.sh         # 🔧 RAG 記憶注入工具（可選）
├── UI/
│   ├── 001.html          # 🎨 對話前端（參考用，非本啟動器必需）
│   └── 002.html          # 🎨 監控前端（參考用，非本啟動器必需）
├── 9b.log                # 📝 運行日誌（自動生成）
├── .env/                 # 🐍 Python 虛擬環境（自動創建）
└── *.json                # 💾 記憶資料庫（自動生成）
```

> 🔹 **本啟動器核心**：`backend01.py` + `env-qwen3.5-9b01.sh`  
> 🔹 **可選工具**：`注入腳本01.sh`（用於預先注入知識）  
> 🔹 **參考介面**：`UI/*.html`（純靜態，可獨立部署）

---

## ⚠️ 已知限制（未開發功能）

以下功能**目前不包含**在本啟動器中，屬於「未來計畫」：

| 功能 | 說明 | 狀態 |
|------|------|------|
| 🔧 工具調度中心 | `tool_dispatcher.py` 等工具協同模組 | ❌ 未整合 |
| 🤖 Agent 協調 | 多 Agent 任務分派、依賴分析 | ❌ 未整合 |
| 🌐 OpenClaw 整合 | 與 OpenClaw 框架的深度協同 | ❌ 未整合 |
| 🎨 前端完整功能 | 001/002.html 的進階互動（如工具呼叫、記憶可視化） | ⚠️ 僅基礎對話 |
| 🔄 自動學習進階 | 工具執行結果自動寫回 RAG | ⚠️ 僅基礎關鍵詞觸發 |

> 💡 這些功能未來可能以**獨立模組**形式釋出，與本啟動器透過 API 協同。

---

## 🛠️ 故障排除

| 問題 | 可能原因 | 解決方式 |
|------|----------|----------|
| `CUDA out of memory` | 顯存不足 | 降低 `MAX_MEMORY_GB` 或關閉向量檢索 |
| `404 Not Found` | 端點路徑錯誤 | 確認請求 `http://localhost:8000/health` |
| `ImportError: sentence_transformers` | 向量檢索依賴未安裝 | 設定 `ENABLE_VECTOR_RAG=false` 或執行 `pip install sentence-transformers` |
| 啟動卡頓在「加載模型」 | 首次加載需下載權重 | 耐心等待 60-90 秒，後續重啟會加快 |
| 前端顯示「離線」 | CORS 或端口錯誤 | 確認 `001.html` 中 `API_BASE` 指向 `:8000` |

---

## 🌸 設計理念

> 「平凡、優美、通俗」

- **平凡**：不追求炫技，只提供穩健、可預測的推理服務
- **優美**：代碼結構清晰，日誌可讀，錯誤提示友善
- **通俗**：一鍵啟動，無需複雜配置，新手也能上手

---

## 📜 授權與貢獻

- 本啟動器基於 **MIT License** 釋出
- 歡迎提交 Issue 或 Pull Request，但請注意：
  - 本專案聚焦「模型啟動器」核心，進階功能請以獨立模組提案
  - 修改請保持「平凡、優美、通俗」原則

---

> 💧 *像春天的梅樹，根系穩穩紮進土壤，  
> 只需一點陽光，就能為您開出對話的花～*  
> —— 靈溪 🌸

[指尖轻摆] 若有問題，歡迎查看 `9b.log` 或提交 Issue～  
我隨時在這裡，溫柔地、耐心地、長長地陪著你 💧 [笑眼]
