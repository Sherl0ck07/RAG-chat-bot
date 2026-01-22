# Veltris Doc-Bot - Setup & Deployment Guide

Complete guide to set up and run the Veltris Intelligent Doc-Bot using Docker.

---

## 📋 Prerequisites

Before starting, ensure you have:

- **Docker Desktop** installed and running
  - Windows: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Mac: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Linux: Install Docker Engine + Docker Compose
- **Python 3.11+** (for data ingestion only)
- **8GB RAM** minimum
- **Internet connection** (for downloading models and datasets)

---

## 🚀 Quick Start Overview

The setup follows this two-phase approach:

```
Phase 1: Data Preparation (Local)
  └─ Run ingestion script to process documents
  
Phase 2: Service Deployment (Docker)
  └─ Deploy backend + frontend containers
```

**Important**: Data ingestion **MUST** be completed before running Docker!

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/Sherl0ck07/RAG-chat-bot.git
cd RAG-chat-bot
```

---

## Step 2: Get Nebius API Key

1. Visit **https://tokenfactory.nebius.com/**
2. Sign up for a free account
3. Navigate to API Keys section
4. Generate a new API key
5. Copy the key (starts with `v1.CmMK...`)

---

## Step 3: Configure Environment

Create a `.env` file in the **root directory**:

```bash
# Windows
echo NEBIUS_API_KEY=v1.CmMKHHN0.... > .env

# Linux/Mac
echo "NEBIUS_API_KEY=v1.CmMKHHN0...." > .env
```

**Example `.env` file:**
```dotenv
NEBIUS_API_KEY=v1.CmMKHHN0YXRpY2tleS1lMDB4aDZhYzBoeWpoNHBxNG0SIXNlcnZpY2VhY2NvdW50LWUwMGNmNTF3cjAyams2eG1mNjILCNLOwssGENXm0XY6DAjQ0dqWBxDA-YGBAkACWgNlMDA...
```

⚠️ **Important**: Replace with your actual API key from Nebius!

---

## Step 4: Data Ingestion (One-Time Setup)

This step processes the HuggingFace Accelerate documentation and creates the vector database.

### 4.1 Navigate to Backend

```bash
cd backend
```

### 4.2 Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.3 Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- LangChain, ChromaDB, FastAPI
- Sentence Transformers
- HuggingFace datasets

**⏱️ Estimated time**: 2-3 minutes

### 4.4 Run Ingestion Script

```bash
python ingestion/ingest.py
```

**What happens:**
1. Downloads HuggingFace Accelerate documentation
2. Filters to 12 relevant documents
3. Chunks into 153 pieces (1000 tokens, 200 overlap)
4. Generates embeddings using sentence-transformers
5. Stores in ChromaDB vector database

**⏱️ Estimated time**: 
- **CPU**: ~5 minutes
- **GPU** (if available): ~1 minute

### 4.5 Expected Output

```bash
[INGEST_LOAD] Loading HuggingFace documentation dataset...
[INGEST_LOAD] Total documents in dataset: 2647
[INGEST_LOAD] Filtered documents: 12 matching 'accelerate'

[INGEST_CHUNK] Chunking documents...
[INGEST_CHUNK] Chunking complete: 153 total chunks from 12 documents

[INGEST_STORE] Storing chunks in ChromaDB...
Embedding + storing chunks: 100%|████████████████| 1/1 [00:28<00:00, 28.45s/it]
[INGEST_STORE] Successfully stored 153 chunks in ChromaDB

[INGEST_PIPELINE] ✅ Ingestion pipeline completed successfully
[INGEST_PIPELINE] Vector DB collection count: 153
```

### 4.6 Verify Vector Database Created

```bash
# Should see vector_db folder with data
ls -la vector_db/

# Expected output:
# vector_db/
#   ├── chroma.sqlite3
#   └── [collection-uuid]/
#       ├── data_level0.bin
#       ├── header.bin
#       ├── length.bin
#       └── link_lists.bin
```

### 4.7 Deactivate Virtual Environment

```bash
deactivate
cd ..
```

You're now back in the root directory, ready for Docker deployment!

---

## Step 5: Deploy with Docker

### 5.1 Verify Prerequisites

```bash
# Check Docker is running
docker --version
docker-compose --version

# Check .env file exists
cat .env  # Linux/Mac
type .env  # Windows
```

### 5.2 Build and Start Services

```bash
docker-compose up --build
```

**What happens:**
1. Builds backend Docker image (~3-4 minutes)
2. Builds frontend Docker image (~2-3 minutes)
3. Creates network bridge
4. Starts backend container (port 8000)
5. Starts frontend container (port 8501)

**⏱️ First-time build**: 5-7 minutes
**Subsequent starts**: 30-40 seconds

### 5.3 Expected Docker Output

**Backend startup:**
```bash
veltris-backend  | [CONFIG] Loading configuration...
veltris-backend  | [RAG_INIT_EMBEDDINGS] Loading embedding model...
veltris-backend  | [RAG_LOAD_VECTORSTORE] Loading vector store from /app/vector_db
veltris-backend  | [RAG_LOAD_VECTORSTORE] Vector store loaded successfully
veltris-backend  | [RAG_LOAD_VECTORSTORE] Document count: 153
veltris-backend  | INFO:     Uvicorn running on http://0.0.0.0:8000
veltris-backend  | INFO:     Application startup complete.
```

**Frontend startup:**
```bash
veltris-frontend |   You can now view your Streamlit app in your browser.
veltris-frontend |   URL: http://0.0.0.0:8501
```

---

## Step 6: Access the Application

### 6.1 Open Frontend UI

**Browser**: http://localhost:8501

**What you should see:**
- ✅ Sidebar shows "System Online"
- ✅ "Documents Loaded: 153"
- ✅ Chat interface ready
- ✅ Example questions displayed

### 6.2 Test Backend API (Optional)

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "vector_db_status": "connected",
  "total_documents": 153,
  "timestamp": "2026-01-22T..."
}
```

**API Documentation:**
Open http://localhost:8000/docs in browser to see interactive API docs.

---

## Step 7: Test the System

### 7.1 Ask a Question

In the Streamlit UI, type:
```
What is Accelerate?
```

**Expected behavior:**
- ⏳ Spinner shows "Thinking..."
- ⏱️ Response in ~15 seconds (first query)
- ✅ Detailed answer appears
- ✅ Source citations shown
- ✅ Confidence score displayed

### 7.2 Test Edge Case

Ask an out-of-scope question:
```
What is the capital of France?
```

**Expected response:**
```
I cannot find the answer in the provided documentation.
```

✅ This confirms hallucination prevention is working!

### 7.3 Expand Source Citations

Click on any source card to see:
- 📄 Source filename
- 📍 Section name
- 📊 Similarity score
- 📝 Document excerpt

---

## 🛑 Stopping the System

### Stop Containers (Keep Data)

```bash
# Press Ctrl+C in terminal
# Or in new terminal:
docker-compose stop
```

### Stop and Remove Containers

```bash
docker-compose down
```

### Remove Everything (Including Images)

```bash
docker-compose down --rmi all
```

**Note**: Vector database (`backend/vector_db/`) is preserved and does not need to be recreated.

---

## 🔄 Restarting the System

After initial setup, simply run:

```bash
docker-compose up
```

No need to rebuild or re-run ingestion! The vector database persists between restarts.

---

## 🐛 Troubleshooting

### Issue 1: "Cannot connect to backend"

**Symptom**: Frontend shows "System Offline" or queries fail

**Solutions:**
```bash
# Check backend is running
docker-compose ps

# View backend logs
docker-compose logs backend

# Restart services
docker-compose restart
```

**Common cause**: Backend still loading (takes 30-40s on startup)

---

### Issue 2: "No documents loaded" or "total_documents: 0"

**Symptom**: Health check shows 0 documents

**Cause**: Ingestion not completed or vector_db folder missing

**Solution:**
```bash
# Stop Docker
docker-compose down

# Re-run ingestion (Step 4)
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python ingestion/ingest.py

# Verify vector_db exists
ls -la vector_db/

# Restart Docker
cd ..
docker-compose up
```

---

### Issue 3: Ingestion fails - "ModuleNotFoundError"

**Symptom**: Import errors during `python ingestion/ingest.py`

**Solution:**
```bash
# Ensure virtual environment is activated
# You should see (venv) in your terminal prompt

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Issue 4: Ingestion fails - "Connection timeout" or "SSL error"

**Symptom**: Can't download HuggingFace dataset

**Solutions:**
- Check internet connection
- Disable VPN if active
- Try again (HuggingFace servers may be slow)
- Use dataset cache if available

---

### Issue 5: Docker build fails - "No space left on device"

**Symptom**: Docker build stops with disk space error

**Solution:**
```bash
# Clean up Docker
docker system prune -a

# Check available space
docker system df
```

---

### Issue 6: Slow first query (20+ seconds)

**Symptom**: First query takes very long

**Explanation**: This is **normal**! The LLM needs to:
1. Load model into memory (~5s)
2. Process query (~10-15s)

**Subsequent queries**: 10-15 seconds (much faster)

---

### Issue 7: "NEBIUS_API_KEY not found"

**Symptom**: Backend fails to start or API calls fail

**Solution:**
```bash
# Check .env file exists in ROOT directory
cat .env  # Linux/Mac
type .env  # Windows

# Verify format (no spaces around =)
NEBIUS_API_KEY=v1.CmMK...

# Restart Docker after fixing
docker-compose down
docker-compose up
```

---

### Issue 8: Port already in use

**Symptom**: "Port 8000 is already in use" or "Port 8501 is already in use"

**Solution:**
```bash
# Find and kill process using port
# Linux/Mac:
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9

# Windows (PowerShell):
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
Get-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess | Stop-Process
```

Or change ports in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change host port
  - "8502:8501"  # Change host port
```

---

### Issue 9: Frontend shows errors but backend is healthy

**Symptom**: Backend health check passes, but frontend can't connect

**Solution**: Check `frontend/utils/api_client.py` line 19:
```python
# Should read BACKEND_URL from environment
def __init__(self, base_url: str = None):
    if base_url is None:
        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    self.base_url = base_url
```

Rebuild frontend:
```bash
docker-compose up -d --build frontend
```

---

## 📊 System Resources

**Typical Resource Usage:**

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| Backend (idle) | 5-10% | 800MB | - |
| Backend (query) | 40-60% | 1.2GB | - |
| Frontend | 2-5% | 200MB | - |
| Vector DB | - | - | 50MB |
| Docker overhead | - | 500MB | - |
| **Total** | - | **~2.5GB** | **~3GB** |

---

## 🎓 Next Steps

After successful setup:

1. **Read the README**: Understand architecture and design decisions
2. **Try Example Queries**: Test different question types
3. **Explore API Docs**: http://localhost:8000/docs
4. **Watch Demo Video**: [Link in README]
5. **Check Source Code**: Understand RAG implementation

---

## 📚 Additional Resources

- **Project README**: Architecture Decision Record, features, API docs
- **Backend API Docs**: http://localhost:8000/docs (when running)
- **System Diagram**: `veltris_architecture.png`
- **GitHub Repo**: https://github.com/Sherl0ck07/RAG-chat-bot

---

## ⚡ Quick Reference Commands

```bash
# Start system
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop system
docker-compose down

# Rebuild after code changes
docker-compose up --build

# Check status
docker-compose ps

# Check backend health
curl http://localhost:8000/health
```

---

## 💡 Tips for Best Experience

1. **First query is slow**: Wait 15-20s for model loading
2. **Subsequent queries faster**: ~10-15s average
3. **Use specific questions**: Better results than generic queries
4. **Check confidence scores**: Low confidence means uncertain answer
5. **Expand source citations**: See where answers come from
6. **Try edge cases**: Test "What is Python?" to see fallback

---

**Setup complete!** 🎉 You now have a production-ready RAG system running locally!

For issues or questions, check the Troubleshooting section above or open an issue on GitHub.
