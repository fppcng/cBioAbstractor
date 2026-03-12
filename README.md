# Synopsis — Local Setup & Deployment Guide

## Files in this bundle

| File | Purpose |
|------|---------|
| `query.py` | FastAPI backend — all API endpoints |
| `app.py` | Uvicorn entry point (`python app.py`) |
| `streamlit_app.py` | Streamlit 3-tab UI |
| `cbioportal_curator.py` | Core curation engine |
| `cbioportal_spec.py` | Embedded cBioPortal format schemas |
| `spec_fetcher.py` | Live GitHub spec fetcher (cached 1 hr) |
| `spec_match.py` | Spec-driven sheet classifier |
| `gene_alteration_analyst.py` | Alteration frequency + LLM code interpreter |
| `utils.py` | `load_chat_model()` router (OpenAI / Bedrock) |
| `pdf_ingest.py` | PDF → LangChain document chunks |
| `vector_store.py` | ChromaDB vector store helpers |
| `system_prompt_config.py` | Named system prompt registry |
| `config.py` | All environment / path constants |
| `cbio_detector.py` | Heuristic + LLM sheet type detector |
| `cbio_transformer.py` | LLM-powered format transformer |
| `few_shot_manager.py` | Save / list / delete few-shot examples |
| `file_parser.py` | Parse any uploaded file to DataFrame |
| `gene_extract.py` | Gene name extraction from text |
| `requirements.txt` | Python dependencies |

---

## 1. First-time setup

```bash
# Navigate to the project folder
cd ~/Downloads/synapse_cbio/

# Activate your Python 3.9 environment
pyenv local 3.9.18

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

Or create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

---

## 3. Run locally

**Terminal 1 — Backend:**
```bash
cd ~/Downloads/synapse_cbio/
python app.py
# Backend available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

**Terminal 2 — Frontend:**
```bash
cd ~/Downloads/synapse_cbio/
streamlit run streamlit_app.py
# UI available at http://localhost:8501
```

---

## 4. Verify the backend is working

```bash
curl http://localhost:8000/
# → {"status":"ok","service":"Synopsis backend","version":"3.1.0"}

curl http://localhost:8000/spec_status
# → {"source":"live","num_formats":13,...}
```

---

## 5. Deploy to Render (free tier)

1. Push all files to a GitHub repository
2. Create a new **Web Service** on render.com pointing to the repo
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn query:app --host 0.0.0.0 --port $PORT`
   - **Environment variable:** `OPENAI_API_KEY = sk-...`
4. After deploy, change `API_URL` in `streamlit_app.py` from `http://localhost:8000`
   to your Render URL (e.g. `https://gene-backend.onrender.com`)

---

## 6. Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `OpenAIError: api_key must be set` | `export OPENAI_API_KEY="sk-..."` |
| `Address already in use` | `lsof -ti:8000 \| xargs kill -9` |
| Port 8501 already in use | `lsof -ti:8501 \| xargs kill -9` |
| Streamlit can't reach backend | Make sure `python app.py` is running in another terminal |
