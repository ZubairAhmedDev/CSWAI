# Engineering Student Support System — UV Version

## Run

1. Install Ollama separately.
2. In PowerShell:

```powershell
ollama pull qwen3:4b
ollama pull nomic-embed-text
uv sync
uv run streamlit run app.py
```

## Knowledge base

Put PDF/TXT/MD files inside `knowledge_base/`.

Then open the app:
Student Portal -> Ask AI -> Build / Rebuild Knowledge Index
