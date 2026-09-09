CSWAI Knowledge Base V2

Copy:
core/rag.py -> CSWAI/core/rag.py
core/llm.py -> CSWAI/core/llm.py
scripts/build_knowledge_base.py -> CSWAI/scripts/build_knowledge_base.py
scripts/inspect_knowledge_base.py -> CSWAI/scripts/inspect_knowledge_base.py

Recommended folders:
knowledge_base/01_core/fundamentals
knowledge_base/01_core/sketching
knowledge_base/01_core/features
knowledge_base/01_core/assemblies
knowledge_base/01_core/drawings
knowledge_base/02_procedures
knowledge_base/03_reasoning
knowledge_base/04_troubleshooting
knowledge_base/05_assessment
knowledge_base/06_instructor
knowledge_base/99_legacy

Build:
uv run python scripts\build_knowledge_base.py

Inspect:
uv run python scripts\inspect_knowledge_base.py

Run:
uv run streamlit run app.py
