from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import re
from typing import Any

import numpy as np
import ollama
import pymupdf

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledge_base"
INDEX_PATH = BASE_DIR / "rag_index.json"

EMBED_MODEL = "nomic-embed-text"

# Tune with scripts/evaluate_rag.py rather than guessing forever.
MIN_DOMAIN_SEMANTIC = 0.34
MIN_DOMAIN_BM25_NORM = 0.10
OUT_OF_SCOPE_SEMANTIC = 0.56
OUT_OF_SCOPE_BM25_NORM = 0.18

STOPWORDS = {
    "a","an","and","are","as","at","be","been","being","by","can","could","did",
    "do","does","for","from","had","has","have","how","i","if","in","into","is",
    "it","its","may","of","on","or","should","that","the","their","then","there",
    "these","this","to","was","were","what","when","where","which","why","will",
    "with","would","you","your"
}

TOPICS = {
    "Sketching": [
        "sketch","sketching","dimension","dimensions","relation","relations",
        "constraint","constraints","fully defined","under defined","over defined",
        "line","circle","rectangle","arc","trim","offset"
    ],
    "Features": [
        "extrude","extruded","boss","base","cut","revolve","revolved","sweep",
        "swept","loft","lofted","fillet","chamfer","shell","pattern","mirror"
    ],
    "Part Modeling": [
        "part","part modeling","feature tree","design intent","material",
        "mass properties","center of mass","density","configuration"
    ],
    "Assemblies": [
        "assembly","assemblies","mate","mates","concentric","coincident",
        "parallel","perpendicular","distance mate","angle mate","component",
        "components","fixed","floating"
    ],
    "Drawings": [
        "drawing","drawings","drawing view","section view","detail view",
        "projected view","annotation","annotations","title block",
        "orthographic","dimensioning"
    ],
    "CSWA": [
        "cswa","certification","certified solidworks associate",
        "practice exam","sample exam"
    ],
}

SUBTOPICS = {
    "Extruded Boss/Base": ["extruded boss","boss/base","boss base","boss-base"],
    "Extruded Cut": ["extruded cut","cut-extrude","cut extrude"],
    "Revolve": ["revolve","revolved boss","revolved cut"],
    "Sweep": ["sweep","swept boss","swept cut"],
    "Loft": ["loft","lofted boss","lofted cut"],
    "Fillet": ["fillet"],
    "Chamfer": ["chamfer"],
    "Sketch Relations": ["sketch relation","geometric relation","relations"],
    "Fully Defined Sketch": ["fully defined","under defined","over defined"],
    "Concentric Mate": ["concentric mate","concentric"],
    "Coincident Mate": ["coincident mate","coincident"],
    "Distance Mate": ["distance mate"],
    "Assembly Mates": ["assembly mate","mates"],
    "Mass Properties": ["mass properties","center of mass"],
    "Section View": ["section view"],
    "Drawing Views": ["drawing view","projected view","orthographic"],
}

QUERY_SYNONYMS = {
    "extrude": ["extruded boss/base", "boss/base", "extruded feature"],
    "extruded": ["extrude", "boss/base"],
    "mate": ["assembly mate", "mates"],
    "concentric": ["concentric mate"],
    "coincident": ["coincident mate"],
    "fully defined": ["fully define", "defined sketch", "sketch relations", "dimensions"],
    "under defined": ["under-defined", "fully defined"],
    "over defined": ["over-defined", "conflicting relations"],
    "section view": ["section drawing view"],
    "mass": ["mass properties", "density", "material"],
    "revolve": ["revolved boss/base", "revolved feature"],
}

PROCEDURE_TERMS = [
    "click","select","choose","step","steps","create","insert","open","drag","press",
    "set","enter","apply","command","toolbar","menu","propertymanager"
]
TROUBLESHOOTING_TERMS = [
    "error","errors","fail","fails","failed","failure","problem","issue","cannot",
    "can't","unable","warning","conflict","invalid","repair","fix","troubleshoot",
    "over defined","under defined","dangling","rebuild","not working","why won't"
]
REASONING_TERMS = [
    "design intent","because","why","reason","dependency","depends","consider",
    "best practice","strategy","decision","efficient"
]
ASSESSMENT_TERMS = [
    "question","exam","quiz","assessment","practice","exercise","correct answer",
    "sample exam","cswa"
]
INSTRUCTOR_TERMS = [
    "instructor","teacher","teaching","student mistake","common mistake",
    "learning objective","classroom","pedagogy"
]

DOMAIN_TERMS = {
    "solidworks","cswa","cad","sketch","sketching","mate","assembly","assemblies",
    "extrude","extruded","revolve","sweep","loft","fillet","chamfer","feature",
    "drawing","section","dimension","constraint","relation","part","model",
    "modeling","mass","propertymanager","boss","cut","component","design",
    "geometry","configuration","pattern","mirror","shell","draft","view"
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9_-]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


def infer_from_map(text: str, mapping: dict[str, list[str]], default="General") -> str:
    q = text.lower()
    scores = {
        key: sum(q.count(keyword) for keyword in keywords)
        for key, keywords in mapping.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else default


def infer_topic(text: str) -> str:
    return infer_from_map(text, TOPICS)


def infer_subtopic(text: str) -> str:
    return infer_from_map(text, SUBTOPICS)


def infer_eke_layer(text: str) -> str:
    q = text.lower()
    scores = {
        "TROUBLESHOOTING": sum(q.count(x) for x in TROUBLESHOOTING_TERMS),
        "PROCEDURE": sum(q.count(x) for x in PROCEDURE_TERMS),
        "REASONING": sum(q.count(x) for x in REASONING_TERMS),
        "ASSESSMENT": sum(q.count(x) for x in ASSESSMENT_TERMS),
        "INSTRUCTOR": sum(q.count(x) for x in INSTRUCTOR_TERMS),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "CONCEPT"


def expand_query(query: str) -> str:
    q = query.lower()
    additions = []
    for phrase, synonyms in QUERY_SYNONYMS.items():
        if phrase in q:
            additions.extend(synonyms)
    if not additions:
        return query
    return query + " " + " ".join(additions)


def is_domain_query(query: str) -> bool:
    q = query.lower()
    if "solidworks" in q or "cswa" in q:
        return True
    if infer_topic(query) != "General":
        return True
    return bool(set(tokenize(query)) & DOMAIN_TERMS)


def _folder_layer(path: Path) -> str | None:
    parts = [p.lower() for p in path.relative_to(KB_DIR).parts[:-1]]
    mapping = {
        "02_procedures": "PROCEDURE",
        "03_reasoning": "REASONING",
        "04_troubleshooting": "TROUBLESHOOTING",
        "05_assessment": "ASSESSMENT",
        "06_instructor": "INSTRUCTOR",
    }
    for part in parts:
        if part in mapping:
            return mapping[part]
    return None


def _source_priority(path: Path) -> float:
    parts = [p.lower() for p in path.relative_to(KB_DIR).parts[:-1]]
    if "01_core" in parts:
        return 1.00
    if any(x in parts for x in (
        "02_procedures","03_reasoning","04_troubleshooting",
        "05_assessment","06_instructor"
    )):
        return 0.98
    return 0.92


def _section_hint(page_text: str) -> str:
    # Lightweight heading heuristic; safe if none is detected.
    raw_lines = [normalize(x) for x in re.split(r"[\r\n]+", page_text) if normalize(x)]
    for line in raw_lines[:12]:
        if 3 <= len(line) <= 90:
            words = line.split()
            if len(words) <= 10 and (
                line.isupper()
                or line.istitle()
                or re.match(r"^(lesson|chapter|exercise|tutorial|section)\b", line, re.I)
            ):
                return line
    return ""


def extract_units(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        doc = pymupdf.open(path)
        units = []
        for page_idx, page in enumerate(doc):
            raw = page.get_text()
            text = normalize(raw)
            if text:
                units.append({
                    "page": page_idx + 1,
                    "text": text,
                    "section_hint": _section_hint(raw),
                })
        doc.close()
        return units

    if suffix in {".txt", ".md"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = normalize(raw)
        return [{
            "page": 1,
            "text": text,
            "section_hint": _section_hint(raw),
        }] if text else []

    return []


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 220) -> list[str]:
    text = normalize(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        target_end = min(len(text), start + chunk_size)
        end = target_end

        if target_end < len(text):
            window_start = max(start + 650, target_end - 350)
            candidates = [
                text.rfind(". ", window_start, target_end),
                text.rfind("? ", window_start, target_end),
                text.rfind("! ", window_start, target_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def discover_documents() -> list[Path]:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    allowed = {".pdf", ".txt", ".md"}
    excluded_dirs = {"99_legacy", "legacy", "_downloads"}
    excluded_files = {"readme.txt", "readme.md"}

    docs = []
    for path in KB_DIR.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        if path.name.lower() in excluded_files:
            continue
        parts = {p.lower() for p in path.relative_to(KB_DIR).parts[:-1]}
        if parts & excluded_dirs:
            continue
        docs.append(path)
    return sorted(docs)


def build_index(batch_size: int = 8) -> int:
    docs = discover_documents()
    if not docs:
        raise RuntimeError("No approved PDF/TXT/MD documents found in knowledge_base.")

    records = []

    print("\nCSWAI Knowledge Base Builder V4")
    print(f"Documents: {len(docs)}")
    print(f"Embedding model: {EMBED_MODEL}\n")

    for dno, path in enumerate(docs, 1):
        rel = path.relative_to(KB_DIR).as_posix()
        print(f"[{dno}/{len(docs)}] {rel}")

        pending = []
        folder_layer = _folder_layer(path)
        priority = _source_priority(path)
        units = extract_units(path)

        for unit in units:
            for chunk_no, chunk in enumerate(chunk_text(unit["text"]), 1):
                auto_layer = infer_eke_layer(chunk)
                eke_layer = folder_layer or auto_layer

                pending.append({
                    "source": path.name,
                    "relative_path": rel,
                    "page": unit["page"],
                    "page_chunk": chunk_no,
                    "section_hint": unit.get("section_hint", ""),
                    "topic": infer_topic(chunk),
                    "subtopic": infer_subtopic(chunk),
                    "eke_layer": eke_layer,
                    "source_priority": priority,
                    "approved": True,
                    "text": chunk,
                    "tokens": tokenize(chunk),
                })

        print(f"  Pages: {len(units)}  Chunks: {len(pending)}")

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            print(f"  Embedding {start+1}-{min(start+len(batch),len(pending))}/{len(pending)}")

            embs = ollama.embed(
                model=EMBED_MODEL,
                input=[x["text"] for x in batch],
            )["embeddings"]

            for rec, emb in zip(batch, embs):
                rec["embedding"] = emb
                records.append(rec)

    INDEX_PATH.write_text(json.dumps(records), encoding="utf-8")
    print(f"\nDONE: {len(records)} chunks -> {INDEX_PATH}")
    return len(records)


def load_index() -> list[dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) or 1.0) * (np.linalg.norm(b) or 1.0)
    return float(np.dot(a, b) / denom)


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = documents
        self.k1 = k1
        self.b = b
        self.n = len(documents)
        self.lengths = [len(d) for d in documents]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 1.0
        self.freqs = [Counter(d) for d in documents]

        df = defaultdict(int)
        for doc in documents:
            for term in set(doc):
                df[term] += 1

        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.freqs, self.lengths):
            score = 0.0
            for term in query_tokens:
                if term not in tf:
                    continue
                freq = tf[term]
                idf = self.idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (freq * (self.k1 + 1)) / denom
            out.append(score)
        return out


def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(x - lo) / (hi - lo) for x in values]


def _metadata_bonus(rec: dict, qt: str, qs: str, ql: str) -> float:
    bonus = 0.0
    if qt != "General" and rec.get("topic") == qt:
        bonus += 0.055
    if qs != "General" and rec.get("subtopic") == qs:
        bonus += 0.075
    if ql != "CONCEPT" and rec.get("eke_layer") == ql:
        bonus += 0.045
    if rec.get("section_hint"):
        bonus += 0.005
    return bonus


def _mmr_select(candidates: list[dict], top_k: int, lambda_relevance: float = 0.78) -> list[dict]:
    if not candidates:
        return []

    selected = [candidates[0]]
    remaining = candidates[1:]

    while remaining and len(selected) < top_k:
        best_item = None
        best_value = -1e9

        for item in remaining:
            emb = np.array(item["embedding"], dtype=float)
            max_sim = max(
                cosine(emb, np.array(s["embedding"], dtype=float))
                for s in selected
            )
            value = lambda_relevance * item["score"] - (1 - lambda_relevance) * max_sim

            # Penalize duplicates from the same exact page.
            same_page = any(
                s["source"] == item["source"] and s["page"] == item["page"]
                for s in selected
            )
            if same_page:
                value -= 0.025

            if value > best_value:
                best_value = value
                best_item = item

        selected.append(best_item)
        remaining.remove(best_item)

    return selected


def _confidence(best_sem: float, best_bm25: float, best_score: float, count: int) -> str:
    if count == 0:
        return "Insufficient"
    if best_sem >= 0.60 and best_bm25 >= 0.30:
        return "High"
    if best_sem >= 0.47 and best_score >= 0.52:
        return "Medium"
    return "Low"


def retrieve_with_diagnostics(query: str, top_k: int = 5, candidate_k: int = 24) -> dict[str, Any]:
    records = load_index()
    qt, qs, ql = infer_topic(query), infer_subtopic(query), infer_eke_layer(query)

    empty = {
        "results": [],
        "confidence": "Insufficient",
        "best_semantic": 0.0,
        "best_bm25": 0.0,
        "best_score": 0.0,
        "in_scope": False,
        "query_topic": qt,
        "query_subtopic": qs,
        "query_layer": ql,
        "expanded_query": expand_query(query),
    }

    if not records:
        return empty

    expanded = expand_query(query)
    q_tokens = tokenize(expanded)

    q_embedding = np.array(
        ollama.embed(model=EMBED_MODEL, input=expanded)["embeddings"][0],
        dtype=float,
    )

    semantic_scores = [
        cosine(q_embedding, np.array(r["embedding"], dtype=float))
        for r in records
    ]

    docs_tokens = [
        r.get("tokens") or tokenize(r["text"])
        for r in records
    ]
    bm25 = BM25(docs_tokens)
    raw_bm25 = bm25.scores(q_tokens)
    norm_bm25 = _normalize_scores(raw_bm25)

    scored = []
    for idx, rec in enumerate(records):
        semantic = semantic_scores[idx]
        lexical = norm_bm25[idx]
        bonus = _metadata_bonus(rec, qt, qs, ql)
        priority = float(rec.get("source_priority", 1.0))

        # Semantic drives retrieval, lexical catches exact SOLIDWORKS terminology.
        hybrid = ((0.72 * semantic) + (0.23 * lexical) + bonus) * priority

        scored.append({
            **rec,
            "semantic_score": round(semantic, 4),
            "bm25_score": round(lexical, 4),
            "score": round(hybrid, 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    candidates = scored[:candidate_k]

    best_sem = max((x["semantic_score"] for x in candidates), default=0.0)
    best_bm = max((x["bm25_score"] for x in candidates), default=0.0)
    best_score = candidates[0]["score"] if candidates else 0.0

    explicit_domain = is_domain_query(query)

    if not explicit_domain:
        if best_sem < OUT_OF_SCOPE_SEMANTIC and best_bm < OUT_OF_SCOPE_BM25_NORM:
            return {
                **empty,
                "best_semantic": best_sem,
                "best_bm25": best_bm,
                "best_score": best_score,
                "expanded_query": expanded,
            }

    filtered = [
        x for x in candidates
        if (
            x["semantic_score"] >= MIN_DOMAIN_SEMANTIC
            or x["bm25_score"] >= MIN_DOMAIN_BM25_NORM
        )
    ]

    # Require at least one strong signal, even for explicit domain questions.
    if not filtered:
        return {
            **empty,
            "best_semantic": best_sem,
            "best_bm25": best_bm,
            "best_score": best_score,
            "expanded_query": expanded,
        }

    selected = _mmr_select(filtered, top_k=top_k)

    return {
        "results": selected,
        "confidence": _confidence(best_sem, best_bm, best_score, len(selected)),
        "best_semantic": round(best_sem, 4),
        "best_bm25": round(best_bm, 4),
        "best_score": round(best_score, 4),
        "in_scope": bool(selected),
        "query_topic": qt,
        "query_subtopic": qs,
        "query_layer": ql,
        "expanded_query": expanded,
    }


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return retrieve_with_diagnostics(query, top_k=top_k)["results"]


def knowledge_base_stats() -> dict[str, Any]:
    records = load_index()

    docs = set()
    pages = set()
    topics: dict[str, int] = {}
    layers: dict[str, int] = {}
    sources: dict[str, int] = {}

    for r in records:
        docs.add(r["relative_path"])
        pages.add((r["relative_path"], r["page"]))
        topics[r["topic"]] = topics.get(r["topic"], 0) + 1
        layers[r["eke_layer"]] = layers.get(r["eke_layer"], 0) + 1
        sources[r["source"]] = sources.get(r["source"], 0) + 1

    return {
        "documents": len(docs),
        "pages": len(pages),
        "chunks": len(records),
        "topics": topics,
        "eke_layers": layers,
        "sources": sources,
    }
