TOPIC_KEYWORDS = {
    "Assemblies": ["mate", "assembly", "concentric", "coincident", "component"],
    "Sketching": ["sketch", "dimension", "constraint", "fully defined", "relation"],
    "Features": ["extrude", "revolve", "loft", "sweep", "fillet", "chamfer", "pattern"],
    "Drawings": ["drawing", "section view", "annotation", "orthographic", "detail view"],
    "Part Modeling": ["part", "mass properties", "material", "center of mass"],
}

def detect_topic(text: str) -> str:
    q = text.lower()
    scores = {
        topic: sum(1 for keyword in keywords if keyword in q)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"
