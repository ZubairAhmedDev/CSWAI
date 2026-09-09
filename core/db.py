import sqlite3
from pathlib import Path
import random

DB_PATH = Path(__file__).resolve().parent.parent / "student_support.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS students(
        student_id TEXT PRIMARY KEY,
        name TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS interactions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        question TEXT NOT NULL,
        topic TEXT NOT NULL,
        answer TEXT,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS assessments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        topic TEXT NOT NULL,
        assessment_type TEXT NOT NULL,
        score REAL NOT NULL,
        attempts INTEGER DEFAULT 1,
        timestamp TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def ensure_demo_students():
    init_db()
    conn = get_connection()
    rows = [(f"S{i:03d}", f"Student {i:02d}") for i in range(1, 31)]
    conn.executemany(
        "INSERT OR IGNORE INTO students(student_id, name) VALUES (?, ?)", rows
    )
    conn.commit()
    conn.close()

def seed_demo_performance():
    ensure_demo_students()
    conn = get_connection()
    if conn.execute("SELECT COUNT(*) AS n FROM assessments").fetchone()["n"] > 0:
        conn.close()
        return

    random.seed(7)
    topics = ["Sketching", "Features", "Part Modeling", "Assemblies", "Drawings"]
    for i in range(1, 31):
        student_id = f"S{i:03d}"
        if i <= 5:
            base = random.randint(45, 58)
        elif i <= 13:
            base = random.randint(61, 72)
        else:
            base = random.randint(76, 90)

        for topic in topics:
            adj = random.randint(-8, 8)
            if topic == "Assemblies":
                adj -= random.randint(5, 12)
            score = max(25, min(100, base + adj))
            attempts = 1 if score >= 75 else random.randint(2, 4)

            for assessment_type, delta in [
                ("quiz", 0),
                ("homework", random.randint(-4, 8)),
                ("practice", random.randint(-8, 6)),
            ]:
                s = max(20, min(100, score + delta))
                conn.execute("""
                    INSERT INTO assessments
                    (student_id, topic, assessment_type, score, attempts)
                    VALUES (?, ?, ?, ?, ?)
                """, (student_id, topic, assessment_type, s, attempts))
    conn.commit()
    conn.close()

def log_interaction(student_id, question, topic, answer=""):
    conn = get_connection()
    conn.execute("""
        INSERT INTO interactions(student_id, question, topic, answer)
        VALUES (?, ?, ?, ?)
    """, (student_id, question, topic, answer))
    conn.commit()
    conn.close()

def save_assessment(student_id, topic, assessment_type, score, attempts=1):
    conn = get_connection()
    conn.execute("""
        INSERT INTO assessments(student_id, topic, assessment_type, score, attempts)
        VALUES (?, ?, ?, ?, ?)
    """, (student_id, topic, assessment_type, float(score), int(attempts)))
    conn.commit()
    conn.close()
