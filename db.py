import sqlite3
from datetime import datetime

DB = "rag.db"

def get_conn():
    return sqlite3.connect(DB, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        question TEXT,
        answer TEXT,
        sources TEXT,
        evaluation TEXT,
        created_at TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id TEXT,
        rating INTEGER,
        comment TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_chat(conversation_id, q, a, s, e):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_history VALUES (NULL,?,?,?,?,?,?)",
        (conversation_id, q, a, str(s), e, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def save_feedback(conversation_id, rating, comment):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback VALUES (NULL,?,?,?,?)",
        (conversation_id, rating, comment, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
