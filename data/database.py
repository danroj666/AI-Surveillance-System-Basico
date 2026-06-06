# ============================================
# AI Surveillance System - Base de datos
# ============================================

import sqlite3
import datetime
from config import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            person_id   INTEGER,
            description TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS captures (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            filename    TEXT NOT NULL,
            person_id   INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stats (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT NOT NULL,
            hour        INTEGER NOT NULL,
            count       INTEGER DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()
    print("[DB] Base de datos inicializada")

def log_event(event_type, person_id=None, description=""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO events (timestamp, event_type, person_id, description) VALUES (?, ?, ?, ?)",
        (timestamp, event_type, person_id, description)
    )
    conn.commit()
    conn.close()

def log_capture(filename, person_id=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO captures (timestamp, filename, person_id) VALUES (?, ?, ?)",
        (timestamp, filename, person_id)
    )
    conn.commit()
    conn.close()

def get_today_events():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    cursor.execute(
        "SELECT * FROM events WHERE timestamp LIKE ? ORDER BY timestamp DESC",
        (f"{today}%",)
    )
    events = cursor.fetchall()
    conn.close()
    return events

def get_hourly_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    cursor.execute('''
        SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
        FROM events
        WHERE timestamp LIKE ? AND event_type = 'detected'
        GROUP BY hour
        ORDER BY hour
    ''', (f"{today}%",))
    stats = cursor.fetchall()
    conn.close()
    return stats

def get_today_captures():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    cursor.execute(
        "SELECT * FROM captures WHERE timestamp LIKE ? ORDER BY timestamp DESC",
        (f"{today}%",)
    )
    captures = cursor.fetchall()
    conn.close()
    return captures