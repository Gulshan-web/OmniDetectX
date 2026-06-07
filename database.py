import sqlite3

DB_NAME = "omnidetect.db"


def create_database():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scan_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        scan_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        objects TEXT,
        authenticity TEXT,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()


def save_scan(
    image_name,
    objects,
    authenticity,
    confidence
):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO scan_history
    (
        image_name,
        objects,
        authenticity,
        confidence
    )
    VALUES (?, ?, ?, ?)
    """,
    (
        image_name,
        objects,
        authenticity,
        confidence
    ))

    conn.commit()
    conn.close()