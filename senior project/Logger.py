import sqlite3
from datetime import datetime

class Logger:
    def __init__(self, db_path="logs/logs.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                file_name TEXT,
                label TEXT,
                confidence_score REAL
            )
        ''')
        self.conn.commit()

    def log(self, file_name, label, confidence_score):
        timestamp = datetime.now().isoformat()
        self.cursor.execute("INSERT INTO recognition_logs (timestamp, file_name, label, confidence_score) VALUES (?, ?, ?, ?)",
                            (timestamp, file_name, label, float(confidence_score)))
        self.conn.commit()
