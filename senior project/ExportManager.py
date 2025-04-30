import sqlite3
import pandas as pd

class ExportManager:
    def __init__(self, db_path="logs/logs.db"):
        self.conn = sqlite3.connect(db_path)

    def export_to_csv(self, out_path="logs/export.csv"):
        df = pd.read_sql_query("SELECT * FROM recognition_logs", self.conn)
        df.to_csv(out_path, index=False)
