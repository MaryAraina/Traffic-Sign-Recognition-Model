import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("logs/logs.db")  # Adjust path if needed

# Query the recognition_logs table
df = pd.read_sql_query("SELECT * FROM recognition_logs", conn)

# Export to CSV
df.to_csv("logs/recognition_logs.csv", index=False)

conn.close()
print("CSV exported successfully.")
