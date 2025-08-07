import psycopg2
import csv
import time
from datetime import datetime, timedelta

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'testco_database',
    'user': 'postgres',
    'password': '170970022Za-'
}
TABLE_NAME = 'measurements'
INTERVAL_MINUTES = 5

# SQL statements
CREATE_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS timescaledb;
"""

CREATE_TABLE = f"""
-- Create table with lowercase column names for consistency
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    timestamp TIMESTAMPTZ NOT NULL,
    component TEXT,
    status TEXT,
    db REAL,
    topfreq1 REAL,
    topfreq2 REAL,
    topfreq3 REAL,
    tester_id TEXT
);
"""

# Use lowercase column identifier for hypertable creation
CREATE_HYPERTABLE = f"""
SELECT create_hypertable('{TABLE_NAME}', 'timestamp', if_not_exists => TRUE);
"""

INSERT_SQL = f"INSERT INTO {TABLE_NAME} (timestamp, component, status, db, topfreq1, topfreq2, topfreq3, tester_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
SELECT_OLD_ROWS = f"SELECT * FROM {TABLE_NAME} WHERE timestamp <= %s"
DELETE_OLD_ROWS = f"DELETE FROM {TABLE_NAME} WHERE timestamp <= %s"


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    """Initialize the database: enable extension, create table and convert to hypertable."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(CREATE_EXTENSION)
    cur.execute(CREATE_TABLE)
    cur.execute(CREATE_HYPERTABLE)
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized: TimescaleDB extension, table and hypertable ensured.")


def insert_row(row):
    """Insert a single row into the TimescaleDB hypertable."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(INSERT_SQL, row)
    conn.commit()
    cur.close()
    conn.close()


def rotate_and_export():
    """Export rows older than INTERVAL_MINUTES into a CSV then delete them."""
    cutoff = datetime.utcnow() - timedelta(minutes=INTERVAL_MINUTES)
    conn = get_connection()
    cur = conn.cursor()

    # Fetch rows older than cutoff
    cur.execute(SELECT_OLD_ROWS, (cutoff,))
    rows = cur.fetchall()

    if rows:
        timestamp_str = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        filename = f"export_{timestamp_str}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header matching lowercase names
            writer.writerow(['timestamp', 'component', 'status', 'db', 'topfreq1', 'topfreq2', 'topfreq3', 'tester_id'])
            writer.writerows(rows)
        print(f"Exported {len(rows)} rows to {filename}")

        # Delete old rows
        cur.execute(DELETE_OLD_ROWS, (cutoff,))
        deleted = cur.rowcount
        conn.commit()
        print(f"Deleted {deleted} rows older than {INTERVAL_MINUTES} minutes")
    else:
        print("No rows to export.")

    cur.close()
    conn.close()


if __name__ == '__main__':
    init_db()
    print(f"Starting rotation every {INTERVAL_MINUTES} minutes...")
    try:
        while True:
            rotate_and_export()
            time.sleep(INTERVAL_MINUTES * 60)
    except KeyboardInterrupt:
        print("Rotation process interrupted.")
