import psycopg2
import psycopg2.pool
import time
import logging
import os
import sys
import csv
import re
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager
import yaml
from pathlib import Path

# Optional alert (kept but guarded). If unavailable, processing continues.
try:
    from Send_Mail_Saturn_Acoustic import send_alert  # noqa: F401
except Exception:  # pragma: no cover
    def send_alert(*args, **kwargs):  # type: ignore
        return

# =========================
# Logging
# =========================
def setup_logging():
    """Setup enhanced logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"csv_etl_{timestamp}.log"

    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_formatter)

    root_logger.addHandler(fh)
    root_logger.addHandler(ch)
    return log_file

log_file = setup_logging()
logger = logging.getLogger(__name__)


class ConsolePrinter:
    @staticmethod
    def print_header(title: str, char: str = "=", width: int = 80):
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")

    @staticmethod
    def print_section(title: str, char: str = "-", width: int = 60):
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")

    @staticmethod
    def print_info(label: str, value: str, indent: int = 2):
        print(f"{' ' * indent}{label}: {value}")

    @staticmethod
    def print_success(message: str):
        print(f"✅ {message}")

    @staticmethod
    def print_warning(message: str):
        print(f"⚠️  {message}")

    @staticmethod
    def print_error(message: str):
        print(f"❌ {message}")

    @staticmethod
    def print_progress(message: str):
        print(f"🔄 {message}")

    @staticmethod
    def print_stats(label: str, value: int, unit: str = ""):
        print(f"📊 {label}: {value:,} {unit}".strip())


console = ConsolePrinter()


# =========================
# Config
# =========================
class DatabaseConfig:
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self._log_config_summary()

    def _load_config(self) -> Dict[str, Any]:
        config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'dbname': os.getenv('DB_NAME', 'testco_database'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'min_connections': int(os.getenv('DB_MIN_CONNECTIONS', '1')),
            'max_connections': int(os.getenv('DB_MAX_CONNECTIONS', '10')),
            'table_name': os.getenv('DB_TABLE_NAME', 'measurements_v4'),
            'interval_minutes': int(os.getenv('INTERVAL_MINUTES', '5')),
            'export_directory': os.getenv('EXPORT_DIR', 'exports'),
            'retention_days': int(os.getenv('RETENTION_DAYS', '30')),
            'delete_after_export': os.getenv('DELETE_AFTER_EXPORT', 'true').lower() == 'true',
            'csv_source_directory': os.getenv('CSV_SOURCE_DIR', 'csv_files')
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if isinstance(file_config, dict):
                        config.update(file_config)
                    logger.info(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
                console.print_warning(f"Failed to load config file: {e}")
        else:
            logger.info("No config file found, using environment variables and defaults")
            console.print_info("Config", "Using environment variables and defaults")
        return config

    def _log_config_summary(self):
        console.print_section("Configuration Summary")
        console.print_info("Database", f"{self.config['user']}@{self.config['host']}:{self.config['port']}/{self.config['dbname']}")
        console.print_info("Table", self.config['table_name'])
        console.print_info("Export Directory", self.config['export_directory'])
        console.print_info("CSV Source Directory", self.config['csv_source_directory'])
        console.print_info("Interval", f"{self.config['interval_minutes']} minutes")
        console.print_info("Retention", f"{self.config['retention_days']} days")
        console.print_info("Delete After Export", "Yes" if self.config['delete_after_export'] else "No")
        console.print_info("Log File", str(log_file))

    def get_db_config(self) -> Dict[str, Any]:
        return {k: v for k, v in self.config.items() if k in ['host', 'port', 'dbname', 'user', 'password']}

    def get_pool_config(self) -> Dict[str, Any]:
        return {k: v for k, v in self.config.items() if k in ['min_connections', 'max_connections']}


# =========================
# DB Manager & Queries
# =========================
class DatabaseManager:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None
        self.table_name = config.config['table_name']
        self._init_connection_pool()

    def _init_connection_pool(self):
        try:
            console.print_progress("Initializing database connection pool...")
            db_config = self.config.get_db_config()
            pool_config = self.config.get_pool_config()
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                pool_config['min_connections'],
                pool_config['max_connections'],
                **db_config
            )
            logger.info("Connection pool initialized successfully")
            console.print_success(f"Connection pool initialized ({pool_config['min_connections']}-{pool_config['max_connections']} connections)")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            console.print_error(f"Failed to initialize connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            console.print_error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    def close(self):
        if self.pool:
            self.pool.closeall()
            logger.info("Connection pool closed")
            console.print_info("Status", "Connection pool closed")


class SQLQueries:
    """Only the required 6 columns: timestamp, db, f1, f2, f3, tester_id"""

    def __init__(self, table_name: str):
        self.table_name = table_name
        self._init_queries()

    def _init_queries(self):
        self.create_extension = "CREATE EXTENSION IF NOT EXISTS timescaledb;"
        self.create_table = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            timestamp TIMESTAMPTZ NOT NULL,
            db        REAL,
            f1        REAL,
            f2        REAL,
            f3        REAL,
            tester_id TEXT
        );
        """
        self.create_hypertable = f"""
        SELECT create_hypertable('{self.table_name}', 'timestamp', if_not_exists => TRUE);
        """
        self.create_index = f"""
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
        ON {self.table_name} (timestamp);
        """
        self.insert_sql = f"""
        INSERT INTO {self.table_name} (timestamp, db, f1, f2, f3, tester_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        self.select_old_rows = f"SELECT * FROM {self.table_name} WHERE timestamp <= %s"
        self.delete_old_rows = f"DELETE FROM {self.table_name} WHERE timestamp <= %s"
        # Minimal migration: ensure tester_id type is TEXT (fix if previously REAL)
        self.migration_queries = [
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS tester_id TEXT;",
            f"DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='{self.table_name}' AND column_name='tester_id' AND data_type!='text') THEN ALTER TABLE {self.table_name} ALTER COLUMN tester_id TYPE TEXT USING tester_id::text; END IF; END $$;",
        ]


# =========================
# CSV Processor
# =========================
class CSVProcessor:
    def __init__(self, db_manager: DatabaseManager, config: DatabaseConfig):
        self.db_manager = db_manager
        self.config = config
        self.queries = SQLQueries(config.config['table_name'])
        self.csv_source_dir = Path(config.config['csv_source_directory'])
        self.export_dir = Path(config.config['export_directory'])
        self.export_dir.mkdir(exist_ok=True)
        self.csv_source_dir.mkdir(exist_ok=True)
        logger.info(f"CSV source directory: {self.csv_source_dir}")
        logger.info(f"Export directory: {self.export_dir}")

    # --- tester_id from filename ---
    @staticmethod
    def _infer_tester_id(csv_file: Path) -> str:
        stem = csv_file.stem  # e.g. saturn_db-20250903_074157
        first = re.split(r"[_\-]", stem, maxsplit=1)[0]
        return first.strip().lower()

    def init_db(self):
        try:
            console.print_progress("Initializing database schema...")
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    console.print_progress("Creating TimescaleDB extension...")
                    cur.execute(self.queries.create_extension)
                    console.print_progress("Creating table structure...")
                    cur.execute(self.queries.create_table)
                    console.print_progress("Creating TimescaleDB hypertable...")
                    cur.execute(self.queries.create_hypertable)
                    console.print_progress("Creating performance indexes...")
                    cur.execute(self.queries.create_index)
                    console.print_progress("Running schema migration...")
                    for q in self.queries.migration_queries:
                        cur.execute(q)
                    conn.commit()
                    logger.info("Database initialized successfully")
                    console.print_success("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            console.print_error(f"Database initialization failed: {e}")
            raise

    def process_all_csv_files(self) -> Dict[str, int]:
        try:
            console.print_progress("Scanning for CSV files...")
            csv_files = list(self.csv_source_dir.glob("*.csv"))
            if not csv_files:
                console.print_info("Status", "No CSV files found to process")
                return {}
            console.print_success(f"Found {len(csv_files)} CSV files to process")

            results: Dict[str, int] = {}
            total_rows_processed = 0
            for csv_file in csv_files:
                try:
                    console.print_progress(f"Processing: {csv_file.name}")
                    rows_processed = self._process_single_csv(csv_file)
                    results[csv_file.name] = rows_processed
                    total_rows_processed += rows_processed
                    if self.config.config.get('delete_after_export', False):
                        self._delete_file(csv_file)
                        console.print_success(f"Deleted processed file: {csv_file.name}")
                except Exception as e:
                    logger.error(f"Failed to process {csv_file.name}: {e}")
                    console.print_error(f"Failed to process {csv_file.name}: {e}")
                    results[csv_file.name] = -1
            console.print_stats("Total Rows Processed", total_rows_processed)
            return results
        except Exception as e:
            logger.error(f"Failed to process CSV files: {e}")
            console.print_error(f"Failed to process CSV files: {e}")
            raise

    def _process_single_csv(self, csv_file: Path) -> int:
        try:
            rows: List[Tuple] = []
            tester_id = self._infer_tester_id(csv_file)
            with open(csv_file, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.reader(f)
                headers = next(reader)
                headers = [h.strip() for h in headers]
                for row_num, row in enumerate(reader, 2):
                    try:
                        parsed = self._parse_csv_row(row, headers, tester_id)
                        if parsed:
                            rows.append(parsed)
                    except Exception as e:
                        logger.warning(f"Failed to parse row {row_num} in {csv_file.name}: {e}")
                        continue
            if not rows:
                console.print_warning(f"No valid data found in {csv_file.name}")
                return 0
            imported_count = self._import_rows_to_db(rows)
            console.print_success(f"Imported {imported_count} rows from {csv_file.name}")
            return imported_count
        except Exception as e:
            logger.error(f"Failed to process CSV file {csv_file}: {e}")
            return 0

    @staticmethod
    def _find_idx(headers: List[str], *candidates: str) -> Optional[int]:
        lower = [h.lower() for h in headers]
        for name in candidates:
            name = name.lower()
            if name in lower:
                return lower.index(name)
        return None

    def _parse_csv_row(self, row: List[str], headers: List[str], tester_id: str) -> Optional[Tuple]:
        try:
            if len(row) < 2:
                return None

            # --- locate columns by header (robust) with fallbacks to legacy positions ---
            idx_ts = self._find_idx(headers, 'timestamp', 'time', 'ts')
            idx_db = self._find_idx(headers, 'db', 'dB', 'db_val')
            idx_f1 = self._find_idx(headers, 'f1', 'topfreq1', 'freq1')
            idx_f2 = self._find_idx(headers, 'f2', 'topfreq2', 'freq2')
            idx_f3 = self._find_idx(headers, 'f3', 'topfreq3', 'freq3')

            # Legacy fixed positions (from the previous schema):
            # 0:timestamp  5:db  6:topfreq1  7:topfreq2  8:topfreq3
            if idx_ts is None and len(row) > 0:
                idx_ts = 0
            if idx_db is None and len(row) > 5:
                idx_db = 5
            if idx_f1 is None and len(row) > 6:
                idx_f1 = 6
            if idx_f2 is None and len(row) > 7:
                idx_f2 = 7
            if idx_f3 is None and len(row) > 8:
                idx_f3 = 8

            if None in (idx_ts, idx_db, idx_f1, idx_f2, idx_f3):
                return None

            timestamp_str = row[idx_ts] if idx_ts is not None else None
            # parse timestamp with multiple formats
            timestamp = datetime.now()
            if timestamp_str:
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y/%m/%d %H:%M:%S']:
                    try:
                        timestamp = datetime.strptime(timestamp_str, fmt)
                        break
                    except ValueError:
                        continue

            def _to_float(x: Optional[str]) -> Optional[float]:
                try:
                    return float(x) if x not in (None, '') else None
                except Exception:
                    return None

            db_val = _to_float(row[idx_db])
            f1 = _to_float(row[idx_f1])
            f2 = _to_float(row[idx_f2])
            f3 = _to_float(row[idx_f3])

            # Optional alert if db threshold exceeded
            try:
                if db_val is not None and db_val > 88:
                    send_alert([timestamp.isoformat(sep=' '), tester_id, db_val])
            except Exception:
                logger.debug("send_alert failed; continuing")

            return (timestamp, db_val, f1, f2, f3, tester_id)
        except Exception as e:
            logger.warning(f"Failed to parse CSV row: {e}")
            return None

    def _import_rows_to_db(self, rows: List[Tuple]) -> int:
        if not rows:
            return 0
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(self.queries.insert_sql, rows)
                    conn.commit()
                    imported_count = cur.rowcount
                    logger.info(f"Imported {imported_count} rows to database")
                    return imported_count
        except Exception as e:
            logger.error(f"Failed to import rows to database: {e}")
            raise

    def _delete_file(self, filepath: Path):
        try:
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Successfully deleted file: {filepath}")
            else:
                logger.warning(f"File not found for deletion: {filepath}")
                console.print_warning(f"File not found for deletion: {filepath}")
        except Exception as e:
            logger.error(f"Failed to delete file {filepath}: {e}")
            console.print_error(f"Failed to delete file {filepath}: {e}")

    def cleanup_old_exports(self):
        retention_days = self.config.config['retention_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        try:
            console.print_progress(f"Cleaning up files older than {retention_days} days...")
            deleted_count = 0
            for file_path in self.export_dir.glob("*.csv"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    logger.info(f"Deleted old export file: {file_path}")
                    deleted_count += 1
            if deleted_count > 0:
                console.print_success(f"Cleanup completed: {deleted_count} old files deleted")
                console.print_stats("Files Deleted", deleted_count)
            else:
                console.print_info("Cleanup Status", "No old files to delete")
        except Exception as e:
            logger.warning(f"Failed to cleanup old exports: {e}")
            console.print_warning(f"Failed to cleanup old exports: {e}")


# =========================
# Service
# =========================
class DataRotationService:
    def __init__(self, config_file: str = "config.yaml"):
        self.config = DatabaseConfig(config_file)
        self.db_manager = DatabaseManager(self.config)
        self.csv_processor = CSVProcessor(self.db_manager, self.config)

    def start(self):
        try:
            console.print_header("CSV ETL Data Import Service", "=", 80)
            console.print_info("Service", "Starting CSV processing service")
            console.print_info("Version", "2.0.0-minimal")
            console.print_info("Start Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            self.csv_processor.init_db()
            interval_minutes = self.config.config['interval_minutes']

            logger.info(f"Starting CSV processing service every {interval_minutes} minutes...")
            logger.info(f"CSV source directory: {self.config.config['csv_source_directory']}")
            logger.info(f"Delete after export: {self.config.config.get('delete_after_export', False)}")

            console.print_section("Service Started")
            console.print_info("Interval", f"{interval_minutes} minutes")
            console.print_info("CSV Source Directory", self.config.config['csv_source_directory'])
            console.print_info("Delete After Processing", "Yes" if self.config.config.get('delete_after_export', False) else "No")

            iteration_count = 0
            while True:
                try:
                    iteration_count += 1
                    console.print_section(f"Iteration #{iteration_count}")
                    start_time = time.time()
                    _ = self.csv_processor.process_all_csv_files()
                    self.csv_processor.cleanup_old_exports()
                    elapsed_time = time.time() - start_time
                    console.print_info("Duration", f"{elapsed_time:.2f} seconds")
                    console.print_info("Next Run", f"in {interval_minutes} minutes")
                except Exception as e:
                    logger.error(f"Service iteration failed: {e}")
                    console.print_error(f"Service iteration failed: {e}")
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
            console.print_header("Service Interrupted", "!", 60)
            console.print_info("Status", "Service stopped by user")
        except Exception as e:
            logger.error(f"Service failed: {e}")
            console.print_error(f"Service failed: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            console.print_progress("Cleaning up resources...")
            self.db_manager.close()
            logger.info("Service cleanup completed")
            console.print_success("Service cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            console.print_error(f"Cleanup failed: {e}")


# =========================
# Entrypoint
# =========================
def main():
    try:
        service = DataRotationService()
        service.start()
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        console.print_error(f"Failed to start service: {e}")
        exit(1)


if __name__ == '__main__':
    main()
