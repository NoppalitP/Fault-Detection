import psycopg2
import psycopg2.pool
import time
import logging
import os
import sys
import csv
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager
import yaml
from pathlib import Path
from Send_Mail_Saturn_Acoustic import Alert
# Configure logging with improved formatting
def setup_logging():
    """Setup enhanced logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"csv_etl_{timestamp}.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler with simplified formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

# Setup logging
log_file = setup_logging()
logger = logging.getLogger(__name__)

# Custom stdout printer for user-friendly output
class ConsolePrinter:
    """Handles user-friendly console output."""
    
    @staticmethod
    def print_header(title: str, char: str = "=", width: int = 80):
        """Print a formatted header."""
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")
    
    @staticmethod
    def print_section(title: str, char: str = "-", width: int = 60):
        """Print a formatted section header."""
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")
    
    @staticmethod
    def print_info(label: str, value: str, indent: int = 2):
        """Print formatted info line."""
        print(f"{' ' * indent}{label}: {value}")
    
    @staticmethod
    def print_success(message: str):
        """Print success message."""
        print(f"✅ {message}")
    
    @staticmethod
    def print_warning(message: str):
        """Print warning message."""
        print(f"⚠️  {message}")
    
    @staticmethod
    def print_error(message: str):
        """Print error message."""
        print(f"❌ {message}")
    
    @staticmethod
    def print_progress(message: str):
        """Print progress message."""
        print(f"🔄 {message}")
    
    @staticmethod
    def print_stats(label: str, value: int, unit: str = ""):
        """Print formatted statistics."""
        print(f"📊 {label}: {value:,} {unit}".strip())

# Initialize console printer
console = ConsolePrinter()

class DatabaseConfig:
    """Database configuration class with environment variable support."""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self._log_config_summary()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment variables."""
        config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'dbname': os.getenv('DB_NAME', 'testco_database'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'min_connections': int(os.getenv('DB_MIN_CONNECTIONS', '1')),
            'max_connections': int(os.getenv('DB_MAX_CONNECTIONS', '10')),
            'table_name': os.getenv('DB_TABLE_NAME', 'measurements_v2'),
            'interval_minutes': int(os.getenv('INTERVAL_MINUTES', '5')),
            'export_directory': os.getenv('EXPORT_DIR', 'exports'),
            'retention_days': int(os.getenv('RETENTION_DAYS', '30')),
            'delete_after_export': os.getenv('DELETE_AFTER_EXPORT', 'true').lower() == 'true',
            'csv_source_directory': os.getenv('CSV_SOURCE_DIR', 'csv_files')
        }
        
        # Try to load from config file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
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
        """Log configuration summary."""
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
        """Get database connection parameters."""
        return {k: v for k, v in self.config.items() 
                if k in ['host', 'port', 'dbname', 'user', 'password']}
    
    def get_pool_config(self) -> Dict[str, Any]:
        """Get connection pool parameters."""
        return {k: v for k, v in self.config.items() 
                if k in ['min_connections', 'max_connections']}

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None
        self.table_name = config.config['table_name']
        self._init_connection_pool()
    
    def _init_connection_pool(self):
        """Initialize the connection pool."""
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
        """Get a database connection from the pool."""
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
        """Close the connection pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("Connection pool closed")
            console.print_info("Status", "Connection pool closed")

class SQLQueries:
    """SQL query definitions."""
    
    def __init__(self, table_name: str):
        self.table_name = table_name
        self._init_queries()
    
    def _init_queries(self):
        """Initialize SQL queries."""
        self.create_extension = "CREATE EXTENSION IF NOT EXISTS timescaledb;"
        
        self.create_table = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            timestamp TIMESTAMPTZ NOT NULL,
            component TEXT,
            component_proba REAL,
            status TEXT,
            status_proba REAL,
            db REAL,
            topfreq1 REAL,
            topfreq2 REAL,
            topfreq3 REAL,
            topfreq4 REAL,
            topfreq5 REAL,
            tester_id TEXT,
            mfcc_1 REAL,
            mfcc_2 REAL,
            mfcc_3 REAL,
            mfcc_4 REAL,
            mfcc_5 REAL,
            mfcc_6 REAL,
            mfcc_7 REAL,
            mfcc_8 REAL,
            mfcc_9 REAL,
            mfcc_10 REAL,
            mfcc_11 REAL,
            mfcc_12 REAL,
            mfcc_13 REAL
        );
        """
        
        self.create_hypertable = f"""
        SELECT create_hypertable('{self.table_name}', 'timestamp', if_not_exists => TRUE);
        """
        
        self.insert_sql = f"""
        INSERT INTO {self.table_name} 
        (timestamp, component, component_proba, status, status_proba, db, topfreq1, topfreq2, topfreq3, topfreq4, topfreq5,tester_id, 
         mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7, mfcc_8, mfcc_9, mfcc_10, mfcc_11, mfcc_12, mfcc_13) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        self.select_old_rows = f"SELECT * FROM {self.table_name} WHERE timestamp <= %s"
        self.delete_old_rows = f"DELETE FROM {self.table_name} WHERE timestamp <= %s"
        
        # Add index for better performance
        self.create_index = f"""
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
        ON {self.table_name} (timestamp);
        """
        
        # Migration queries to add new columns to existing tables
        self.migration_queries = [
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS component_proba REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS status_proba REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS db REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS topfreq1 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS topfreq2 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS topfreq3 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS topfreq4 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS topfreq5 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS tester_id REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_1 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_2 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_3 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_4 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_5 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_6 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_7 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_8 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_9 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_10 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_11 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_12 REAL;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS mfcc_13 REAL;"
        ]

class CSVProcessor:
    """Handles CSV file processing and database operations."""
    
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
    
    def init_db(self):
        """Initialize the database schema."""
        try:
            console.print_progress("Initializing database schema...")
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create extension
                    console.print_progress("Creating TimescaleDB extension...")
                    cur.execute(self.queries.create_extension)
                    
                    # Create table
                    console.print_progress("Creating table structure...")
                    cur.execute(self.queries.create_table)
                    
                    # Create hypertable
                    console.print_progress("Creating TimescaleDB hypertable...")
                    cur.execute(self.queries.create_hypertable)
                    
                    # Create index
                    console.print_progress("Creating performance indexes...")
                    cur.execute(self.queries.create_index)
                    
                    # Run migration queries to add new columns if they don't exist
                    console.print_progress("Running schema migration...")
                    for migration_query in self.queries.migration_queries:
                        cur.execute(migration_query)
                    
                    conn.commit()
                    logger.info("Database initialized successfully")
                    console.print_success("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            console.print_error(f"Database initialization failed: {e}")
            raise
    
    def process_all_csv_files(self) -> Dict[str, int]:
        """Process all CSV files in the source directory and import them to database."""
        try:
            console.print_progress("Scanning for CSV files...")
            
            # Find all CSV files
            csv_files = list(self.csv_source_dir.glob("*.csv"))
            
            if not csv_files:
                console.print_info("Status", "No CSV files found to process")
                return {}
            
            console.print_success(f"Found {len(csv_files)} CSV files to process")
            
            results = {}
            total_rows_processed = 0
            
            for csv_file in csv_files:
                try:
                    console.print_progress(f"Processing: {csv_file.name}")
                    rows_processed = self._process_single_csv(csv_file)
                    results[csv_file.name] = rows_processed
                    total_rows_processed += rows_processed
                    
                    # Delete the CSV file after processing if configured
                    if self.config.config.get('delete_after_export', False):
                        self._delete_file(csv_file)
                        console.print_success(f"Deleted processed file: {csv_file.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {csv_file.name}: {e}")
                    console.print_error(f"Failed to process {csv_file.name}: {e}")
                    results[csv_file.name] = -1  # Error indicator
            
            console.print_stats("Total Rows Processed", total_rows_processed)
            return results
            
        except Exception as e:
            logger.error(f"Failed to process CSV files: {e}")
            console.print_error(f"Failed to process CSV files: {e}")
            raise
    
    def _process_single_csv(self, csv_file: Path) -> int:
        """Process a single CSV file and import data to database."""
        try:
            rows = []
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Skip header row
                
                for row_num, row in enumerate(reader, 2):  # Start from 2 (after header)
                    try:
                        # Parse the row data
                        parsed_row = self._parse_csv_row(row, headers)
                
                        if parsed_row:
                            db_val = parsed_row[5]  # ตำแหน่ง db_val หลัง parse
                            if db_val is not None and db_val > 90:
                                Alert([parsed_row[0], parsed_row[1], parsed_row[3], db_val, parsed_row[11]])
                            rows.append(parsed_row)
                    except Exception as e:
                        logger.warning(f"Failed to parse row {row_num} in {csv_file.name}: {e}")
                        continue
            
            if not rows:
                console.print_warning(f"No valid data found in {csv_file.name}")
                return 0
            
            # Import to database
            imported_count = self._import_rows_to_db(rows)
            console.print_success(f"Imported {imported_count} rows from {csv_file.name}")
            
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to process CSV file {csv_file}: {e}")
            raise
    
    def _parse_csv_row(self, row: List[str], headers: List[str]) -> Optional[Tuple]:
        """Parse a CSV row into the expected format."""
        try:
            if len(row) < len(headers):
                return None
            
            # Map CSV columns to database columns
            # Expected CSV format: timestamp, component, status, db, topfreq1, topfreq2, topfreq3, topfreq4, topfreq5,tester_id
            # component_proba, status_proba, mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7, mfcc_8, mfcc_9, 
            # mfcc_10, mfcc_11, mfcc_12, mfcc_13
            # Total: 25 columns (0-24)
            timestamp_str = row[0] if len(row) > 0 else None
            component = row[1] if len(row) > 1 else None
            component_proba = float(row[2]) if len(row) > 2 and row[2] else None
            status = row[3] if len(row) > 3 else None
            status_proba = float(row[4]) if len(row) > 4 and row[4] else None
            db_val = float(row[5]) if len(row) > 5 and row[5] else None
            topfreq1 = float(row[6]) if len(row) > 6 and row[6] else None
            topfreq2 = float(row[7]) if len(row) > 7 and row[7] else None
            topfreq3 = float(row[8]) if len(row) > 8 and row[8] else None
            topfreq4 = float(row[9]) if len(row) > 9 and row[9] else None
            topfreq5 = float(row[10]) if len(row) > 10 and row[10] else None
            tester_id = row[11] if len(row) > 11 and row[11] else None
            # Parse MFCC features (13 columns)
            mfcc_features = []
            for i in range(13):
                mfcc_val = float(row[12 + i]) if len(row) > 12 + i and row[12 + i] else None
                mfcc_features.append(mfcc_val)
            
            
            
            # Parse timestamp
            try:
                if timestamp_str:
                    # Try different timestamp formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                        try:
                            timestamp = datetime.strptime(timestamp_str, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        # If no format matches, use current time
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()
            except Exception:
                timestamp = datetime.now()
            
            return (timestamp, component,component_proba, status, status_proba, db_val, topfreq1, topfreq2, topfreq3, topfreq4, topfreq5,tester_id,
                     *mfcc_features)
            
        except Exception as e:
            logger.warning(f"Failed to parse CSV row: {e}")
            return None
    
    def _import_rows_to_db(self, rows: List[Tuple]) -> int:
        """Import rows to database."""
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
        """Delete a file safely."""
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
        """Clean up old export files based on retention policy."""
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

class DataRotationService:
    """Main service for data rotation and export."""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config = DatabaseConfig(config_file)
        self.db_manager = DatabaseManager(self.config)
        self.csv_processor = CSVProcessor(self.db_manager, self.config)
    
    def start(self):
        """Start the data rotation service."""
        try:
            console.print_header("CSV ETL Data Import Service", "=", 80)
            console.print_info("Service", "Starting CSV processing service")
            console.print_info("Version", "2.0.0")
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
                    
                    # Process all CSV files
                    results = self.csv_processor.process_all_csv_files()
                    
                    # Cleanup old files
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
        """Clean up resources."""
        try:
            console.print_progress("Cleaning up resources...")
            self.db_manager.close()
            logger.info("Service cleanup completed")
            console.print_success("Service cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            console.print_error(f"Cleanup failed: {e}")

def main():
    """Main entry point."""
    try:
        service = DataRotationService()
        service.start()
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        console.print_error(f"Failed to start service: {e}")
        exit(1)

if __name__ == '__main__':
    main()
