import psycopg2
import psycopg2.pool
import time
import logging
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sql_etl.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration class with environment variable support."""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
    
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
            'table_name': os.getenv('DB_TABLE_NAME', 'measurements'),
            'interval_minutes': int(os.getenv('INTERVAL_MINUTES', '5')),
            'export_directory': os.getenv('EXPORT_DIR', 'exports'),
            'retention_days': int(os.getenv('RETENTION_DAYS', '30')),
            'delete_after_export': os.getenv('DELETE_AFTER_EXPORT', 'true').lower() == 'true'
        }
        
        # Try to load from config file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        return config
    
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
            db_config = self.config.get_db_config()
            pool_config = self.config.get_pool_config()
            
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                pool_config['min_connections'],
                pool_config['max_connections'],
                **db_config
            )
            logger.info("Connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
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
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def close(self):
        """Close the connection pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("Connection pool closed")

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
            status TEXT,
            db REAL,
            topfreq1 REAL,
            topfreq2 REAL,
            topfreq3 REAL,
            tester_id TEXT
        );
        """
        
        self.create_hypertable = f"""
        SELECT create_hypertable('{self.table_name}', 'timestamp', if_not_exists => TRUE);
        """
        
        self.insert_sql = f"""
        INSERT INTO {self.table_name} 
        (timestamp, component, status, db, topfreq1, topfreq2, topfreq3, tester_id) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        self.select_old_rows = f"SELECT * FROM {self.table_name} WHERE timestamp <= %s"
        self.delete_old_rows = f"DELETE FROM {self.table_name} WHERE timestamp <= %s"
        
        # Add index for better performance
        self.create_index = f"""
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
        ON {self.table_name} (timestamp);
        """

class DataExporter:
    """Handles data export and rotation operations."""
    
    def __init__(self, db_manager: DatabaseManager, config: DatabaseConfig):
        self.db_manager = db_manager
        self.config = config
        self.queries = SQLQueries(config.config['table_name'])
        self.export_dir = Path(config.config['export_directory'])
        self.export_dir.mkdir(exist_ok=True)
    
    def init_db(self):
        """Initialize the database schema."""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(self.queries.create_extension)
                    cur.execute(self.queries.create_table)
                    cur.execute(self.queries.create_hypertable)
                    cur.execute(self.queries.create_index)
                    conn.commit()
                    logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def insert_rows_batch(self, rows: List[Tuple]) -> int:
        """Insert multiple rows in a batch for better performance."""
        if not rows:
            return 0
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.executemany(self.queries.insert_sql, rows)
                    conn.commit()
                    inserted_count = cur.rowcount
                    logger.info(f"Inserted {inserted_count} rows in batch")
                    return inserted_count
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            raise
    
    def export_and_rotate(self) -> Optional[str]:
        """Export old data to SQL and delete from database."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.config.config['interval_minutes'])
        
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    # Fetch old rows
                    cur.execute(self.queries.select_old_rows, (cutoff,))
                    rows = cur.fetchall()
                    
                    if not rows:
                        logger.info("No rows to export")
                        return None
                    
                    # Export to SQL
                    filename = self._export_to_sql(rows)
                    
                    # Delete old rows from database
                    cur.execute(self.queries.delete_old_rows, (cutoff,))
                    deleted_count = cur.rowcount
                    conn.commit()
                    
                    logger.info(f"Exported {len(rows)} rows to {filename}")
                    logger.info(f"Deleted {deleted_count} rows older than {self.config.config['interval_minutes']} minutes from database")
                    
                    # Delete the exported file if configured
                    if self.config.config.get('delete_after_export', False):
                        self._delete_file(filename)
                        logger.info(f"Deleted exported file: {filename}")
                    
                    return filename
                    
        except Exception as e:
            logger.error(f"Export and rotate operation failed: {e}")
            raise
    
    def _export_to_sql(self, rows: List[Tuple]) -> str:
        """Export rows to SQL file with INSERT statements."""
        timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = self.export_dir / f"export_{timestamp_str}.sql"
        
        with open(filename, 'w', encoding='utf-8') as f:
            # Write header comment
            f.write(f"-- Data export from {self.config.config['table_name']} table\n")
            f.write(f"-- Export timestamp: {datetime.utcnow().isoformat()}\n")
            f.write(f"-- Total rows: {len(rows)}\n\n")
            
            # Write table creation statement (if needed for import)
            f.write(f"-- Table structure (if needed for import)\n")
            f.write(f"CREATE TABLE IF NOT EXISTS {self.config.config['table_name']} (\n")
            f.write("    timestamp TIMESTAMPTZ NOT NULL,\n")
            f.write("    component TEXT,\n")
            f.write("    status TEXT,\n")
            f.write("    db REAL,\n")
            f.write("    topfreq1 REAL,\n")
            f.write("    topfreq2 REAL,\n")
            f.write("    topfreq3 REAL,\n")
            f.write("    tester_id TEXT\n")
            f.write(");\n\n")
            
            # Write INSERT statements
            f.write("-- Data INSERT statements\n")
            for row in rows:
                # Format the row values properly for SQL
                formatted_values = []
                for value in row:
                    if value is None:
                        formatted_values.append('NULL')
                    elif isinstance(value, str):
                        # Escape single quotes in strings
                        escaped_value = value.replace("'", "''")
                        formatted_values.append(f"'{escaped_value}'")
                    elif isinstance(value, datetime):
                        formatted_values.append(f"'{value.isoformat()}'")
                    else:
                        formatted_values.append(str(value))
                
                f.write(f"INSERT INTO {self.config.config['table_name']} "
                       f"(timestamp, component, status, db, topfreq1, topfreq2, topfreq3, tester_id) "
                       f"VALUES ({', '.join(formatted_values)});\n")
        
        return str(filename)
    
    def _delete_file(self, filepath: str):
        """Delete a file safely."""
        try:
            file_path = Path(filepath)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Successfully deleted file: {filepath}")
            else:
                logger.warning(f"File not found for deletion: {filepath}")
        except Exception as e:
            logger.error(f"Failed to delete file {filepath}: {e}")
    
    def cleanup_old_exports(self):
        """Clean up old export files based on retention policy."""
        retention_days = self.config.config['retention_days']
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        try:
            for file_path in self.export_dir.glob("export_*.sql"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    logger.info(f"Deleted old export file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup old exports: {e}")
    
    def delete_specific_file(self, filename: str):
        """Delete a specific file by name."""
        file_path = self.export_dir / filename
        self._delete_file(str(file_path))
    
    def list_export_files(self) -> List[str]:
        """List all export files in the export directory."""
        try:
            files = [f.name for f in self.export_dir.glob("export_*.sql")]
            return sorted(files)
        except Exception as e:
            logger.error(f"Failed to list export files: {e}")
            return []

class DataRotationService:
    """Main service for data rotation and export."""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config = DatabaseConfig(config_file)
        self.db_manager = DatabaseManager(self.config)
        self.exporter = DataExporter(self.db_manager, self.config)
    
    def start(self):
        """Start the data rotation service."""
        try:
            self.exporter.init_db()
            interval_minutes = self.config.config['interval_minutes']
            
            logger.info(f"Starting data rotation service every {interval_minutes} minutes...")
            logger.info(f"Export directory: {self.config.config['export_directory']}")
            logger.info(f"Delete after export: {self.config.config.get('delete_after_export', False)}")
            
            while True:
                try:
                    self.exporter.export_and_rotate()
                    self.exporter.cleanup_old_exports()
                except Exception as e:
                    logger.error(f"Service iteration failed: {e}")
                
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
        except Exception as e:
            logger.error(f"Service failed: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.db_manager.close()
            logger.info("Service cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

def main():
    """Main entry point."""
    try:
        service = DataRotationService()
        service.start()
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        exit(1)

if __name__ == '__main__':
    main()
