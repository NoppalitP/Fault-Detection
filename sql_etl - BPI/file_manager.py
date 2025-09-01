#!/usr/bin/env python3
"""
File Manager Utility for SQL Export Management

This script provides utilities to manage exported SQL files including:
- Listing exported files
- Deleting specific files
- Cleaning up old files
- Viewing file information
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import os

# Configure logging with improved formatting
def setup_logging():
    """Setup enhanced logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"file_manager_{timestamp}.log"
    
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

class FileManager:
    """Manages CSV files in the source and export directories."""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config = self._load_config(config_file)
        self.export_dir = Path(self.config.get('export_directory', 'exports'))
        self.csv_source_dir = Path(self.config.get('csv_source_directory', 'csv_files'))
        self._log_config_summary()
    
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from file."""
        config = {
            'export_directory': os.getenv('EXPORT_DIR', 'exports'),
            'csv_source_directory': os.getenv('CSV_SOURCE_DIR', 'csv_files'),
            'retention_days': int(os.getenv('RETENTION_DAYS', '30'))
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config.update(file_config)
                    logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
                console.print_warning(f"Failed to load config file: {e}")
        else:
            logger.info("No config file found, using environment variables and defaults")
            console.print_info("Config", "Using environment variables and defaults")
        
        return config
    
    def _log_config_summary(self):
        """Log configuration summary."""
        console.print_section("File Manager Configuration")
        console.print_info("CSV Source Directory", str(self.csv_source_dir))
        console.print_info("Export Directory", str(self.export_dir))
        console.print_info("Retention Days", f"{self.config.get('retention_days', 30)} days")
        console.print_info("Log File", str(log_file))
    
    def list_files(self, show_details: bool = False, source_dir: bool = True) -> list:
        """List all CSV files in the specified directory."""
        try:
            target_dir = self.csv_source_dir if source_dir else self.export_dir
            dir_name = "CSV Source" if source_dir else "Export"
            
            console.print_progress(f"Scanning {dir_name.lower()} directory for CSV files...")
            files = []
            for file_path in target_dir.glob("*.csv"):
                file_info = {
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                    'path': str(file_path)
                }
                files.append(file_info)
            
            files.sort(key=lambda x: x['modified'], reverse=True)
            
            if show_details:
                console.print_section(f"{dir_name} CSV Files")
                console.print_info("Directory", str(target_dir))
                console.print_info("Total Files", len(files))
                
                if files:
                    print(f"\n{'Filename':<30} {'Size':<12} {'Modified':<20} {'Age':<10}")
                    print("-" * 80)
                    
                    for file_info in files:
                        age = datetime.now() - file_info['modified']
                        age_str = self._format_age(age)
                        size_str = self._format_size(file_info['size'])
                        print(f"{file_info['name']:<30} {size_str:<12} {file_info['modified'].strftime('%Y-%m-%d %H:%M:%S'):<20} {age_str:<10}")
                else:
                    console.print_info("Status", f"No CSV files found in {dir_name.lower()} directory")
            else:
                if files:
                    console.print_section(f"{dir_name} CSV Files ({len(files)} files)")
                    for file_info in files:
                        age = datetime.now() - file_info['modified']
                        age_str = self._format_age(age)
                        print(f"  📄 {file_info['name']} ({age_str})")
                else:
                    console.print_info("Status", f"No CSV files found in {dir_name.lower()} directory")
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            console.print_error(f"Failed to list files: {e}")
            return []
    
    def _format_age(self, age: timedelta) -> str:
        """Format age as human readable string."""
        if age.days > 0:
            return f"{age.days}d ago"
        elif age.seconds > 3600:
            hours = age.seconds // 3600
            return f"{hours}h ago"
        elif age.seconds > 60:
            minutes = age.seconds // 60
            return f"{minutes}m ago"
        else:
            return f"{age.seconds}s ago"
    
    def delete_file(self, filename: str, source_dir: bool = True) -> bool:
        """Delete a specific file from the specified directory."""
        try:
            target_dir = self.csv_source_dir if source_dir else self.export_dir
            dir_name = "CSV Source" if source_dir else "Export"
            
            console.print_progress(f"Deleting file from {dir_name.lower()} directory: {filename}")
            file_path = target_dir / filename
            if file_path.exists():
                file_size = file_path.stat().st_size
                file_path.unlink()
                logger.info(f"Successfully deleted: {filename} from {dir_name.lower()} directory")
                console.print_success(f"File deleted: {filename}")
                console.print_stats("Size Freed", file_size, "bytes")
                return True
            else:
                logger.warning(f"File not found: {filename} in {dir_name.lower()} directory")
                console.print_warning(f"File not found: {filename} in {dir_name.lower()} directory")
                return False
        except Exception as e:
            logger.error(f"Failed to delete {filename}: {e}")
            console.print_error(f"Failed to delete {filename}: {e}")
            return False
    
    def cleanup_old_files(self, days: int = None) -> int:
        """Clean up files older than specified days."""
        if days is None:
            days = self.config.get('retention_days', 30)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        freed_space = 0
        
        try:
            console.print_progress(f"Cleaning up files older than {days} days...")
            console.print_info("Cutoff Date", cutoff_date.strftime('%Y-%m-%d %H:%M:%S'))
            
            for file_path in target_dir.glob("*.csv"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path.name}")
                    deleted_count += 1
                    freed_space += file_size
            
            if deleted_count > 0:
                console.print_success(f"Cleanup completed: {deleted_count} old files deleted")
                console.print_stats("Files Deleted", deleted_count)
                console.print_stats("Space Freed", freed_space, "bytes")
            else:
                console.print_info("Cleanup Status", "No old files to delete")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
            console.print_error(f"Failed to cleanup old files: {e}")
            return 0
    
    def get_file_info(self, filename: str, source_dir: bool = True) -> dict:
        """Get detailed information about a specific file in the specified directory."""
        try:
            target_dir = self.csv_source_dir if source_dir else self.export_dir
            dir_name = "CSV Source" if source_dir else "Export"
            
            console.print_progress(f"Getting information for: {filename} from {dir_name.lower()} directory")
            file_path = target_dir / filename
            if not file_path.exists():
                logger.warning(f"File not found: {filename} in {dir_name.lower()} directory")
                console.print_warning(f"File not found: {filename} in {dir_name.lower()} directory")
                return {}
            
            stat = file_path.stat()
            file_info = {
                'name': filename,
                'size': stat.st_size,
                'size_human': self._format_size(stat.st_size),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'path': str(file_path)
            }
            
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to get file info: {e}")
            console.print_error(f"Failed to get file info: {e}")
            return {}
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def show_summary(self, source_dir: bool = True):
        """Show summary of CSV files in the specified directory."""
        target_dir = self.csv_source_dir if source_dir else self.export_dir
        dir_name = "CSV Source" if source_dir else "Export"
        
        console.print_progress(f"Generating {dir_name.lower()} summary...")
        files = self.list_files(source_dir=source_dir)
        
        if not files:
            console.print_info("Status", f"No CSV files found in {dir_name.lower()} directory")
            return
        
        total_size = sum(f['size'] for f in files)
        oldest_file = min(files, key=lambda x: x['modified'])
        newest_file = max(files, key=lambda x: x['modified'])
        
        console.print_section(f"{dir_name} Summary")
        console.print_info("Total Files", f"{len(files):,}")
        console.print_info("Total Size", self._format_size(total_size))
        console.print_info("Oldest File", f"{oldest_file['name']} ({oldest_file['modified'].strftime('%Y-%m-%d %H:%M:%S')})")
        console.print_info("Newest File", f"{newest_file['name']} ({newest_file['modified'].strftime('%Y-%m-%d %H:%M:%S')})")
        console.print_info("Directory", str(target_dir))
        
        # Show size distribution
        size_ranges = {'Small (<1MB)': 0, 'Medium (1-10MB)': 0, 'Large (>10MB)': 0}
        for file_info in files:
            size_mb = file_info['size'] / (1024 * 1024)
            if size_mb < 1:
                size_ranges['Small (<1MB)'] += 1
            elif size_mb < 10:
                size_ranges['Medium (1-10MB)'] += 1
            else:
                size_ranges['Large (>10MB)'] += 1
        
        console.print_section("File Size Distribution")
        for range_name, count in size_ranges.items():
            if count > 0:
                console.print_stats(range_name, count, "files")

def main():
    """Main entry point for command line interface."""
    try:
        console.print_header("CSV File Manager", "=", 80)
        console.print_info("Version", "2.0.0")
        console.print_info("Start Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        parser = argparse.ArgumentParser(description="Manage CSV files in source and export directories")
        parser.add_argument('--config', '-c', default='config.yaml', help='Configuration file path')
        parser.add_argument('--list', '-l', action='store_true', help='List all CSV files')
        parser.add_argument('--details', '-d', action='store_true', help='Show detailed file information')
        parser.add_argument('--delete', help='Delete specific file by name')
        parser.add_argument('--cleanup', type=int, metavar='DAYS', help='Clean up files older than DAYS')
        parser.add_argument('--summary', '-s', action='store_true', help='Show summary of CSV files')
        parser.add_argument('--info', help='Show detailed info for specific file')
        parser.add_argument('--source', action='store_true', help='Work with CSV source directory (default)')
        parser.add_argument('--export', action='store_true', help='Work with export directory')
        
        args = parser.parse_args()
        
        manager = FileManager(args.config)
        
        # Determine which directory to work with
        source_dir = not args.export  # Default to source directory unless --export is specified
        
        if args.list:
            manager.list_files(args.details, source_dir)
        elif args.delete:
            manager.delete_file(args.delete, source_dir)
        elif args.cleanup is not None:
            manager.cleanup_old_files(args.cleanup, source_dir)
        elif args.summary:
            manager.show_summary(source_dir)
        elif args.info:
            info = manager.get_file_info(args.info, source_dir)
            if info:
                dir_name = "CSV Source" if source_dir else "Export"
                console.print_section(f"File Information: {args.info} ({dir_name})")
                for key, value in info.items():
                    if key != 'path':
                        console.print_info(key.capitalize(), str(value))
        else:
            # Default action: show summary
            manager.show_summary(source_dir)
        
        console.print_section("Operation Completed")
        console.print_info("Status", "File manager operation completed successfully")
            
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        console.print_error(f"Operation failed: {e}")
        exit(1)

if __name__ == '__main__':
    main()
