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
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FileManager:
    """Manages exported SQL files."""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config = self._load_config(config_file)
        self.export_dir = Path(self.config.get('export_directory', 'exports'))
    
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from file."""
        config = {
            'export_directory': os.getenv('EXPORT_DIR', 'exports'),
            'retention_days': int(os.getenv('RETENTION_DAYS', '30'))
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        return config
    
    def list_files(self, show_details: bool = False) -> list:
        """List all exported SQL files."""
        try:
            files = []
            for file_path in self.export_dir.glob("export_*.sql"):
                file_info = {
                    'name': file_path.name,
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                    'path': str(file_path)
                }
                files.append(file_info)
            
            files.sort(key=lambda x: x['modified'], reverse=True)
            
            if show_details:
                print(f"\nExported SQL Files in: {self.export_dir}")
                print("-" * 80)
                print(f"{'Filename':<25} {'Size (KB)':<12} {'Modified':<20} {'Age'}")
                print("-" * 80)
                
                for file_info in files:
                    age = datetime.now() - file_info['modified']
                    age_str = self._format_age(age)
                    size_kb = file_info['size'] / 1024
                    print(f"{file_info['name']:<25} {size_kb:>8.1f} KB {file_info['modified'].strftime('%Y-%m-%d %H:%M:%S'):<20} {age_str}")
            else:
                print(f"\nExported SQL Files ({len(files)} files):")
                for file_info in files:
                    print(f"  {file_info['name']}")
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
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
    
    def delete_file(self, filename: str) -> bool:
        """Delete a specific file."""
        try:
            file_path = self.export_dir / filename
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Successfully deleted: {filename}")
                return True
            else:
                logger.warning(f"File not found: {filename}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete {filename}: {e}")
            return False
    
    def cleanup_old_files(self, days: int = None) -> int:
        """Clean up files older than specified days."""
        if days is None:
            days = self.config.get('retention_days', 30)
        
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        try:
            for file_path in self.export_dir.glob("export_*.sql"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path.name}")
                    deleted_count += 1
            
            logger.info(f"Cleanup completed: {deleted_count} files deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
            return 0
    
    def get_file_info(self, filename: str) -> dict:
        """Get detailed information about a specific file."""
        try:
            file_path = self.export_dir / filename
            if not file_path.exists():
                logger.warning(f"File not found: {filename}")
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
            return {}
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def show_summary(self):
        """Show summary of exported files."""
        files = self.list_files()
        
        if not files:
            print("No exported SQL files found.")
            return
        
        total_size = sum(f['size'] for f in files)
        oldest_file = min(files, key=lambda x: x['modified'])
        newest_file = max(files, key=lambda x: x['modified'])
        
        print(f"\nExport Summary:")
        print(f"  Total files: {len(files)}")
        print(f"  Total size: {self._format_size(total_size)}")
        print(f"  Oldest file: {oldest_file['name']} ({oldest_file['modified'].strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"  Newest file: {newest_file['name']} ({newest_file['modified'].strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"  Export directory: {self.export_dir}")

def main():
    """Main entry point for command line interface."""
    parser = argparse.ArgumentParser(description="Manage exported SQL files")
    parser.add_argument('--config', '-c', default='config.yaml', help='Configuration file path')
    parser.add_argument('--list', '-l', action='store_true', help='List all exported files')
    parser.add_argument('--details', '-d', action='store_true', help='Show detailed file information')
    parser.add_argument('--delete', help='Delete specific file by name')
    parser.add_argument('--cleanup', type=int, metavar='DAYS', help='Clean up files older than DAYS')
    parser.add_argument('--summary', '-s', action='store_true', help='Show summary of exported files')
    parser.add_argument('--info', help='Show detailed info for specific file')
    
    args = parser.parse_args()
    
    try:
        manager = FileManager(args.config)
        
        if args.list:
            manager.list_files(args.details)
        elif args.delete:
            manager.delete_file(args.delete)
        elif args.cleanup is not None:
            manager.cleanup_old_files(args.cleanup)
        elif args.summary:
            manager.show_summary()
        elif args.info:
            info = manager.get_file_info(args.info)
            if info:
                print(f"\nFile Information for: {args.info}")
                print("-" * 50)
                for key, value in info.items():
                    if key != 'path':
                        print(f"{key.capitalize()}: {value}")
        else:
            # Default action: show summary
            manager.show_summary()
            
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        exit(1)

if __name__ == '__main__':
    main()
