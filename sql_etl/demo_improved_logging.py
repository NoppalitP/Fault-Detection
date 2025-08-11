#!/usr/bin/env python3
"""
Demo script for improved logging and console output

This script demonstrates the enhanced logging and user-friendly console output
features of the SQL export system.
"""

import time
from datetime import datetime, timedelta
from export_sql import DataRotationService, ConsolePrinter
from file_manager import FileManager

def demo_console_output():
    """Demonstrate the improved console output features."""
    console = ConsolePrinter()
    
    console.print_header("Improved Logging & Console Output Demo", "=", 80)
    console.print_info("Version", "2.0.0")
    console.print_info("Demo Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Demo different output types
    console.print_section("Console Output Types")
    console.print_info("Info", "This is an informational message")
    console.print_success("Success message with checkmark")
    console.print_warning("Warning message with warning symbol")
    console.print_error("Error message with X symbol")
    console.print_progress("Progress message with spinner")
    console.print_stats("Sample Count", 12345, "items")
    
    # Demo headers and sections
    console.print_section("Header and Section Examples")
    console.print_header("This is a Header", "*", 60)
    console.print_section("This is a Section", "~", 50)
    
    print("\n" + "="*80)
    print("Console output demo completed!")
    print("="*80)

def demo_file_manager():
    """Demonstrate the improved file manager output."""
    console = ConsolePrinter()
    
    console.print_header("File Manager Demo", "=", 80)
    
    try:
        # Initialize file manager
        console.print_progress("Initializing file manager...")
        file_manager = FileManager()
        
        # Show summary
        console.print_section("File Summary Demo")
        file_manager.show_summary()
        
        # List files
        console.print_section("File Listing Demo")
        files = file_manager.list_files()
        
        if files:
            console.print_success(f"Found {len(files)} exported files")
        else:
            console.print_info("Status", "No exported files found")
        
    except Exception as e:
        console.print_error(f"File manager demo failed: {e}")

def demo_export_service():
    """Demonstrate the improved export service output."""
    console = ConsolePrinter()
    
    console.print_header("Export Service Demo", "=", 80)
    
    try:
        # Initialize service (without starting the loop)
        console.print_progress("Initializing export service...")
        service = DataRotationService()
        
        # Show configuration
        console.print_section("Service Configuration")
        console.print_info("Database", f"{service.config.config['user']}@{service.config.config['host']}:{service.config.config['port']}/{service.config.config['dbname']}")
        console.print_info("Table", service.config.config['table_name'])
        console.print_info("Export Directory", service.config.config['export_directory'])
        console.print_info("Interval", f"{service.config.config['interval_minutes']} minutes")
        console.print_info("Delete After Export", "Yes" if service.config.config.get('delete_after_export', False) else "No")
        
        # Test database initialization (without actually connecting)
        console.print_section("Database Initialization Demo")
        console.print_progress("This would initialize the database schema...")
        console.print_success("Database initialization would complete successfully")
        
    except Exception as e:
        console.print_error(f"Export service demo failed: {e}")

def demo_logging_features():
    """Demonstrate the improved logging features."""
    console = ConsolePrinter()
    
    console.print_header("Logging Features Demo", "=", 80)
    
    console.print_section("Log File Structure")
    console.print_info("Log Directory", "logs/")
    console.print_info("Log Files", "Timestamped log files (e.g., sql_etl_20231201_120000.log)")
    console.print_info("File Format", "Detailed logging with function names and line numbers")
    console.print_info("Console Format", "Simplified logging for user readability")
    
    console.print_section("Log Levels")
    console.print_info("DEBUG", "Detailed information for debugging (file only)")
    console.print_info("INFO", "General information about program execution")
    console.print_info("WARNING", "Warning messages for potential issues")
    console.print_info("ERROR", "Error messages for failed operations")
    
    console.print_section("Log Content")
    console.print_info("Timestamps", "ISO format timestamps")
    console.print_info("Function Names", "Which function generated the log")
    console.print_info("Context", "Relevant context information")
    console.print_info("Performance", "Operation timing and statistics")

def main():
    """Main demo function."""
    console = ConsolePrinter()
    
    try:
        # Run all demos
        demo_console_output()
        time.sleep(1)
        
        demo_file_manager()
        time.sleep(1)
        
        demo_export_service()
        time.sleep(1)
        
        demo_logging_features()
        
        # Final summary
        console.print_header("Demo Completed Successfully", "🎉", 80)
        console.print_success("All demo sections completed")
        console.print_info("Next Steps", "Run the actual services to see real output")
        console.print_info("Commands", "python export_sql.py or python file_manager.py --summary")
        
    except Exception as e:
        console.print_error(f"Demo failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
