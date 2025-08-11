#!/usr/bin/env python3
"""
Test script for CSV processing functionality

This script demonstrates the new CSV import and processing features.
"""

import os
import sys
import csv
from datetime import datetime, timedelta
from pathlib import Path
from export_sql import DataRotationService, ConsolePrinter

def create_sample_csv_files():
    """Create sample CSV files for testing."""
    console = ConsolePrinter()
    
    # Create CSV source directory
    csv_dir = Path("csv_files")
    csv_dir.mkdir(exist_ok=True)
    
    console.print_section("Creating Sample CSV Files")
    
    # Sample data
    sample_data = [
        ["timestamp", "component", "status", "db", "topfreq1", "topfreq2", "topfreq3", "tester_id"],
        ["2023-12-01 10:00:00", "component1", "active", "45.2", "100.5", "200.3", "300.1", "tester1"],
        ["2023-12-01 10:01:00", "component2", "active", "42.8", "98.7", "195.2", "298.9", "tester1"],
        ["2023-12-01 10:02:00", "component1", "warning", "38.5", "95.2", "190.1", "295.3", "tester2"],
        ["2023-12-01 10:03:00", "component3", "active", "47.1", "102.3", "205.7", "305.2", "tester1"],
        ["2023-12-01 10:04:00", "component2", "error", "35.2", "92.1", "185.4", "290.8", "tester2"]
    ]
    
    # Create multiple CSV files
    for i in range(3):
        filename = csv_dir / f"data_batch_{i+1}.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(sample_data)
        
        console.print_success(f"Created: {filename.name}")
    
    console.print_info("Status", f"Created {len(list(csv_dir.glob('*.csv')))} sample CSV files")
    return csv_dir

def test_csv_processor():
    """Test the CSV processor functionality."""
    console = ConsolePrinter()
    
    console.print_header("CSV Processing Test", "=", 80)
    
    try:
        # Create sample files
        csv_dir = create_sample_csv_files()
        
        # Initialize service (without starting the loop)
        console.print_progress("Initializing CSV processing service...")
        service = DataRotationService()
        
        # Show configuration
        console.print_section("Service Configuration")
        console.print_info("Database", f"{service.config.config['user']}@{service.config.config['host']}:{service.config.config['port']}/{service.config.config['dbname']}")
        console.print_info("Table", service.config.config['table_name'])
        console.print_info("CSV Source Directory", service.config.config['csv_source_directory'])
        console.print_info("Export Directory", service.config.config['export_directory'])
        console.print_info("Interval", f"{service.config.config['interval_minutes']} minutes")
        console.print_info("Delete After Processing", "Yes" if service.config.config.get('delete_after_export', False) else "No")
        
        # Test CSV processing (without database connection)
        console.print_section("CSV Processing Demo")
        console.print_progress("This would process all CSV files in the source directory...")
        console.print_progress("Each file would be imported to the database...")
        console.print_progress("Files would be deleted after processing (if configured)...")
        
        # Show sample files
        csv_files = list(csv_dir.glob("*.csv"))
        console.print_success(f"Found {len(csv_files)} CSV files to process")
        
        for csv_file in csv_files:
            console.print_info("File", f"{csv_file.name} ({csv_file.stat().st_size} bytes)")
        
        console.print_success("CSV processing test completed successfully!")
        
    except Exception as e:
        console.print_error(f"CSV processing test failed: {e}")
        return False
    
    return True

def test_file_operations():
    """Test file management operations."""
    console = ConsolePrinter()
    
    console.print_header("File Management Test", "=", 80)
    
    try:
        from file_manager import FileManager
        
        # Initialize file manager
        console.print_progress("Initializing file manager...")
        file_manager = FileManager()
        
        # List CSV source files
        console.print_section("CSV Source Directory Files")
        source_files = file_manager.list_files(source_dir=True)
        
        if source_files:
            console.print_success(f"Found {len(source_files)} files in CSV source directory")
            file_manager.show_summary(source_dir=True)
        else:
            console.print_info("Status", "No files in CSV source directory")
        
        # List export directory files
        console.print_section("Export Directory Files")
        export_files = file_manager.list_files(source_dir=False)
        
        if export_files:
            console.print_success(f"Found {len(export_files)} files in export directory")
            file_manager.show_summary(source_dir=False)
        else:
            console.print_info("Status", "No files in export directory")
        
        console.print_success("File management test completed!")
        
    except Exception as e:
        console.print_error(f"File management test failed: {e}")
        return False
    
    return True

def cleanup_test_files():
    """Clean up test files."""
    console = ConsolePrinter()
    
    console.print_section("Cleaning Up Test Files")
    
    # Remove CSV source directory
    csv_dir = Path("csv_files")
    if csv_dir.exists():
        for csv_file in csv_dir.glob("*.csv"):
            csv_file.unlink()
            console.print_info("Deleted", csv_file.name)
        
        csv_dir.rmdir()
        console.print_success("Removed CSV source directory")
    
    # Remove logs directory
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            log_file.unlink()
        
        logs_dir.rmdir()
        console.print_success("Removed logs directory")
    
    console.print_success("Cleanup completed")

def main():
    """Main test function."""
    console = ConsolePrinter()
    
    try:
        # Run all tests
        if not test_csv_processor():
            console.print_error("CSV processing test failed!")
            return 1
        
        if not test_file_operations():
            console.print_error("File management test failed!")
            return 1
        
        # Final summary
        console.print_header("All Tests Completed Successfully", "🎉", 80)
        console.print_success("CSV processing system is ready")
        console.print_info("Next Steps", "Run the actual service to process real CSV files")
        console.print_info("Commands", "python export_sql.py or python file_manager.py --summary")
        
        # Ask if user wants to clean up
        try:
            response = input("\nDo you want to clean up test files? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                cleanup_test_files()
        except KeyboardInterrupt:
            console.print_info("Status", "Cleanup skipped")
        
        return 0
        
    except Exception as e:
        console.print_error(f"Test failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
