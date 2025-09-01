#!/usr/bin/env python3
"""
Test script for SQL export functionality

This script demonstrates the SQL export and file management features.
"""

import os
import sys
from datetime import datetime, timedelta
from export_sql import DataRotationService, DatabaseConfig
from file_manager import FileManager

def test_sql_export():
    """Test the SQL export functionality."""
    print("Testing SQL Export Functionality")
    print("=" * 50)
    
    try:
        # Initialize the service
        service = DataRotationService()
        
        # Test configuration
        print(f"Export directory: {service.config.config['export_directory']}")
        print(f"Delete after export: {service.config.config.get('delete_after_export', False)}")
        print(f"Interval minutes: {service.config.config['interval_minutes']}")
        
        # Initialize database
        print("\nInitializing database...")
        service.exporter.init_db()
        print("Database initialized successfully")
        
        # Test export and rotate
        print("\nTesting export and rotate...")
        result = service.exporter.export_and_rotate()
        if result:
            print(f"Export completed: {result}")
        else:
            print("No data to export")
        
        # Test file management
        print("\nTesting file management...")
        file_manager = FileManager()
        
        # List files
        files = file_manager.list_files()
        print(f"Found {len(files)} exported files")
        
        if files:
            # Show file details
            file_manager.list_files(show_details=True)
            
            # Show summary
            file_manager.show_summary()
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    
    return True

def test_file_operations():
    """Test file management operations."""
    print("\nTesting File Management Operations")
    print("=" * 50)
    
    try:
        file_manager = FileManager()
        
        # List all files
        print("Listing all exported files:")
        files = file_manager.list_files()
        
        if files:
            # Get info about first file
            first_file = files[0]['name']
            print(f"\nGetting info for: {first_file}")
            info = file_manager.get_file_info(first_file)
            if info:
                for key, value in info.items():
                    if key != 'path':
                        print(f"  {key}: {value}")
        
        # Show summary
        print("\nFile summary:")
        file_manager.show_summary()
        
        print("\nFile operations test completed!")
        
    except Exception as e:
        print(f"File operations test failed: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    print("SQL Export and File Management Test Suite")
    print("=" * 60)
    
    # Test SQL export
    if not test_sql_export():
        print("SQL export test failed!")
        return 1
    
    # Test file operations
    if not test_file_operations():
        print("File operations test failed!")
        return 1
    
    print("\nAll tests completed successfully!")
    return 0

if __name__ == '__main__':
    exit(main())
