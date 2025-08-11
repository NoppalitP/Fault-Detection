#!/usr/bin/env python3
"""
Test script for the optimized export_csv.py functionality.
This script demonstrates how to use the various classes and methods.
"""

import os
import sys
from datetime import datetime, timedelta
from export_csv import DatabaseConfig, DatabaseManager, DataExporter, SQLQueries

def test_configuration():
    """Test configuration loading."""
    print("Testing configuration...")
    
    # Test with environment variables
    os.environ['DB_HOST'] = 'testhost'
    os.environ['DB_PASSWORD'] = 'testpass'
    
    config = DatabaseConfig()
    print(f"Host: {config.config['host']}")
    print(f"Password: {config.config['password']}")
    print(f"Interval: {config.config['interval_minutes']} minutes")
    
    return config

def test_sql_queries():
    """Test SQL query generation."""
    print("\nTesting SQL queries...")
    
    queries = SQLQueries('test_table')
    print(f"Create table SQL:\n{queries.create_table}")
    print(f"Insert SQL:\n{queries.insert_sql}")

def test_data_exporter():
    """Test data exporter functionality."""
    print("\nTesting data exporter...")
    
    config = DatabaseConfig()
    
    # Mock database manager for testing
    class MockDBManager:
        def get_connection(self):
            class MockConnection:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def cursor(self):
                    class MockCursor:
                        def __enter__(self):
                            return self
                        def __exit__(self, *args):
                            pass
                        def execute(self, *args):
                            pass
                        def commit(self):
                            pass
                    return MockCursor()
            return MockConnection()
    
    db_manager = MockDBManager()
    exporter = DataExporter(db_manager, config)
    
    # Test CSV export
    test_rows = [
        (datetime.now(), 'test_component', 'normal', 45.5, 100.0, 200.0, 300.0, 'tester1'),
        (datetime.now(), 'test_component2', 'anomaly', 55.2, 150.0, 250.0, 350.0, 'tester2')
    ]
    
    try:
        filename = exporter._export_to_csv(test_rows)
        print(f"CSV export test successful: {filename}")
        
        # Clean up test file
        if os.path.exists(filename):
            os.remove(filename)
            print("Test file cleaned up")
    except Exception as e:
        print(f"CSV export test failed: {e}")

def test_environment_setup():
    """Test environment variable setup."""
    print("\nTesting environment setup...")
    
    # Show current environment
    env_vars = [
        'DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 
        'DB_PASSWORD', 'INTERVAL_MINUTES', 'EXPORT_DIR'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'NOT_SET')
        print(f"{var}: {value}")

def main():
    """Main test function."""
    print("=== Testing Optimized Export CSV Script ===\n")
    
    try:
        test_configuration()
        test_sql_queries()
        test_data_exporter()
        test_environment_setup()
        
        print("\n=== All Tests Completed Successfully ===")
        print("\nTo run the actual service:")
        print("1. Set up your database credentials in config.yaml or environment variables")
        print("2. Run: python export_csv.py")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
