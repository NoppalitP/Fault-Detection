# SQL ETL Data Export and Rotation Service

This optimized version of the data export script provides a robust, production-ready solution for managing TimescaleDB data with automatic export and rotation capabilities.

## Features

- **Connection Pooling**: Efficient database connection management
- **Configuration Management**: Environment variables and YAML config file support
- **Error Handling**: Comprehensive error handling and logging
- **Batch Operations**: Optimized database operations for better performance
- **Automatic Cleanup**: File retention management and cleanup
- **Type Hints**: Full type annotations for better code clarity
- **Logging**: Structured logging with file and console output
- **Resource Management**: Proper connection handling with context managers

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (optional):
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=testco_database
export DB_USER=postgres
export DB_PASSWORD=your_password
export INTERVAL_MINUTES=5
export EXPORT_DIR=exports
export RETENTION_DAYS=30
```

3. Or create a `config.yaml` file (see `config.yaml` template)

## Usage

### Basic Usage
```bash
python export_csv.py
```

### With Custom Config
```bash
python export_csv.py --config custom_config.yaml
```

## Configuration

The service can be configured through:

1. **Environment Variables**: Set database and service parameters
2. **Config File**: YAML configuration file (default: `config.yaml`)
3. **Code Defaults**: Fallback values if neither above is provided

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `host` | Database host | localhost |
| `port` | Database port | 5432 |
| `dbname` | Database name | testco_database |
| `user` | Database user | postgres |
| `password` | Database password | (empty) |
| `min_connections` | Min connection pool size | 1 |
| `max_connections` | Max connection pool size | 10 |
| `table_name` | Target table name | measurements |
| `interval_minutes` | Export interval in minutes | 5 |
| `export_directory` | CSV export directory | exports |
| `retention_days` | File retention period | 30 |

## Architecture

The code is organized into several classes:

- **`DatabaseConfig`**: Configuration management
- **`DatabaseManager`**: Connection pool and database operations
- **`SQLQueries`**: SQL query definitions
- **`DataExporter`**: Data export and rotation logic
- **`DataRotationService`**: Main service orchestration

## Performance Improvements

- **Connection Pooling**: Reuses database connections
- **Batch Operations**: Uses `executemany` for multiple inserts
- **Indexing**: Automatic timestamp index creation
- **Context Managers**: Proper resource cleanup
- **Efficient Queries**: Optimized SQL with proper indexing

## Logging

The service provides comprehensive logging:

- **File Logging**: All logs saved to `sql_etl.log`
- **Console Output**: Real-time log display
- **Structured Format**: Timestamp, level, and message information
- **Error Tracking**: Detailed error logging with context

## Error Handling

- **Database Failures**: Automatic rollback and retry logic
- **Connection Issues**: Pool management and reconnection
- **File Operations**: Safe file handling with proper cleanup
- **Service Recovery**: Graceful degradation and restart capabilities

## Security Features

- **No Hardcoded Credentials**: All sensitive data externalized
- **Environment Variable Support**: Secure credential management
- **Input Validation**: SQL injection prevention
- **Resource Isolation**: Proper connection isolation

## Monitoring

The service provides several monitoring points:

- **Export Counts**: Number of rows exported
- **Deletion Counts**: Number of rows removed
- **File Management**: Export file creation and cleanup
- **Performance Metrics**: Database operation timing

## Troubleshooting

### Common Issues

1. **Connection Failures**: Check database credentials and network
2. **Permission Errors**: Verify file system permissions for export directory
3. **Memory Issues**: Adjust connection pool size based on system resources
4. **Performance**: Monitor database query performance and adjust indexes

### Log Analysis

Check the `sql_etl.log` file for detailed error information and operation status.

## Contributing

When modifying the code:

1. Maintain type hints
2. Add proper error handling
3. Update documentation
4. Follow the existing class structure
5. Add appropriate logging

## License

This code is part of the Fault-Detection project.
