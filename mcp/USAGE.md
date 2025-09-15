# Avinor MCP Server - Usage Guide

## Quick Start with GitHub Copilot

### 1. Configuration
Add this to your MCP client configuration:
```json
{
  "mcpServers": {
    "avinor-flight-data": {
      "command": "python3",
      "args": ["/lhome/kristsey/AvinorHackathon/mcp/run_server.py"],
      "env": {
        "PYTHONPATH": "/lhome/kristsey/AvinorHackathon/mcp/src"
      }
    }
  }
}
```

### 2. Example Prompts for Copilot

#### Analyze CSV Metadata
```
"Use the avinor-flight-data MCP server to analyze the metadata of /path/to/historical_flights.csv"
```

#### Flight Pattern Analysis
```
"Using the avinor MCP server, analyze route patterns in the historical flight data, grouped by airport groups"
```

#### Data Quality Check
```
"Please use the avinor-flight-data server to validate the data quality of my flight dataset"
```

#### Route Combination Analysis
```
"Use the MCP server to find missing route combinations between airport groups A-G"
```

## Tool Reference

### get_csv_metadata
**Purpose**: Extract comprehensive metadata from CSV files
**Parameters**:
- `file_path` (string): Absolute path to CSV file
- `sample_rows` (int, optional): Number of sample rows (default: 5)

**Example**:
```python
get_csv_metadata("/data/historical_flights.csv", sample_rows=10)
```

### analyze_flight_patterns
**Purpose**: Analyze flight patterns by various dimensions
**Parameters**:
- `file_path` (string): Path to flight data CSV
- `analysis_type` (string): "routes", "temporal", "cancellations", or "delays"
- `group_by` (string, optional): Column to group analysis by

**Examples**:
```python
# Route analysis by airport group
analyze_flight_patterns("/data/flights.csv", "routes", "dep_airport_group")

# Temporal patterns by month
analyze_flight_patterns("/data/flights.csv", "temporal", "month")

# Cancellation analysis
analyze_flight_patterns("/data/flights.csv", "cancellations")
```

### analyze_airport_routes
**Purpose**: Analyze route combinations between airport groups
**Parameters**:
- `historical_data_path` (string): Path to historical flight data
- `airport_groups_path` (string): Path to airport groups mapping
- `combination_type` (string): "directional", "bidirectional", or "matrix"

**Example**:
```python
analyze_airport_routes(
    "/data/historical_flights.csv",
    "/data/airportgroups.csv", 
    "bidirectional"
)
```

### validate_data_quality
**Purpose**: Comprehensive data quality assessment
**Parameters**:
- `file_path` (string): Path to CSV file
- `expected_schema` (string, optional): JSON string of expected column types

**Examples**:
```python
# Basic validation
validate_data_quality("/data/flights.csv")

# With schema validation
schema = '{"dep_airport": "string", "dep_delay": "float", "cancelled": "boolean"}'
validate_data_quality("/data/flights.csv", schema)
```

## Resources Available

### flight_data_schema
Documentation of expected flight data structure and column definitions.

### analysis_examples  
Example queries and analysis patterns for flight data exploration.

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Ensure pandas, numpy, and mcp packages are installed
   - Run: `pip3 install mcp pandas numpy python-dateutil --user`

2. **"File not found" errors**
   - Use absolute file paths
   - Ensure CSV files exist and are readable

3. **Server won't start**
   - Check Python version (3.8+ required)
   - Verify all dependencies are installed
   - Run validation: `python3 validate_setup.py`

### Debug Commands
```bash
# Test server startup
cd /lhome/kristsey/AvinorHackathon/mcp
python3 run_server.py

# Validate setup
python3 validate_setup.py

# Check dependencies
python3 -c "import pandas, numpy, mcp; print('All dependencies OK')"
```

## Data File Locations
Expected data files in the project:
- `/lhome/kristsey/AvinorHackathon/data/historical_flights.csv`
- `/lhome/kristsey/AvinorHackathon/data/airportgroups.csv`
- `/lhome/kristsey/AvinorHackathon/data/schedule_oct2025.csv`
