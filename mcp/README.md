# Avinor Flight Data MCP Server

A Python-based Model Context Protocol (MCP) server for advanced analysis of Avinor flight datasets.

## Features 🚀

- **CSV Metadata Extraction**: Comprehensive analysis of CSV file structure, data types, and statistics
- **Flight Pattern Analysis**: Analyze routes, temporal patterns, cancellations, and delays
- **Airport Route Analysis**: Route combinations between airport groups with missing route detection
- **Data Quality Validation**: Comprehensive data quality assessment with scoring and recommendations

## Quick Setup ⚡

### Prerequisites
- Python 3.8+ with pandas, numpy, and python-dateutil
- MCP package installed (`pip3 install mcp --user`)

### Installation

1. **Test the server**:
   ```bash
   cd /lhome/kristsey/AvinorHackathon/mcp
   python3 run_server.py
   ```

2. **For GitHub Copilot integration**, use the configuration from `copilot_config.json`

## Available Tools 🛠️

| Tool | Description | Usage |
|------|-------------|-------|
| `get_csv_metadata` | Extract file metadata and column analysis | `get_csv_metadata(file_path, sample_rows=5)` |
| `analyze_flight_patterns` | Analyze flight patterns by type | `analyze_flight_patterns(file_path, analysis_type, group_by=None)` |
| `analyze_airport_routes` | Route combination analysis | `analyze_airport_routes(historical_data, airport_groups, combination_type)` |
| `validate_data_quality` | Data quality assessment | `validate_data_quality(file_path, expected_schema=None)` |

### Analysis Types
- **Routes**: `"routes"` - Analyze flight route patterns
- **Temporal**: `"temporal"` - Time-based flight patterns  
- **Cancellations**: `"cancellations"` - Cancellation rate analysis
- **Delays**: `"delays"` - Delay pattern analysis

### Route Combination Types
- **Directional**: `"directional"` - A→B routes (considers direction)
- **Bidirectional**: `"bidirectional"` - A↔B routes (ignores direction)
- **Matrix**: `"matrix"` - Full route matrix analysis

## Available Resources 📚

| Resource | URI | Description |
|----------|-----|-------------|
| Flight Data Schema | `flight_data_schema` | Expected CSV structure and column definitions |
| Analysis Examples | `analysis_examples` | Example queries and analysis patterns |

## GitHub Copilot Integration 🤖

To use this MCP server with GitHub Copilot:

1. **Copy the configuration** from `copilot_config.json`
2. **Add it to your MCP settings** in your Copilot client
3. **Restart your Copilot client** to load the new server

### Configuration Details
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

## Sample Data Expected 📊

The server works with Avinor flight datasets:
- `historical_flights.csv` - Historical flight data
- `airportgroups.csv` - Airport group mappings  
- `schedule_oct2025.csv` - Scheduled flight data
