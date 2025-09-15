#!/usr/bin/env python3
"""
Avinor Flight Data MCP Server

A Model Context Protocol server for advanced analysis of Avinor flight datasets.
Provides tools for CSV metadata extraction, flight pattern analysis, and data quality validation.
"""

import json
import sys
import asyncio
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime

# MCP imports - using the Python SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        Resource,
        TextResourceContents,
        CallToolResult,
        ListResourcesResult,
        ListToolsResult,
        ReadResourceResult,
    )
except ImportError:
    print("MCP Python SDK not found. Please install with: uv add mcp", file=sys.stderr)
    sys.exit(1)

# Import analyzer from the same directory
import sys
import os

sys.path.append(os.path.dirname(__file__))
from analyzer import AvinorDataAnalyzer


class AvinorFlightDataServer:
    """MCP Server for Avinor flight data analysis."""

    def __init__(self):
        self.server = Server("avinor-flight-data")
        self.analyzer = AvinorDataAnalyzer()

        # Register tools
        self._register_tools()

        # Register resources
        self._register_resources()

    def _register_tools(self):
        """Register all available tools."""

        @self.server.call_tool()
        async def get_csv_paths():
            """Get paths to sample CSV files for testing."""
            sample_dir = Path(__file__).parent / "data"
            if not sample_dir.exists():
                return [{"error": "Sample data directory not found."}]

            csv_files = list(sample_dir.glob("*.csv"))
            if not csv_files:
                return [{"error": "No CSV files found in sample data directory."}]

            return [{"csv_paths": [str(f) for f in csv_files]}]

        @self.server.call_tool()
        async def get_csv_metadata(
            file_path: str, sample_rows: int = 5
        ) -> List[Dict[str, Any]]:
            """Extract comprehensive metadata from a CSV file.

            Args:
                file_path: Absolute path to the CSV file
                sample_rows: Number of sample rows to include (default: 5)

            Returns:
                Detailed metadata including file info, column types, and sample data
            """
            try:
                metadata = await self.analyzer.get_csv_metadata(file_path, sample_rows)
                return [{"metadata": metadata}]
            except Exception as e:
                return [{"error": f"Failed to analyze CSV: {str(e)}"}]

        @self.server.call_tool()
        async def analyze_flight_patterns(
            file_path: str, analysis_type: str, group_by: Optional[str] = None
        ) -> List[Dict[str, Any]]:
            """Analyze flight patterns in the dataset.

            Args:
                file_path: Path to the flight data CSV file
                analysis_type: Type of analysis ('routes', 'temporal', 'cancellations', 'delays')
                group_by: Optional column to group analysis by

            Returns:
                Pattern analysis results with insights and statistics
            """
            try:
                if analysis_type not in [
                    "routes",
                    "temporal",
                    "cancellations",
                    "delays",
                ]:
                    return [
                        {
                            "error": "Invalid analysis_type. Must be one of: routes, temporal, cancellations, delays"
                        }
                    ]

                results = await self.analyzer.analyze_flight_patterns(
                    file_path, analysis_type, group_by
                )
                return [{"analysis": results}]
            except Exception as e:
                return [{"error": f"Flight pattern analysis failed: {str(e)}"}]

        @self.server.call_tool()
        async def analyze_airport_routes(
            historical_data_path: str,
            airport_groups_path: str,
            combination_type: str = "directional",
        ) -> List[Dict[str, Any]]:
            """Analyze route combinations between airport groups.

            Args:
                historical_data_path: Path to historical flight data CSV
                airport_groups_path: Path to airport groups mapping CSV
                combination_type: Analysis type ('directional', 'bidirectional', 'matrix')

            Returns:
                Route combination analysis with statistics and missing combinations
            """
            try:
                if combination_type not in ["directional", "bidirectional", "matrix"]:
                    return [
                        {
                            "error": "Invalid combination_type. Must be one of: directional, bidirectional, matrix"
                        }
                    ]

                results = await self.analyzer.analyze_airport_routes(
                    historical_data_path, airport_groups_path, combination_type
                )
                return [{"route_analysis": results}]
            except Exception as e:
                return [{"error": f"Airport route analysis failed: {str(e)}"}]

        @self.server.call_tool()
        async def validate_data_quality(
            file_path: str, expected_schema: Optional[str] = None
        ) -> List[Dict[str, Any]]:
            """Validate data quality and identify issues.

            Args:
                file_path: Path to the CSV file to validate
                expected_schema: Optional JSON string of expected column types

            Returns:
                Data quality report with issues, score, and recommendations
            """
            try:
                schema = None
                if expected_schema:
                    schema = json.loads(expected_schema)

                report = await self.analyzer.validate_data_quality(file_path, schema)
                return [{"quality_report": report}]
            except Exception as e:
                return [{"error": f"Data quality validation failed: {str(e)}"}]

    def _register_resources(self):
        """Register available resources."""

        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available documentation resources."""
            return [
                Resource(
                    uri="flight_data_schema",
                    name="Flight Data Schema",
                    mimeType="text/markdown",
                    description="Expected structure and column definitions for flight data CSV files",
                ),
                Resource(
                    uri="analysis_examples",
                    name="Analysis Examples",
                    mimeType="text/markdown",
                    description="Example queries and analysis patterns for flight data exploration",
                ),
            ]

        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read resource content by URI."""
            if uri == "flight_data_schema":
                content = """# Flight Data Schema

## Expected Columns

### Historical Flight Data
- `dep_airport`: Departure airport code (string)
- `arr_airport`: Arrival airport code (string)
- `dep_airport_group`: Departure airport group (A-G)
- `arr_airport_group`: Arrival airport group (A-G)
- `scheduled_dep`: Scheduled departure time
- `actual_dep`: Actual departure time
- `dep_delay`: Departure delay in minutes (numeric)
- `cancelled`: Cancellation status (boolean or 0/1)
- `month`: Month of flight (numeric)
- `weekday`: Day of week (numeric or string)

### Airport Groups Data
- `airport`: Airport code (string)
- `airport_group`: Group classification (A-G)

### Schedule Data
- Similar to historical data with additional scheduling fields
"""
                return ReadResourceResult(
                    contents=[TextResourceContents(type="text", text=content)]
                )

            elif uri == "analysis_examples":
                content = """# Analysis Examples

## CSV Metadata Extraction
```python
# Get comprehensive metadata for a flight data file
metadata = get_csv_metadata("/path/to/historical_flights.csv", sample_rows=10)
```

## Flight Pattern Analysis
```python
# Analyze route patterns
route_patterns = analyze_flight_patterns(
    "/path/to/historical_flights.csv", 
    "routes", 
    group_by="dep_airport_group"
)

# Analyze temporal patterns by month
temporal_patterns = analyze_flight_patterns(
    "/path/to/historical_flights.csv", 
    "temporal", 
    group_by="month"
)

# Analyze cancellation patterns
cancellation_analysis = analyze_flight_patterns(
    "/path/to/historical_flights.csv", 
    "cancellations", 
    group_by="dep_airport_group"
)
```

## Route Combination Analysis
```python
# Analyze directional routes between airport groups
route_combinations = analyze_airport_routes(
    "/path/to/historical_flights.csv",
    "/path/to/airportgroups.csv", 
    "directional"
)

# Find bidirectional route pairs
bidirectional_routes = analyze_airport_routes(
    "/path/to/historical_flights.csv",
    "/path/to/airportgroups.csv", 
    "bidirectional"
)
```

## Data Quality Validation
```python
# Basic data quality check
quality_report = validate_data_quality("/path/to/historical_flights.csv")

# With expected schema validation
schema = {"dep_airport": "string", "dep_delay": "numeric", "cancelled": "boolean"}
detailed_report = validate_data_quality(
    "/path/to/historical_flights.csv", 
    json.dumps(schema)
)
```
"""
                return ReadResourceResult(
                    contents=[TextResourceContents(type="text", text=content)]
                )

            else:
                raise ValueError(f"Unknown resource: {uri}")


async def main():
    """Main entry point for the MCP server."""
    server_instance = AvinorFlightDataServer()

    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server_instance.server.run(
            read_stream,
            write_stream,
            server_instance.server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
