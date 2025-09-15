"""
Avinor Data Analyzer

Core analysis functionality for flight data processing and insights.
Provides methods for CSV metadata extraction, pattern analysis, and data quality validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json


class AvinorDataAnalyzer:
    """Core analyzer for Avinor flight data."""

    async def get_csv_metadata(
        self, file_path: str, sample_rows: int = 5
    ) -> Dict[str, Any]:
        """Extract comprehensive metadata from a CSV file.

        Args:
            file_path: Path to the CSV file
            sample_rows: Number of sample rows to include

        Returns:
            Dictionary containing file metadata, column information, and sample data
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file stats
        stat = path.stat()

        # Read CSV with pandas
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {str(e)}")

        # Get sample data
        sample_data = df.head(sample_rows).to_dict("records")

        # Analyze columns
        columns_info = []
        for col in df.columns:
            col_data = df[col]
            null_count = col_data.isnull().sum()
            unique_count = col_data.nunique()

            # Infer data type
            dtype = self._infer_column_type(col_data)

            # Get sample values (non-null)
            sample_values = col_data.dropna().head(3).tolist()

            # Basic statistics for numeric columns
            statistics = None
            if dtype in ["integer", "float"]:
                numeric_data = pd.to_numeric(col_data, errors="coerce")
                statistics = {
                    "min": (
                        float(numeric_data.min())
                        if not numeric_data.isna().all()
                        else None
                    ),
                    "max": (
                        float(numeric_data.max())
                        if not numeric_data.isna().all()
                        else None
                    ),
                    "mean": (
                        float(numeric_data.mean())
                        if not numeric_data.isna().all()
                        else None
                    ),
                    "median": (
                        float(numeric_data.median())
                        if not numeric_data.isna().all()
                        else None
                    ),
                }

            columns_info.append(
                {
                    "name": col,
                    "index": df.columns.get_loc(col),
                    "data_type": dtype,
                    "null_count": int(null_count),
                    "unique_count": int(unique_count),
                    "sample_values": sample_values,
                    "statistics": statistics,
                }
            )

        return {
            "file_name": path.name,
            "file_path": str(path.absolute()),
            "file_size": stat.st_size,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": columns_info,
            "sample_rows": sample_data,
            "encoding": "utf-8",  # pandas default
            "delimiter": ",",  # assuming comma-separated
            "has_header": True,  # pandas default assumption
            "created_date": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_date": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    async def analyze_flight_patterns(
        self, file_path: str, analysis_type: str, group_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze flight patterns in the dataset.

        Args:
            file_path: Path to the flight data CSV
            analysis_type: Type of analysis ('routes', 'temporal', 'cancellations', 'delays')
            group_by: Optional column to group by

        Returns:
            Analysis results with patterns and insights
        """
        df = pd.read_csv(file_path)

        patterns = []
        insights = []
        summary = {}

        if analysis_type == "routes":
            patterns = self._analyze_route_patterns(df, group_by)
            insights.append(f"Analyzed {len(patterns)} unique routes")
            summary["top_route"] = patterns[0] if patterns else None

        elif analysis_type == "temporal":
            patterns = self._analyze_temporal_patterns(df, group_by)
            insights.append(
                f"Identified temporal patterns across {group_by or 'time periods'}"
            )

        elif analysis_type == "cancellations":
            patterns = self._analyze_cancellation_patterns(df, group_by)
            total_cancelled = sum(p.get("cancelled", 0) for p in patterns)
            cancel_rate = (total_cancelled / len(df) * 100) if len(df) > 0 else 0
            insights.append(f"Overall cancellation rate: {cancel_rate:.2f}%")
            summary["cancellation_rate"] = cancel_rate

        elif analysis_type == "delays":
            patterns = self._analyze_delay_patterns(df, group_by)
            avg_delay = (
                np.mean([p.get("avg_delay", 0) for p in patterns]) if patterns else 0
            )
            insights.append(f"Average delay across patterns: {avg_delay:.2f} minutes")
            summary["average_delay"] = avg_delay

        return {
            "analysis_type": analysis_type,
            "total_flights": len(df),
            "group_by": group_by,
            "patterns": patterns,
            "insights": insights,
            "summary": summary,
        }

    async def analyze_airport_routes(
        self,
        historical_data_path: str,
        airport_groups_path: str,
        combination_type: str = "directional",
    ) -> Dict[str, Any]:
        """Analyze route combinations between airport groups.

        Args:
            historical_data_path: Path to historical flight data
            airport_groups_path: Path to airport groups mapping
            combination_type: Type of analysis ('directional', 'bidirectional', 'matrix')

        Returns:
            Route combination analysis results
        """
        flights_df = pd.read_csv(historical_data_path)
        groups_df = pd.read_csv(airport_groups_path)

        # Create airport to group mapping
        airport_to_group = dict(zip(groups_df["airport"], groups_df["airport_group"]))

        # Filter valid flights with both departure and arrival groups
        valid_flights = flights_df[
            (flights_df["dep_airport_group"].notna())
            & (flights_df["arr_airport_group"].notna())
        ].copy()

        # Count route combinations
        route_combinations = {}

        for _, flight in valid_flights.iterrows():
            dep_group = flight["dep_airport_group"]
            arr_group = flight["arr_airport_group"]

            if combination_type == "bidirectional":
                route_key = "↔".join(sorted([dep_group, arr_group]))
            else:
                route_key = f"{dep_group}→{arr_group}"

            route_combinations[route_key] = route_combinations.get(route_key, 0) + 1

        # Sort by frequency
        sorted_routes = sorted(
            route_combinations.items(), key=lambda x: x[1], reverse=True
        )
        route_list = [
            {"route": route, "count": count} for route, count in sorted_routes
        ]

        # Find missing combinations
        missing_combinations = self._find_missing_combinations(
            route_combinations, combination_type
        )

        # Calculate statistics
        within_group_flights = len(
            valid_flights[
                valid_flights["dep_airport_group"] == valid_flights["arr_airport_group"]
            ]
        )
        between_group_flights = len(valid_flights) - within_group_flights

        return {
            "combination_type": combination_type,
            "total_routes": len(route_combinations),
            "route_combinations": route_list,
            "missing_combinations": missing_combinations,
            "top_routes": route_list[:10],
            "statistics": {
                "within_group_flights": within_group_flights,
                "between_group_flights": between_group_flights,
                "route_distribution": route_combinations,
            },
        }

    async def validate_data_quality(
        self, file_path: str, expected_schema: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Validate data quality and identify issues.

        Args:
            file_path: Path to the CSV file
            expected_schema: Optional dictionary of expected column types

        Returns:
            Data quality report with issues and recommendations
        """
        df = pd.read_csv(file_path)
        issues = []

        if df.empty:
            issues.append(
                {
                    "type": "missing_values",
                    "count": 1,
                    "severity": "high",
                    "description": "File is empty or could not be read",
                }
            )

            return {
                "file_name": Path(file_path).name,
                "total_rows": 0,
                "total_columns": 0,
                "issues": issues,
                "score": 0,
                "recommendations": ["File appears to be empty or corrupted"],
            }

        # Check for missing values
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_percentage = (null_count / len(df)) * 100
                severity = (
                    "high"
                    if null_percentage > 10
                    else "medium" if null_percentage > 1 else "low"
                )

                issues.append(
                    {
                        "type": "missing_values",
                        "column": col,
                        "count": int(null_count),
                        "severity": severity,
                        "description": f"Column '{col}' has {null_count} missing values ({null_percentage:.1f}%)",
                    }
                )

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(
                {
                    "type": "duplicates",
                    "count": int(duplicate_count),
                    "severity": "medium",
                    "description": f"Found {duplicate_count} duplicate rows",
                }
            )

        # Schema validation if provided
        if expected_schema:
            for col, expected_type in expected_schema.items():
                if col in df.columns:
                    actual_type = self._infer_column_type(df[col])
                    if actual_type != expected_type:
                        issues.append(
                            {
                                "type": "type_mismatch",
                                "column": col,
                                "count": 1,
                                "severity": "medium",
                                "description": f"Column '{col}' expected {expected_type}, found {actual_type}",
                            }
                        )

        # Calculate quality score (0-100)
        total_penalty = sum(
            {"high": 20, "medium": 10, "low": 5}.get(issue["severity"], 0)
            for issue in issues
        )

        score = max(0, 100 - total_penalty)

        # Generate recommendations
        recommendations = []
        if any(issue["type"] == "missing_values" for issue in issues):
            recommendations.append(
                "Consider data imputation or filtering for missing values"
            )
        if any(issue["type"] == "duplicates" for issue in issues):
            recommendations.append("Remove duplicate rows to improve data quality")
        if any(issue["type"] == "type_mismatch" for issue in issues):
            recommendations.append("Review and standardize column data types")

        if not recommendations:
            recommendations.append("Data quality looks good!")

        return {
            "file_name": Path(file_path).name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "issues": issues,
            "score": score,
            "recommendations": recommendations,
        }

    def _infer_column_type(self, series: pd.Series) -> str:
        """Infer the data type of a pandas Series."""
        if series.empty:
            return "unknown"

        # Remove null values for type inference
        non_null_series = series.dropna()
        if non_null_series.empty:
            return "null"

        # Check if numeric
        try:
            numeric_series = pd.to_numeric(non_null_series, errors="raise")
            # Check if all values are integers
            if (numeric_series % 1 == 0).all():
                return "integer"
            else:
                return "float"
        except (ValueError, TypeError):
            pass

        # Check if boolean
        if non_null_series.dtype == bool or set(non_null_series.unique()).issubset(
            {True, False, "true", "false", "True", "False", 1, 0}
        ):
            return "boolean"

        # Check if datetime
        try:
            pd.to_datetime(non_null_series.head(10), errors="raise")
            return "datetime"
        except (ValueError, TypeError):
            pass

        return "string"

    def _analyze_route_patterns(
        self, df: pd.DataFrame, group_by: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Analyze route frequency patterns."""
        # Create route combinations
        routes = {}

        for _, flight in df.iterrows():
            dep = flight.get("dep_airport", flight.get("dep_airport_group", "Unknown"))
            arr = flight.get("arr_airport", flight.get("arr_airport_group", "Unknown"))
            route_key = f"{dep}-{arr}"

            if route_key not in routes:
                routes[route_key] = {
                    "route": route_key,
                    "count": 0,
                    "departure": dep,
                    "arrival": arr,
                }
            routes[route_key]["count"] += 1

        return sorted(routes.values(), key=lambda x: x["count"], reverse=True)

    def _analyze_temporal_patterns(
        self, df: pd.DataFrame, group_by: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Analyze temporal flight patterns."""
        patterns = {}

        # Use group_by column or default to month/weekday
        time_col = group_by or "month"
        if time_col not in df.columns:
            time_col = "weekday" if "weekday" in df.columns else df.columns[0]

        for _, flight in df.iterrows():
            key = flight.get(time_col, "unknown")
            if key not in patterns:
                patterns[key] = {"period": key, "count": 0, "flights": []}
            patterns[key]["count"] += 1
            patterns[key]["flights"].append(flight.to_dict())

        return sorted(patterns.values(), key=lambda x: x["count"], reverse=True)

    def _analyze_cancellation_patterns(
        self, df: pd.DataFrame, group_by: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Analyze flight cancellation patterns."""
        patterns = {}

        # Determine cancellation column
        cancel_col = "cancelled"
        if cancel_col not in df.columns:
            return [
                {
                    "group": "all",
                    "total": len(df),
                    "cancelled": 0,
                    "cancellation_rate": 0,
                }
            ]

        group_col = group_by or "all"

        for _, flight in df.iterrows():
            key = flight.get(group_col, "all") if group_col != "all" else "all"

            if key not in patterns:
                patterns[key] = {
                    "group": key,
                    "total": 0,
                    "cancelled": 0,
                    "cancellation_rate": 0,
                }

            patterns[key]["total"] += 1

            # Check if cancelled (handle various formats)
            cancelled = flight.get(cancel_col, 0)
            if cancelled in [1, "1", True, "true", "True", "yes", "Yes"]:
                patterns[key]["cancelled"] += 1

        # Calculate rates
        for pattern in patterns.values():
            if pattern["total"] > 0:
                pattern["cancellation_rate"] = (
                    pattern["cancelled"] / pattern["total"]
                ) * 100

        return sorted(
            patterns.values(), key=lambda x: x["cancellation_rate"], reverse=True
        )

    def _analyze_delay_patterns(
        self, df: pd.DataFrame, group_by: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Analyze flight delay patterns."""
        patterns = {}

        delay_col = "dep_delay"
        if delay_col not in df.columns:
            return [{"group": "all", "total_flights": len(df), "avg_delay": 0}]

        group_col = group_by or "all"

        for _, flight in df.iterrows():
            key = flight.get(group_col, "all") if group_col != "all" else "all"

            if key not in patterns:
                patterns[key] = {
                    "group": key,
                    "total_flights": 0,
                    "delayed_flights": 0,
                    "total_delay": 0,
                    "avg_delay": 0,
                    "max_delay": 0,
                }

            patterns[key]["total_flights"] += 1

            try:
                delay = float(flight.get(delay_col, 0))
                patterns[key]["total_delay"] += delay
                patterns[key]["max_delay"] = max(patterns[key]["max_delay"], delay)

                if delay > 0:
                    patterns[key]["delayed_flights"] += 1

            except (ValueError, TypeError):
                continue

        # Calculate averages
        for pattern in patterns.values():
            if pattern["total_flights"] > 0:
                pattern["avg_delay"] = pattern["total_delay"] / pattern["total_flights"]

        return sorted(patterns.values(), key=lambda x: x["avg_delay"], reverse=True)

    def _find_missing_combinations(
        self, existing: Dict[str, int], combination_type: str
    ) -> List[str]:
        """Find missing route combinations between airport groups."""
        # Standard airport groups A-G
        groups = ["A", "B", "C", "D", "E", "F", "G"]
        missing = []

        if combination_type == "bidirectional":
            # Check all bidirectional pairs
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    combo = "↔".join(sorted([groups[i], groups[j]]))
                    if combo not in existing:
                        missing.append(combo)
        else:
            # Check all directional combinations
            for dep_group in groups:
                for arr_group in groups:
                    combo = f"{dep_group}→{arr_group}"
                    if combo not in existing:
                        missing.append(combo)

        return missing
