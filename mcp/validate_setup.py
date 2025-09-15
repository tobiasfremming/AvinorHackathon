#!/usr/bin/env python3
"""
Avinor MCP Server Validation Script
Validates that the MCP server is properly configured and ready for use with GitHub Copilot
"""

import sys
import os
import json
from pathlib import Path


def main():
    print("🔍 Avinor MCP Server Validation")
    print("=" * 40)

    # Check current directory
    mcp_dir = Path(__file__).parent
    print(f"📍 MCP Directory: {mcp_dir.absolute()}")

    # Check Python version
    python_version = sys.version_info
    print(
        f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}"
    )

    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False

    # Check required files
    required_files = [
        "run_server.py",
        "src/server.py",
        "src/analyzer.py",
        "copilot_config.json",
    ]

    missing_files = []
    for file_path in required_files:
        if not (mcp_dir / file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files present")

    # Check dependencies
    try:
        import pandas
        import numpy
        from mcp.server import Server

        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

    # Test server import
    try:
        sys.path.append(str(mcp_dir / "src"))
        from server import AvinorFlightDataServer

        print("✅ Server imports successfully")
    except Exception as e:
        print(f"❌ Server import failed: {e}")
        return False

    # Read and validate configuration
    try:
        config_path = mcp_dir / "copilot_config.json"
        with open(config_path) as f:
            config = json.load(f)

        if "mcpServers" in config and "avinor-flight-data" in config["mcpServers"]:
            server_config = config["mcpServers"]["avinor-flight-data"]
            print("✅ Copilot configuration valid")
            print(f"   Command: {server_config.get('command', 'N/A')}")
            print(f"   Args: {server_config.get('args', [])}")
        else:
            print("❌ Invalid Copilot configuration")
            return False

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

    # Final status
    print("\n🎉 MCP Server Validation Complete!")
    print("✅ Ready for GitHub Copilot integration")
    print("\n📋 Next Steps:")
    print("1. Copy the configuration from copilot_config.json")
    print("2. Add it to your Copilot MCP settings")
    print("3. Restart your Copilot client")
    print("4. Test with: 'Use the avinor-flight-data MCP server to analyze CSV files'")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
