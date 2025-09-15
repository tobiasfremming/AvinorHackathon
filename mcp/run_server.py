#!/usr/bin/env python3
"""
Avinor MCP Server Launcher
Entry point for running the Avinor Flight Data MCP Server
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the src directory to the Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from server import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 MCP Server stopped by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"❌ MCP Server error: {e}", file=sys.stderr)
        sys.exit(1)
