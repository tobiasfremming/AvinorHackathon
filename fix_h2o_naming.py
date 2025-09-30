#!/usr/bin/env python3
"""
Quick script to clean up any naming conflicts and ensure H2O can be imported properly.
This removes any compiled Python files that might cause import issues.
"""

import os
import sys
from pathlib import Path

def clean_python_cache():
    """Remove all __pycache__ directories and .pyc files."""
    current_dir = Path(".")

    # Remove __pycache__ directories
    for pycache_dir in current_dir.rglob("__pycache__"):
        print(f"Removing {pycache_dir}")
        for file in pycache_dir.iterdir():
            file.unlink()
        pycache_dir.rmdir()

    # Remove .pyc files
    for pyc_file in current_dir.rglob("*.pyc"):
        print(f"Removing {pyc_file}")
        pyc_file.unlink()

    print("Python cache cleanup complete!")

def check_h2o_import():
    """Test if H2O can be imported without conflicts."""
    try:
        import h2o
        print(f"✓ H2O import successful - version: {h2o.__version__}")
        return True
    except Exception as e:
        print(f"✗ H2O import failed: {e}")
        return False

def main():
    print("Cleaning up Python cache to fix H2O import issues...")
    clean_python_cache()

    print("\nTesting H2O import...")
    if check_h2o_import():
        print("✓ H2O is ready to use!")
    else:
        print("✗ H2O import still has issues. Check your installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()