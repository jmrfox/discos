"""
Test configuration for GenCoMo.
"""

import sys
from pathlib import Path

# Add the package to the path for testing
package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(package_dir))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_DATA_DIR.mkdir(exist_ok=True)
