"""
Utility functions for DISCOS.
"""

import os
from pathlib import Path

def project_root() -> Path:
    """
    Return the root directory of the project.
    """
    return Path(__file__).parent.parent

def data_path(filename: str) -> Path:
    """
    Return the full path of the specified data file from the project data directory using pathlib.
    """
    return project_root() / "data" / filename
    
