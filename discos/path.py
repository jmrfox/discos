"""
Utility functions for DISCOS.
"""

import os
import sys
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


def add_project_dir_to_path():
    """
    Add the project directory to the Python path.
    """
    sys.path.insert(0, str(project_root()))
