"""
Utility functions for DISCOS.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def project_root() -> Path:
    """
    Return the root directory of the project.
    """
    root = Path(__file__).parent.parent
    # Not noisy by default; keep as debug
    logger.debug("Project root resolved to %s", root)
    return root


def data_path(filename: str) -> Path:
    """
    Return the full path of the specified data file from the project data directory using pathlib.
    """
    p = project_root() / "data" / filename
    logger.debug("Data path for '%s' -> %s", filename, p)
    return p


def add_project_dir_to_path():
    """
    Add the project directory to the Python path.
    """
    pr = project_root()
    sys.path.insert(0, str(pr))
    logger.debug("Inserted project root into sys.path: %s", pr)
