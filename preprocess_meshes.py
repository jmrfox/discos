#!/usr/bin/env python3
"""
Mesh Preprocessing Script for GenCoMo

This script preprocesses neuron mesh data to fix common issues:
- Negative volumes (incorrect face winding)
- Non-watertight meshes
- Degenerate faces
- Duplicate faces/vertices
- Poor mesh quality

Usage:
    python preprocess_meshes.py [input] [output] [options]

Examples:
    python preprocess_meshes.py data/ processed_data/
    python preprocess_meshes.py data/TS2_alone.obj processed_data/TS2_processed.obj
    python preprocess_meshes.py data/TS2_alone.obj --analyze-only
"""

import os
import sys
import argparse
import numpy as np
import trimesh
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings

# Add the gencomo package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from gencomo.mesh import MeshProcessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess neuron mesh data for GenCoMo")
    parser.add_argument("input", help="Input file or directory")
    parser.add_argument("output", nargs="?", help="Output file or directory")
    parser.add_argument("--pattern", default="*.obj", help="File pattern for directory processing (default: *.obj)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze the mesh without preprocessing")

    args = parser.parse_args()

    # Check if output is required
    if not args.analyze_only and not args.output:
        parser.error("output argument is required unless using --analyze-only")

    # Check if input is file or directory
    input_path = Path(args.input)

    if args.analyze_only:
        if input_path.is_file():
            # Create simple analysis report
            processor = MeshProcessor(verbose=True)
            processor.analyze_single_mesh(str(input_path))
        else:
            # Analyze directory
            processor = MeshProcessor(verbose=True)
            processor.analyze_directory(str(input_path), pattern=args.pattern)
        return

    # Create preprocessor
    processor = MeshProcessor(verbose=not args.quiet)

    if input_path.is_file():
        # Single file processing
        processor.preprocess_single_mesh(args.input, args.output)
    else:
        # Directory processing
        processor.preprocess_directory(args.input, args.output, pattern=args.pattern)

    # Print statistics
    if not args.quiet:
        processor.print_statistics()


if __name__ == "__main__":
    main()
