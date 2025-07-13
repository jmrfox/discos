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

from gencomo.mesh import repair_mesh, analyze_mesh


class MeshPreprocessor:
    """Handles comprehensive mesh preprocessing for neuronal data."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "volume_fixed": 0,
            "watertight_fixed": 0,
            "degenerate_removed": 0,
        }

    def log(self, message: str, level: str = "INFO"):
        """Log messages if verbose mode is enabled."""
        if self.verbose:
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "PROCESSING": "🔧"}.get(level, "📝")
            print(f"{prefix} {message}")

    def analyze_mesh_issues(self, mesh: trimesh.Trimesh, filename: str = "") -> Dict:
        """Analyze mesh for common issues."""
        analysis = analyze_mesh(mesh)
        issues = []

        # Check for negative volume
        volume = analysis.get("volume", 0)
        if volume is not None and volume < 0:
            issues.append("negative_volume")

        # Check watertightness
        if not analysis.get("is_watertight", False):
            issues.append("not_watertight")

        # Check winding consistency
        if not analysis.get("is_winding_consistent", False):
            issues.append("inconsistent_winding")

        # Check for degenerate faces
        if len(mesh.faces) > 0:
            degenerate_count = 0
            for face in mesh.faces:
                if len(np.unique(face)) < 3:
                    degenerate_count += 1
            if degenerate_count > 0:
                issues.append(f"{degenerate_count}_degenerate_faces")

        # Check for very small faces
        if hasattr(mesh, "area_faces"):
            very_small = np.sum(mesh.area_faces < 1e-10)
            if very_small > 0:
                issues.append(f"{very_small}_tiny_faces")

        return {
            "analysis": analysis,
            "issues": issues,
            "volume": volume,
            "surface_area": analysis.get("surface_area", 0),
            "num_vertices": analysis.get("num_vertices", 0),
            "num_faces": analysis.get("num_faces", 0),
        }

    def fix_negative_volume(self, mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, bool]:
        """Fix negative volume by correcting face winding order."""
        if mesh.volume >= 0:
            return mesh, False

        self.log("Fixing negative volume by correcting face winding...", "PROCESSING")

        # Create a copy to work on
        fixed_mesh = mesh.copy()

        # Method 1: Flip all face normals
        try:
            fixed_mesh.faces = np.fliplr(fixed_mesh.faces)

            # Check if this fixed the volume
            if fixed_mesh.volume > 0:
                self.log(f"Volume fixed: {mesh.volume:.3f} → {fixed_mesh.volume:.3f}", "SUCCESS")
                return fixed_mesh, True
            else:
                # If still negative, revert
                fixed_mesh.faces = np.fliplr(fixed_mesh.faces)
        except Exception as e:
            self.log(f"Face flipping failed: {e}", "WARNING")

        # Method 2: Use trimesh's fix_normals
        try:
            fixed_mesh.fix_normals()
            if fixed_mesh.volume > 0:
                self.log(f"Volume fixed with fix_normals: {mesh.volume:.3f} → {fixed_mesh.volume:.3f}", "SUCCESS")
                return fixed_mesh, True
        except Exception as e:
            self.log(f"fix_normals failed: {e}", "WARNING")

        # Method 3: Try manual mesh repair
        try:
            fixed_mesh = repair_mesh(fixed_mesh, fix_normals=True)
            if fixed_mesh.volume > 0:
                self.log(f"Volume fixed with repair_mesh: {mesh.volume:.3f} → {fixed_mesh.volume:.3f}", "SUCCESS")
                return fixed_mesh, True
        except Exception as e:
            self.log(f"repair_mesh failed: {e}", "WARNING")

        self.log("Could not fix negative volume", "WARNING")
        return mesh, False

    def preprocess_single_mesh(self, input_path: str, output_path: str) -> bool:
        """Preprocess a single mesh file."""
        try:
            self.log(f"Processing: {input_path}")
            self.stats["processed"] += 1

            # Load mesh
            mesh = trimesh.load_mesh(input_path)

            # Handle scene objects
            if isinstance(mesh, trimesh.Scene):
                geometries = list(mesh.geometry.values())
                if geometries:
                    mesh = geometries[0]
                else:
                    raise ValueError("No geometry found in mesh scene")

            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Loaded object is not a mesh: {type(mesh)}")

            # Analyze initial state
            initial_analysis = self.analyze_mesh_issues(mesh, input_path)
            self.log(f"Initial state: {len(initial_analysis['issues'])} issues found")
            for issue in initial_analysis["issues"]:
                self.log(f"  • {issue}", "WARNING")

            # Start preprocessing
            processed_mesh = mesh.copy()
            volume_fixed = False
            watertight_fixed = False

            # Step 1: Fix negative volume
            if "negative_volume" in initial_analysis["issues"]:
                processed_mesh, volume_fixed = self.fix_negative_volume(processed_mesh)
                if volume_fixed:
                    self.stats["volume_fixed"] += 1

            # Step 2: General mesh repair
            try:
                repaired_mesh = repair_mesh(
                    processed_mesh, fix_holes=True, remove_duplicates=True, fix_normals=True, remove_degenerate=True
                )

                # Check if watertightness was improved
                if not processed_mesh.is_watertight and repaired_mesh.is_watertight:
                    watertight_fixed = True
                    self.stats["watertight_fixed"] += 1

                processed_mesh = repaired_mesh

            except Exception as e:
                self.log(f"Mesh repair failed: {e}", "WARNING")

            # Step 3: Final validation
            final_analysis = self.analyze_mesh_issues(processed_mesh, input_path)

            # Save processed mesh
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            processed_mesh.export(output_path)

            # Report results
            self.log(f"Preprocessing complete:", "SUCCESS")
            self.log(f"  Volume: {initial_analysis['volume']:.3f} → {final_analysis['volume']:.3f}")
            self.log(f"  Issues: {len(initial_analysis['issues'])} → {len(final_analysis['issues'])}")
            self.log(
                f"  Watertight: {initial_analysis['analysis']['is_watertight']} → {final_analysis['analysis']['is_watertight']}"
            )
            self.log(f"  Saved to: {output_path}")

            self.stats["successful"] += 1
            return True

        except Exception as e:
            self.log(f"Failed to process {input_path}: {e}", "ERROR")
            self.stats["failed"] += 1
            return False

    def preprocess_directory(self, input_dir: str, output_dir: str, pattern: str = "*.obj") -> None:
        """Preprocess all mesh files in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            self.log(f"Input directory does not exist: {input_dir}", "ERROR")
            return

        # Find all mesh files
        mesh_files = list(input_path.glob(pattern))
        if not mesh_files:
            self.log(f"No mesh files found matching pattern: {pattern}", "WARNING")
            return

        self.log(f"Found {len(mesh_files)} mesh files to process")

        # Process each file
        for mesh_file in mesh_files:
            output_file = output_path / f"{mesh_file.stem}_processed{mesh_file.suffix}"
            self.preprocess_single_mesh(str(mesh_file), str(output_file))

    def analyze_single_mesh(self, input_path: str) -> None:
        """Analyze a single mesh file without preprocessing."""
        try:
            self.log(f"Analyzing mesh: {input_path}")

            # Load mesh
            mesh = trimesh.load_mesh(input_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = list(mesh.geometry.values())[0]

            # Analyze issues
            analysis = self.analyze_mesh_issues(mesh, Path(input_path).name)

            self.log(f"Analysis complete for {input_path}")

        except Exception as e:
            self.log(f"Failed to analyze {input_path}: {e}", "ERROR")

    def analyze_directory(self, input_dir: str, pattern: str = "*.obj") -> None:
        """Analyze all mesh files in a directory without preprocessing."""
        input_path = Path(input_dir)

        if not input_path.exists():
            self.log(f"Input directory does not exist: {input_dir}", "ERROR")
            return

        # Find all mesh files
        mesh_files = list(input_path.glob(pattern))
        if not mesh_files:
            self.log(f"No mesh files found matching pattern: {pattern}", "WARNING")
            return

        self.log(f"Found {len(mesh_files)} mesh files to analyze")

        # Analyze each file
        for mesh_file in mesh_files:
            self.analyze_single_mesh(str(mesh_file))

    def print_statistics(self):
        """Print processing statistics."""
        print("\n" + "=" * 60)
        print("📊 PREPROCESSING STATISTICS")
        print("=" * 60)
        print(f"Total files processed: {self.stats['processed']}")
        print(f"Successful: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Volume issues fixed: {self.stats['volume_fixed']}")
        print(f"Watertightness fixed: {self.stats['watertight_fixed']}")

        if self.stats["processed"] > 0:
            success_rate = (self.stats["successful"] / self.stats["processed"]) * 100
            print(f"Success rate: {success_rate:.1f}%")


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
            processor = MeshPreprocessor(verbose=True)
            processor.analyze_single_mesh(str(input_path))
        else:
            # Analyze directory
            processor = MeshPreprocessor(verbose=True)
            processor.analyze_directory(str(input_path), pattern=args.pattern)
        return

    # Create preprocessor
    processor = MeshPreprocessor(verbose=not args.quiet)

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
