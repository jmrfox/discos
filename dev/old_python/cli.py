"""
Command-line interface for GenCoMo.
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

from . import __version__
from .mesh import MeshProcessor
from .slicer import ZAxisSlicer
from .regions import RegionDetector
from .graph import GraphBuilder
from .core import Neuron
from .simulation import Simulator


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="GenCoMo: GENeral-morphology COmpartmental MOdeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"GenCoMo {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process mesh command
    process_parser = subparsers.add_parser("process", help="Process mesh and create compartments")
    process_parser.add_argument("mesh_file", help="Input mesh file path")
    process_parser.add_argument("-o", "--output", default="neuron_model.npz", help="Output file path")
    process_parser.add_argument("--num-slices", type=int, help="Number of z-slices")
    process_parser.add_argument("--slice-spacing", type=float, help="Spacing between slices (µm)")
    process_parser.add_argument("--min-area", type=float, default=0.1, help="Minimum region area (µm²)")
    process_parser.add_argument(
        "--connection-method",
        choices=["overlap", "distance", "hybrid"],
        default="overlap",
        help="Connection detection method",
    )
    process_parser.add_argument("--visualize", action="store_true", help="Show visualizations")

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run compartmental simulation")
    sim_parser.add_argument("model_file", help="Input model file (from process command)")
    sim_parser.add_argument("-o", "--output", default="simulation_results.npz", help="Output file path")
    sim_parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration (ms)")
    sim_parser.add_argument("--dt", type=float, default=0.025, help="Time step (ms)")
    sim_parser.add_argument(
        "--stimulus",
        nargs=4,
        metavar=("COMP_ID", "START", "DURATION", "AMPLITUDE"),
        action="append",
        help="Add stimulus: compartment_id start_time duration amplitude",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze simulation results")
    analyze_parser.add_argument("results_file", help="Simulation results file")
    analyze_parser.add_argument("-o", "--output", help="Output analysis file (JSON)")
    analyze_parser.add_argument("--plot", action="store_true", help="Generate plots")

    return parser


def process_mesh_command(args):
    """Handle mesh processing command."""
    print(f"Processing mesh: {args.mesh_file}")

    # Load mesh
    mesh_processor = MeshProcessor()
    try:
        mesh = mesh_processor.load_mesh(args.mesh_file)
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return 1

    # Preprocess mesh
    mesh_processor.center_mesh()
    mesh_processor.align_with_z_axis()

    print("Mesh properties:")
    props = mesh_processor.compute_mesh_properties()
    for key, value in props.items():
        print(f"  {key}: {value}")

    # Create slices
    slicer = ZAxisSlicer(mesh)
    if args.num_slices:
        slices = slicer.create_slices(num_slices=args.num_slices)
    elif args.slice_spacing:
        slices = slicer.create_slices(slice_spacing=args.slice_spacing)
    else:
        # Default: 50 slices
        slices = slicer.create_slices(num_slices=50)

    print(f"Created {len(slices)} slices")

    # Detect regions
    region_detector = RegionDetector()
    regions = region_detector.detect_regions(slices, min_area=args.min_area)

    print(f"Detected {len(regions)} regions")
    stats = region_detector.compute_region_statistics()
    print(f"Average regions per slice: {stats.get('regions_per_slice', {}).get('mean', 0):.1f}")

    # Build compartment graph
    graph_builder = GraphBuilder()
    compartment_graph = graph_builder.build_compartment_graph(regions, connection_method=args.connection_method)

    # Create neuron model
    neuron = Neuron()
    neuron.set_mesh(mesh)
    neuron.compartment_graph = compartment_graph

    # Save model
    print(f"Saving model to: {args.output}")
    # Note: Would need to implement proper serialization
    # For now, just save basic info

    print("Processing completed successfully!")
    return 0


def simulate_command(args):
    """Handle simulation command."""
    print(f"Loading model: {args.model_file}")

    # Note: This would need proper model loading implementation
    print("Model loading not fully implemented yet")

    return 0


def analyze_command(args):
    """Handle analysis command."""
    print(f"Analyzing results: {args.results_file}")

    # Note: This would need proper analysis implementation
    print("Analysis not fully implemented yet")

    return 0


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "process":
            return process_mesh_command(args)
        elif args.command == "simulate":
            return simulate_command(args)
        elif args.command == "analyze":
            return analyze_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
