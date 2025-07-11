# GenCoMo: GENeral-morphology COmpartmental MOdeling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GenCoMo is a Python package for compartmental simulation of neurons with complex morphologies using mesh-based geometric approaches. Unlike traditional neuron simulation software that relies on simplified cylinder-based models (like NEURON's SWC format), GenCoMo works directly with detailed neuronal mesh geometries to create more accurate compartmental models.

## Key Features

- **Mesh-based compartmentalization**: Work directly with neuronal meshes instead of simplified cylinder models
- **Z-stack conversion**: Convert 3D meshes to binary z-stack representations for analysis
- **Interactive visualization**: Explore z-stack slices interactively with slider controls
- **Z-axis slicing**: Automatically slice complex morphologies along the z-axis 
- **Region detection**: Identify closed regions within each slice for compartment generation
- **Graph construction**: Build connectivity between compartments across adjacent z-levels
- **Biophysical simulation**: Solve coupled ODEs for membrane potential using Hodgkin-Huxley dynamics
- **Flexible stimulation**: Support for complex stimulation protocols
- **Analysis tools**: Built-in analysis for spike detection, propagation velocity, and more
- **Demo functions**: Pre-built functions for creating test geometries (cylinders, Y-shapes, meshes with holes)

## Installation

### From source

```bash
git clone https://github.com/jmrfox/gencomo.git
cd gencomo
pip install -e .
```

### Dependencies

GenCoMo requires several scientific computing packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `numpy`, `scipy` - numerical computing
- `trimesh`, `open3d` - mesh processing  
- `scikit-image`, `opencv-python` - image processing for cross-sections
- `networkx` - graph algorithms for connectivity
- `matplotlib`, `plotly` - visualization (optional)

## Quick Start

### Basic Usage

```python
import numpy as np
from gencomo import MeshProcessor, ZAxisSlicer, RegionDetector, GraphBuilder, Neuron, Simulator

# 1. Load and process a neuronal mesh
mesh_processor = MeshProcessor()
mesh = mesh_processor.load_mesh("path/to/neuron.stl")
mesh_processor.center_mesh()
mesh_processor.align_with_z_axis()

# Alternative: Work with z-stack representations
from gencomo import mesh_to_zstack, visualize_zstack_slices
zstack = mesh_to_zstack(mesh, resolution=(100, 100, 50))
visualize_zstack_slices(zstack)  # Interactive slice viewer

# 2. Slice the mesh along z-axis
slicer = ZAxisSlicer(mesh)
slices = slicer.create_slices(num_slices=50)

# 3. Detect closed regions in each slice
region_detector = RegionDetector()
regions = region_detector.detect_regions(slices, min_area=1.0)

# 4. Build compartment connectivity graph
graph_builder = GraphBuilder()
compartment_graph = graph_builder.build_compartment_graph(regions)

# 5. Create neuron model
neuron = Neuron("my_neuron")
neuron.set_mesh(mesh)
neuron.compartment_graph = compartment_graph

# 6. Set up and run simulation
simulator = Simulator(neuron)
simulator.set_biophysics(
    capacitance=1.0,      # µF/cm²
    na_conductance=0.12,  # S/cm²
    k_conductance=0.036   # S/cm²
)

# Add stimulus
compartment_ids = list(neuron.compartment_graph.compartments.keys())
simulator.add_stimulus(
    compartment_id=compartment_ids[0],
    start_time=10.0,  # ms
    duration=1.0,     # ms  
    amplitude=5.0     # nA
)

# Run simulation
results = simulator.run_simulation(duration=100.0, dt=0.025)

if results.success:
    print("Simulation completed!")
    analysis = simulator.analyze_results()
    print(f"Total spikes: {analysis['overall']['total_spikes']}")
```

### Working with Demo Functions

GenCoMo includes built-in demo functions for creating test geometries:

```python
from gencomo import create_cylinder_mesh, create_y_shaped_mesh, create_mesh_with_hole
from gencomo import create_cylinder_zstack, visualize_zstack_3d

# Create demo meshes
cylinder = create_cylinder_mesh(radius=5.0, height=20.0)
y_shaped = create_y_shaped_mesh(radius=3.0, height=15.0, branch_angle=45.0)
holey_mesh = create_mesh_with_hole(outer_radius=8.0, inner_radius=3.0, height=10.0)

# Create demo z-stacks directly
cylinder_zstack = create_cylinder_zstack(radius=5.0, height=20.0, resolution=(64, 64, 32))
visualize_zstack_3d(cylinder_zstack)  # 3D visualization
```

### Command Line Interface

GenCoMo also provides a command-line interface:

```bash
# Process a mesh file
gencomo process neuron.stl -o model.npz --num-slices 50

# Run simulation  
gencomo simulate model.npz -o results.npz --duration 100 --stimulus comp_0 10 1 5

# Analyze results
gencomo analyze results.npz --plot
```

## Examples

See the `examples/` directory for detailed usage examples:

- `basic_example.py` - Complete workflow with a simple test mesh
- `advanced_example.py` - Advanced features with real neuronal meshes
- `z_stack_examples.py` - Working with z-stack representations and interactive visualization
- `demo_functions.py` - Using built-in demo functions to create test geometries

Run examples:
```bash
cd examples
python basic_example.py
python advanced_example.py
python z_stack_examples.py
```

## Physical Units

GenCoMo uses consistent units throughout the modeling pipeline. Understanding these units is crucial for setting appropriate parameters and interpreting results.

### Electrical Properties
- **Voltage**: mV (millivolts)
- **Current**: nA (nanoamperes) for membrane and stimulus currents
- **Conductance**: S/cm² (siemens per square centimeter) for membrane conductances
- **Resistance**: MΩ (megaohms) for axial resistances
- **Capacitance**: µF/cm² (microfarads per square centimeter) for membrane capacitance

### Geometric Properties
- **Distance**: µm (micrometers) for all spatial measurements
- **Area**: µm² (square micrometers) for membrane surface area
- **Volume**: µm³ (cubic micrometers) for compartment volumes

### Temporal Properties
- **Time**: ms (milliseconds) for simulation time steps and durations
- **Frequency**: Hz (hertz) for oscillatory phenomena

### Biophysical Constants
- **Temperature**: °C (degrees Celsius) for temperature-dependent kinetics
- **Concentration**: mM (millimolar) for ionic concentrations (when used)

### Fundamental Physics Equations

The units are consistent with the fundamental equations governing neuronal electrophysiology:

**Ohm's Law**:
$$I = G \times V \rightarrow \text{[nA]} = \text{[S/cm²]} \times \text{[cm²]} \times \text{[mV]} \rightarrow \text{[nA]} = \text{[S]} \times \text{[mV]}$$

Note: 1 S·mV = 1000 nA, so the conversion factor accounts for mV→V scaling.

**Cable Equation (Membrane)**:
$$C_m \times \frac{dV}{dt} = -I_{ion} + I_{axial} + I_{stim}$$
$$\text{[µF/cm²]} \times \text{[cm²]} \times \text{[mV/ms]} = \text{[nA]} + \text{[nA]} + \text{[nA]}$$
$$\text{[µF]} \times \text{[mV/ms]} = \text{[nA]}$$

Since 1 µF·mV/ms = 1000 nA, units are consistent.

**Axial Current (Cable Theory)**:
$$I_{axial} = \frac{V_1 - V_2}{R_{axial}} \rightarrow \text{[nA]} = \frac{\text{[mV]}}{\text{[MΩ]}}$$

Since 1 mV/MΩ = 1 nA, units check out.

**Conductance-Resistance Relationship**:
$$G = \frac{1}{R} \rightarrow \text{[S]} = \frac{1}{\text{[Ω]}} \text{ or } \text{[S/cm²]} = \frac{1}{\text{[Ω·cm²]}}$$

**Current Density**:
$$J = \frac{I}{A} \rightarrow \text{[nA/µm²]} = \frac{\text{[nA]}}{\text{[µm²]}}$$

### Unit Conversions in Code
The code handles unit conversions automatically where needed:
- Membrane areas are converted from µm² to cm² (factor of 1e-8) for conductance calculations
- Currents are scaled appropriately between S and nA units (factor of 1e9 for mV scaling)
- Spatial coordinates maintain µm units throughout mesh and z-stack operations
- Capacitance calculations use µF = 1e-6 F for proper scaling with mV and ms

### Example Parameter Values
```python
# Typical parameter ranges
simulator.set_biophysics(
    capacitance=1.0,        # µF/cm² (membrane capacitance)
    na_conductance=0.12,    # S/cm² (sodium channel density)
    k_conductance=0.036,    # S/cm² (potassium channel density)  
    leak_conductance=0.0003, # S/cm² (leak conductance)
    temperature=6.3         # °C (experimental temperature)
)

simulator.add_stimulus(
    start_time=10.0,        # ms (stimulus onset)
    duration=1.0,           # ms (stimulus duration)
    amplitude=5.0           # nA (injected current)
)
```

## Documentation

### Core Concepts

**Mesh Processing**: GenCoMo starts with a 3D mesh representing the neuronal membrane. The mesh is preprocessed (centered, aligned, smoothed) before analysis.

**Z-axis Slicing**: The mesh is sliced along the z-axis at regular intervals to create 2D cross-sections. Each slice contains contours representing the neuronal boundary. GenCoMo also supports direct z-stack representations as binary 3D arrays.

**Interactive Visualization**: Z-stack data can be explored interactively using built-in visualization functions that provide slider controls for navigating through different z-levels and comparing slices side-by-side.

**Region Detection**: Within each slice, closed regions are identified. These regions become the basis for compartments in the model.

**Graph Construction**: Connections between regions in adjacent z-levels are established based on geometric overlap or proximity.

**Compartmental Modeling**: Each region becomes a compartment with membrane area, volume, and biophysical properties. The system of coupled ODEs is solved for membrane potential dynamics.

### API Reference

#### Core Classes

- `Neuron` - Main neuron model container
- `Compartment` - Individual compartment with geometry and biophysics  
- `CompartmentGraph` - Graph of compartment connectivity
- `MeshProcessor` - Mesh loading and preprocessing
- `ZAxisSlicer` - Mesh slicing along z-axis
- `RegionDetector` - Closed region detection in slices
- `GraphBuilder` - Compartment connectivity construction
- `ODESystem` - ODE system for biophysical simulation
- `Simulator` - High-level simulation interface

#### Core Functions by Category

**Mesh Processing**:
- `visualize_mesh_3d()` - 3D mesh visualization
- `analyze_mesh_properties()` - Mesh geometry analysis

**Z-stack Operations**:
- `mesh_to_zstack()` - Convert mesh to z-stack representation
- `visualize_zstack_3d()` - 3D z-stack visualization  
- `visualize_zstack_slices()` - Interactive slice viewer with slider controls
- `compare_zstack_slices()` - Side-by-side slice comparison
- `save_zstack_data()` / `load_zstack_data()` - Z-stack I/O operations
- `analyze_zstack_properties()` - Z-stack analysis

**Demo Functions**:
- `create_cylinder_mesh()`, `create_y_shaped_mesh()`, `create_mesh_with_hole()` - Test geometry creation
- `create_cylinder_zstack()`, `create_y_shaped_zstack()`, `create_hole_zstack()` - Test z-stack creation
- `save_test_meshes()`, `save_test_zstacks()` - Save demo data to disk

#### Key Parameters

**Slicing Parameters**:
- `num_slices` or `slice_spacing` - Resolution of z-axis discretization
- `z_min`, `z_max` - Range for slicing

**Region Detection**:
- `min_area` - Minimum area threshold for valid regions
- `hole_detection` - Whether to detect holes within regions

**Connectivity**:
- `connection_method` - 'overlap', 'distance', or 'hybrid'
- `min_overlap_ratio` - Minimum overlap for connections
- `max_connection_distance` - Maximum distance for connections

**Biophysics**:
- `capacitance` - Membrane capacitance (µF/cm²)
- `na_conductance`, `k_conductance` - Ionic conductances (S/cm²)
- `temperature` - Temperature for rate constants (°C)

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/jmrfox/gencomo.git
cd gencomo
pip install -e ".[dev]"
```

### Code Style

We use `black` for code formatting and `flake8` for linting:

```bash
black gencomo/
flake8 gencomo/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use GenCoMo in your research, please cite:

```bibtex
@software{gencomo,
  title={GenCoMo: GENeral-morphology COmpartmental MOdeling},
  author={Fox, Jordan},
  year={2025},
  url={https://github.com/jmrfox/gencomo}
}
```

## Comparison with Other Tools

| Feature                   | GenCoMo | NEURON | Brian2 | Arbor |
| ------------------------- | ------- | ------ | ------ | ----- |
| Mesh-based morphology     | ✅       | ❌      | ❌      | ❌     |
| Z-stack representation    | ✅       | ❌      | ❌      | ❌     |
| Interactive visualization | ✅       | ⚠️      | ⚠️      | ⚠️     |
| SWC support               | 🔲       | ✅      | ✅      | ✅     |
| Complex geometry          | ✅       | ⚠️      | ⚠️      | ⚠️     |
| Python native             | ✅       | ⚠️      | ✅      | ⚠️     |
| Demo functions            | ✅       | ⚠️      | ⚠️      | ❌     |
| GPU acceleration          | 🔲       | ❌      | ⚠️      | ✅     |

✅ Full support, ⚠️ Partial support, ❌ Not supported, 🔲 Planned

## Acknowledgments

- Inspired by compartmental modeling approaches in NEURON and Brian2
- Mesh processing powered by trimesh and Open3D
- Scientific computing with NumPy, SciPy, and NetworkX

## Support

- **Documentation**: [GitHub Wiki](https://github.com/jmrfox/gencomo/wiki)
- **Issues**: [GitHub Issues](https://github.com/jmrfox/gencomo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jmrfox/gencomo/discussions)
