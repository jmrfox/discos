# GenCoMo Project Plan

## Project Overview
GenCoMo enables multi-compartmental neuron simulations starting from meshes—typically derived from imaging data of biological neurons. Unlike traditional compartmental modeling codes that require the neuron structure to be a tree (a graph with no cycles) and expect the 3D structure as sequentially connected cables (usually connected at a soma), GenCoMo is designed for neurite modeling, including complex shapes and spines with no soma. Instead of skeletonization, GenCoMo segments a mesh by slicing along the z-axis at regular intervals, determines the set of closed segments from the slicing, and computes their connectivity. This produces a connectivity graph containing all morphology information, allowing for straightforward modeling of membrane potential using coupled ODEs and running dynamical simulations. This approach avoids the challenges of skeleton fitting and supports more general morphologies.

## Goals
- Provide accurate compartmental modeling using detailed neuronal meshes
- Enable flexible, mesh-based simulation and analysis workflows
- Support interactive visualization and analysis tools

## Task List
- [x] Locate and read the main README.md file
- [ ] Maintain and update this project plan file
- [ ] Implement or review core modules: MeshProcessor, ZAxisSlicer, RegionDetector, GraphBuilder, Neuron, Simulator
- [ ] Ensure robust installation and dependency management
- [ ] Expand and document demo functions and usage examples
- [ ] Develop and maintain analysis tools for simulation results

## Next Steps
- Review and update this plan as the project evolves
- Prioritize tasks based on project needs and user feedback

---

*This file should be updated regularly to reflect project status, goals, and next actions.*
