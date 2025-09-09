# Make demo meshes for cylinder, torus, and "neuron" and save them in data/mesh/demos

import os
from pathlib import Path

from discos import create_cylinder_mesh, create_demo_neuron_mesh, create_torus_mesh
from discos.path import data_path


def main() -> None:
    out_dir = data_path("mesh/demo")
    os.makedirs(out_dir, exist_ok=True)

    # Cylinder
    cylinder = create_cylinder_mesh(length=100.0, radius=5.0, resolution=16)
    cylinder_path = Path(out_dir) / "cylinder.obj"
    cylinder.export(str(cylinder_path))

    # Torus
    torus = create_torus_mesh(major_radius=20.0, minor_radius=5.0)
    torus_path = Path(out_dir) / "torus.obj"
    torus.export(str(torus_path))

    # Neuron
    neuron = create_demo_neuron_mesh()
    neuron_path = Path(out_dir) / "neuron.obj"
    neuron.export(str(neuron_path))

    print("Demo meshes saved:")
    print(f"  cylinder: {cylinder_path}")
    print(f"  torus:    {torus_path}")
    print(f"  neuron:   {neuron_path}")


if __name__ == "__main__":
    main()
