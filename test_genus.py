import trimesh
from gencomo.mesh.utils import analyze_mesh, print_mesh_analysis

# Create test shapes
cylinder = trimesh.primitives.Cylinder()
sphere = trimesh.primitives.Sphere()
# Create a torus-like shape by creating a hollow cylinder
torus = trimesh.primitives.Cylinder(radius=1.0, height=0.2, sections=32)
torus_inner = trimesh.primitives.Cylinder(radius=0.5, height=0.2, sections=32)
torus_inner.apply_translation([0, 0, 0])
torus = trimesh.boolean.difference([torus, torus_inner])

print("===== CYLINDER =====")
print(f"Built-in trimesh: Euler number = {cylinder.euler_number}")
analysis = analyze_mesh(cylinder)
print(f"Our analyze_mesh: Euler characteristic = {analysis['euler_characteristic']}, Genus = {analysis['genus']}")
print("\nDetailed analysis:")
print_mesh_analysis(cylinder)

print("\n\n===== SPHERE =====")
print(f"Built-in trimesh: Euler number = {sphere.euler_number}")
analysis = analyze_mesh(sphere)
print(f"Our analyze_mesh: Euler characteristic = {analysis['euler_characteristic']}, Genus = {analysis['genus']}")
print("\nDetailed analysis:")
print_mesh_analysis(sphere)

print("\n\n===== TORUS-LIKE SHAPE =====")
print(f"Built-in trimesh: Euler number = {torus.euler_number}")
analysis = analyze_mesh(torus)
print(f"Our analyze_mesh: Euler characteristic = {analysis['euler_characteristic']}, Genus = {analysis['genus']}")
print("\nDetailed analysis:")
print_mesh_analysis(torus)
