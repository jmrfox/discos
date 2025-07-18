import trimesh
from gencomo.mesh.utils import analyze_mesh, print_mesh_analysis

# Create test shapes
cylinder = trimesh.primitives.Cylinder()
sphere = trimesh.primitives.Sphere()

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
