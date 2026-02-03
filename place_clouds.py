import trimesh
import os

mesh_dir = "output_mesh/"

combined_mesh = []
for file in os.listdir(mesh_dir):
    mesh_path = mesh_dir + file
    cloud = trimesh.load_mesh(mesh_path)
    combined_mesh.append(cloud)

combined_mesh.pop(0)
combined_mesh = trimesh.util.concatenate(combined_mesh)
combined_mesh.export("combined.glb")
