import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

depth_image = "depth_map.jpg"
predicted_depths = "predicted_depths.txt"
colors_image = "orignal_image.jpg"

depth_vals = []
with open(predicted_depths, "r") as f:
    for line in f:
        depths = [float(x) for x in line.strip().split()]
        depth_vals.append(depths)

depth_vals = np.array(depth_vals)
depth_vals = depth_vals.astype(np.float32)

depth_img = cv2.imread(depth_image)
depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)  # convert to RGB
colors_img = cv2.imread(colors_image)
colors_img = cv2.cvtColor(colors_img, cv2.COLOR_BGR2RGB)

# depth_img = cv2.resize(depth_img, None, fx=0.01, fy=0.01)
height, width, _ = depth_img.shape

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# just get a list of all the raw pixel coordinates (0, 0), (0, 1) and so on
u, v = np.meshgrid(np.arange(width), np.arange(height))

# Camera specs
focal_length = -600  # in pixels, works much better than like mm distance for some reason, larger values tend to scale well with the small edepth values, maybe we can scale up the x and the y's then? Comes from the camera's intrinsic matrix

# Apply 214 formula, with updated focal_length
X = u * depth_vals / focal_length
Y = v * depth_vals / focal_length
Z = depth_vals * (1)

# Flatten for Open3D
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
colors_flat = colors_img.reshape(-1, 3) / 255.0

# Flatten coordinates and colors
points = np.stack((X_flat, Y_flat, Z_flat), axis=-1)

# Plot out the cloud generated with the colors
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors_flat)
pcd.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1,
        max_nn=40
    )
)
pcd.orient_normals_consistent_tangent_plane(k=50)
# generate normals is used to convert from point cloud to mesh, not totally necessary

# Visualize out point cloud
o3d.visualization.draw_geometries(
    [pcd],
    point_show_normal=False,
    window_name="Point Cloud",
    width=800,
    height=600
)

o3d.io.write_point_cloud("point_cloud.ply", pcd)
print("Saved point cloud!")

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

dec_mesh.remove_degenerate_triangles()
dec_mesh.remove_duplicated_triangles()
dec_mesh.remove_duplicated_vertices()
dec_mesh.remove_non_manifold_edges()

o3d.visualization.draw_geometries([dec_mesh], window_name="BPA Mesh Post-Processed")

o3d.io.write_triangle_mesh("cloud_mesh.ply", dec_mesh)
