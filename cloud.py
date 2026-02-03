import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageOps
import open3d as o3d

# Define relevant image paths that we need
depth_image = "depth_map.jpg"
predicted_depths = "predicted_depths.txt"
colors_image = "orignal_image.jpg"
instance_masks_path = "out_mask2former/instance_masks/"

# Read in the scale as well as the depths that we have from the previous stage
scale = 0
depth_vals = []
with open(predicted_depths, "r") as f:
    for line in f:
        values = line.strip().split()
        if len(values) == 1:
            scale = float(values[0])
            break
        depths = [float(x) for x in line.strip().split()]
        depth_vals.append(depths)

# # read in the depth image (if we decide to use the depth image instead of the predicted depths) and colors image
# depth_img = cv2.imread(depth_image)
# depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)  # convert to RGB

# # If we want to use the depth map image as the basis of our new depths
# depth_vals = []
# for i in range(height):
#     row = []
#     for j in range(width):
#         row.append(np.mean(depth_img[i][j]))
#     depth_vals.append(row)

# Convert the depth_vals into a numpy array for vectorized calculations later
depth_vals = np.array(depth_vals)
depth_vals = depth_vals 

colors_img = cv2.imread(colors_image)
colors_img = cv2.cvtColor(colors_img, cv2.COLOR_BGR2RGB)

height, width, _ = colors_img.shape
sky_path = ""

for file in os.listdir(instance_masks_path):
    file_path = instance_masks_path + file
    components = file.strip().split("_")
    if components[1] == "sky":
        sky_path = file_path
        break

# Get a list of which coordinates are in the sky, if the model was able to perform a sky segmentation
sky_coordinates = []
if sky_path != "":
    # Read sky segmented image into a mask
    mask_image = cv2.imread(sky_path)
    mask_image = cv2.resize(mask_image, (width, height))
    cv2.imwrite("sky_mask.jpg", mask_image)

    # Find the sky coordinates
    height, width, _ = mask_image.shape
    for i in range(mask_image.shape[0]):
        for j in range(mask_image.shape[1]):
            if mask_image[i][j][0] == 255:
                sky_coordinates.append((j, i))
    sky_set = set(sky_coordinates) # Convert to a set for O(1) lookup, faster than a list
else:
    sky_set = set(sky_coordinates) # If there was no segmentation performed, just leave that set empty

# Get the center of our image, where we center our point cloud on
cx = width / 2
cy = height / 2

print(height, width)
# just get a list of all the raw pixel coordinates, np array of al the x coordinates and all the y coordinates
u, v = np.meshgrid(np.arange(width), np.arange(height))

# Camera specs
focal_length = 300  # in pixels, works much better than like mm distance for some reason, larger values tend to scale well with the small edepth values, maybe we can scale up the x and the y's then? Comes from the camera's intrinsic matrix

# Apply 214 formula, with updated focal_length
X = -(u - cx) * depth_vals / focal_length
Y = -(v - cy) * depth_vals / focal_length
Z = depth_vals

# Transform the sky coordinates so that we can omit them from the point cloud
sky_transformed_points = []
for coord in sky_coordinates:
    x, y = coord
    # Perform the transformation and store the results
    transformed_x = -(x - cx) * depth_vals[y][x] / focal_length
    transformed_y = -(y - cy) * depth_vals[y][x] / focal_length
    sky_transformed_points.append((transformed_x, transformed_y))

sky_transformed_points_set = set(sky_transformed_points) # again convert to set for quick operations

# Flatten for Open3D
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
colors_flat = colors_img.reshape(-1, 3) / 255.0

# Flatten coordinates and colors
points = np.stack((X_flat, Y_flat, Z_flat), axis=-1)

# Get a list of the indices that we would like to remove since the points are flat right now
indices_to_remove = []
for i, point in enumerate(points):
    x = point[0]
    y = point[1]
    pixel = (x, y)
    if pixel in sky_transformed_points_set:
        indices_to_remove.append(i)

# Delete the points and the colors from the point cloud
points_filtered = np.delete(points, indices_to_remove, axis=0)
colors_filtered = np.delete(colors_flat, indices_to_remove, axis=0)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot out the cloud generated with the colors
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_filtered)
pcd.colors = o3d.utility.Vector3dVector(colors_filtered)
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

o3d.io.write_point_cloud("output_mesh/point_cloud.ply", pcd)
print("Saved point cloud!")

# Generate a mesh using the ball pivoting algorithm
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

o3d.io.write_triangle_mesh("output_mesh/cloud_mesh.ply", dec_mesh)
