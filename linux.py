import json
import numpy as np
import open3d as o3d
import subprocess
import os

JSON_PATH = "m_out.json"
OUT_DIR = "./out/"

def get_clouds_from_server(json_path):
    data_cloud = None
    with open(json_path, "r") as f:
        data_cloud = json.load(f)

    data_cloud = data_cloud[1]["mapillary"]["data"]
    counter = 0
    for element in data_cloud:
        counter += 1
        if counter >= 207:
            break
        # element = element["mapillary"]
        elid = element["id"]
        if not element.get("sfm_cluster", {}).get("url"):
            continue
        elurl = element["sfm_cluster"]["url"]
        file_name = "sfm_cluster" + str(elid)
        out_name = "out" + str(elid) + ".json"
        print(file_name)

        subprocess.run(
            ["curl", elurl, "-o", f"./tmp/{file_name}"],
        )

        with open(f"./tmp/{file_name}", "rb") as fin, \
            open(f"./out/{out_name}", "w") as fout:

            subprocess.run(
                ["python3", "decompress.py"],
                stdin=fin,
                stdout=fout,
                check=True
            )

def read_in_clouds(json_path):
    with open(json_path, "r") as f:
        data_cloud = json.load(f)

    point_colors = []
    point_locations = []
    for point in data_cloud[0]["points"]:
        point_colors.append(data_cloud[0]["points"][point]["color"])
        point_locations.append(data_cloud[0]["points"][point]["coordinates"])

    point_colors = np.array(point_colors) / 255.0
    point_locations = np.array(point_locations)

    # # Plot out the cloud generated with the colors
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_locations)
    # pcd.colors = o3d.utility.Vector3dVector(point_colors)

    # # Visualize out point cloud
    # o3d.visualization.draw_geometries(
    #     [pcd],
    #     point_show_normal=False,
    #     window_name="Point Cloud",
    #     width=800,
    #     height=600
    # )

    return point_locations, point_colors

def main():
    # get_clouds_from_server(JSON_PATH)

    all_points = []
    all_colors = []
    for file in os.listdir(OUT_DIR):
        print(file)
        points, colors = read_in_clouds(OUT_DIR + file)
        print(points.shape)
        all_points.append(points)
        all_colors.append(colors)

    print("Finished merging clouds")
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    # Plot out the cloud generated with the colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_points)
    pcd.colors = o3d.utility.Vector3dVector(merged_colors)

    # pcd.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(
    #         radius=0.1,
    #         max_nn=10
    #     )
    # )
    # pcd.orient_normals_consistent_tangent_plane(k=10)

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

    # # Generate a mesh using the ball pivoting algorithm
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 3 * avg_dist

    # bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

    # dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

    # dec_mesh.remove_degenerate_triangles()
    # dec_mesh.remove_duplicated_triangles()
    # dec_mesh.remove_duplicated_vertices()
    # dec_mesh.remove_non_manifold_edges()

    # o3d.visualization.draw_geometries([dec_mesh], window_name="BPA Mesh Post-Processed")

    # o3d.io.write_triangle_mesh("output_mesh/cloud_mesh.ply", dec_mesh)

if __name__ == "__main__":
    main()