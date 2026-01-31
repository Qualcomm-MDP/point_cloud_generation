# point_cloud_generation

/bin:
- Contains the bash script to run the pipeline

# Depth Map
Utilize a machine learning model zoedepth-nyu-kitti found from Huggingface to calculate the depth per pixel of the image.
Utilize perspective projection that was developed in MATH 214 to transform the pixels to their perspective depths. An additional
step to convert the point cloud into a mesh was implemented by using the Ball Pivoting Algorithm in Open3D.

An example input and output are shown below:

Input: 


<img width="300" alt="IMG_5906" src="https://github.com/user-attachments/assets/c32393e1-8e8c-4e8c-a004-a57f519b2a57" />

Outputs:

| | |
|---|---|
| <img width="271" height="243" src="https://github.com/user-attachments/assets/c9c58576-d56b-4c57-bc2b-cb9f9b8eea35" /> | <img width="386" height="355" src="https://github.com/user-attachments/assets/a8d03268-1e15-47fe-8da3-b73de0e8d2a1" /> |



