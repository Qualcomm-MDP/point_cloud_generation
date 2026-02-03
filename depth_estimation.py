import torch
from transformers import pipeline
from PIL import Image, ImageOps
import cv2
import os
import shutil
import numpy as np

scale = 0.10

output_dir = "output_mesh"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print(f"Directory '{output_dir}' exist and cleard its contents!")
else:
    os.mkdir(output_dir)
    print(f"Directory '{output_dir}' did not exist and we created it now.")
os.makedirs(output_dir, exist_ok=True)

# Resize image in pillow and then convert to cv2 image so that we can filter it and stuff
pil_image = Image.open("images/IMG_6879.JPG")
pil_image = ImageOps.exif_transpose(pil_image)
resized_width = int(scale * pil_image.size[0]) # Pillow uses width by height, unlike openCV
resized_height = int(scale * pil_image.size[1])
pil_image = pil_image.resize((resized_width, resized_height))
image_np = np.array(pil_image)
cv2_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# Will need to filter out noise (How much to filter is still a mysetery tho, like we need to filter until me filter out the noise, but not too much to lose details and features (4 is good enough for now) )
cv2_image = cv2.bilateralFilter(cv2_image,9, 75,75)
cv2_image = cv2.bilateralFilter(cv2_image,9, 75,75)
cv2_image = cv2.bilateralFilter(cv2_image,9, 75,75)
cv2_image = cv2.bilateralFilter(cv2_image,9, 75,75)

# Save the image
cv2.imwrite("orignal_image.jpg", cv2_image)
pil_image = Image.fromarray(
    cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB) # convert back to pillow for pipeline
)

# Run the pipeline with the filtered pillow and scaled image
output_path = "predicted_depths.txt"
pipeline = pipeline(
    task="depth-estimation",
    model="Intel/zoedepth-nyu-kitti",
    dtype=torch.float16,
    device=0
)
results = pipeline(pil_image)
depth_map = results["depth"]
predicted_depths = results["predicted_depth"] # Try inverting it since everything seems inverted
predicted_depths = predicted_depths.tolist()
depth_map.save("depth_map.jpg")

# Write the predicted depth values out to somewhere so that we can use it later
with open(output_path, "w") as f:
    for row in predicted_depths:
        for col in row:
            f.write(f"{col} ")
        f.write('\n')
    f.write(f"{scale}\n")