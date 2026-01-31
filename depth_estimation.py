import torch
from transformers import pipeline
from PIL import Image, ImageOps
import cv2
import numpy as np

scale = 0.10

pil_image = Image.open("IMG_6869.JPG")
pil_image = ImageOps.exif_transpose(pil_image)
resized_width = int(scale * pil_image.size[0]) # Pillow uses width by height, unlike openCV
resized_height = int(scale * pil_image.size[1])
pil_image = pil_image.resize((resized_width, resized_height))
image_np = np.array(pil_image)
cv2_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
cv2_image = cv2.bilateralFilter(cv2_image,5,450,450)
cv2.imwrite("orignal_image.jpg", cv2_image)
pil_image = Image.fromarray(
    cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
)

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

with open(output_path, "w") as f:
    for row in predicted_depths:
        for col in row:
            f.write(f"{col} ")
        f.write('\n')