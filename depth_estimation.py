import torch
from transformers import pipeline
from PIL import Image, ImageOps

scale = 0.15

image = Image.open("IMG_6870.JPG")
image = ImageOps.exif_transpose(image)
resized_width = int(scale * image.size[0]) # Pillow uses width by height, unlike openCV
resized_height = int(scale * image.size[1])
image = image.resize((resized_width, resized_height))
image.save("orignal_image.jpg")
output_path = "predicted_depths.txt"
pipeline = pipeline(
    task="depth-estimation",
    model="Intel/zoedepth-nyu-kitti",
    dtype=torch.float16,
    device=0
)
results = pipeline(image)
depth_map = results["depth"]
predicted_depths = results["predicted_depth"] # Try inverting it since everything seems inverted
predicted_depths = predicted_depths.tolist()
depth_map.save("depth_map.jpg")

with open(output_path, "w") as f:
    for row in predicted_depths:
        for col in row:
            f.write(f"{col} ")
        f.write('\n')