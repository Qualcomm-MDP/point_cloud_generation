import os
import shutil

OUT_DIR = "out_mask2former"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
    os.mkdir(OUT_DIR)
    print(f"Directory '{OUT_DIR}' exist and cleard its contents!")
else:
    os.mkdir(OUT_DIR)
    print(f"Directory '{OUT_DIR}' did not exist and we created it now.")
os.makedirs(OUT_DIR, exist_ok=True)

output_dir = "output_mesh"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    print(f"Directory '{output_dir}' exist and cleard its contents!")
else:
    os.mkdir(output_dir)
    print(f"Directory '{output_dir}' did not exist and we created it now.")
os.makedirs(output_dir, exist_ok=True)