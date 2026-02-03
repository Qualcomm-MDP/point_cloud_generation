import os
import numpy as np
import cv2
from PIL import Image
import shutil
from transformers import pipeline


IMAGE_PATH = "images/IMG_6878.JPG"
OUT_DIR = "out_mask2former"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
    os.mkdir(OUT_DIR)
    print(f"Directory '{OUT_DIR}' exist and cleard its contents!")
else:
    os.mkdir(OUT_DIR)
    print(f"Directory '{OUT_DIR}' did not exist and we created it now.")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID = "facebook/mask2former-swin-large-cityscapes-panoptic"
# smaller/faster model
# MODEL_ID = "facebook/mask2former-swin-small-cityscapes-panoptic"

ALPHA = 0.45  # overlay transparency
EXPORT_INSTANCE_PNGS = True  # saves mask

def overlay_panoptic(img_rgb: np.ndarray, segments, alpha=0.45, seed=0):
    rng = np.random.default_rng(seed)
    out = img_rgb.copy()

    for seg in segments:
        m = np.array(seg["mask"], dtype=np.uint8)  # 0..255
        # m = np.transpose(m)
        m = m > 0

        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        out[m] = (
            out[m].astype(np.float32) * (1 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)

    return out

def save_instance_masks(segments, out_dir):
    meta_lines = []
    for i, seg in enumerate(segments):
        label = seg["label"]
        score = float(seg.get("score", 0.0))
        mask = np.array(seg["mask"], dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8) * 255

        fname = f"{i:03d}_{label.replace(' ', '_')}_score{score:.3f}.png"
        cv2.imwrite(os.path.join(out_dir, fname), mask)
        meta_lines.append(f"{fname}\t{label}\t{score:.6f}")

    with open(os.path.join(out_dir, "segments.tsv"), "w") as f:
        f.write("file\tlabel\tscore\n")
        f.write("\n".join(meta_lines))


img = Image.open(IMAGE_PATH).convert("RGB")
img_rgb = np.array(img)

seg_pipe = pipeline("image-segmentation", model=MODEL_ID)
results = seg_pipe(img)

vis = overlay_panoptic(img_rgb, results, alpha=ALPHA)

out_overlay = os.path.join(OUT_DIR, "panoptic_overlay.png")
cv2.imwrite(out_overlay, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

if EXPORT_INSTANCE_PNGS:
    inst_dir = os.path.join(OUT_DIR, "instance_masks")
    os.makedirs(inst_dir, exist_ok=True)
    save_instance_masks(results, inst_dir)
    print("Saved instance masks to:", inst_dir)
