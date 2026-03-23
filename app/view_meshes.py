import os
import json
import numpy as np
import cv2
import trimesh

IMAGE_PATH  = "input/building.jpg"
OUTPUT_DIR  = "output"
MASKS_DIR   = os.path.join(OUTPUT_DIR, "masks")
MESH_DIR    = os.path.join(OUTPUT_DIR, "per_class")
JSON_PATH   = os.path.join(OUTPUT_DIR, "sionna_scene.json")
COMBINED    = os.path.join(OUTPUT_DIR, "combined_scene.obj")

# Class colours for overlay 
CLASS_COLORS = {
    "window":        (0,   200, 255),   # cyan
    "glass_window":  (0,   200, 255),
    "door":          (255, 140,   0),   # orange
    "brick_wall":    (180,  80,  40),   # brown
    "concrete_wall": (160, 160, 160),   # grey
}
DEFAULT_COLOR = (80, 255, 80)           # green


def color_for(filename: str):
    for key, col in CLASS_COLORS.items():
        if key in filename:
            return col
    return DEFAULT_COLOR


# ── 1. Mask overlay ────────────────────────────────────────────────────────
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"⚠  Could not load {IMAGE_PATH}")
else:
    overlay = image.copy()
    mask_files = sorted(f for f in os.listdir(MASKS_DIR) if f.endswith(".png"))

    for mf in mask_files:
        mask = cv2.imread(os.path.join(MASKS_DIR, mf), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        color  = color_for(mf)
        region = overlay.copy()
        region[mask > 128] = color
        overlay = cv2.addWeighted(overlay, 0.55, region, 0.45, 0)

        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

    overlay_path = os.path.join(OUTPUT_DIR, "mask_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"Mask overlay → {overlay_path}")

# ── 2. Scale summary ───────────────────────────────────────────────────────
if os.path.exists(JSON_PATH):
    with open(JSON_PATH) as f:
        meta = json.load(f)

    px2m = meta["px_to_meter"]
    print(f"\n{'─'*50}")
    print(f"  {meta['scene_name']}")
    print(f"  Scale  : {px2m*100:.3f} cm / pixel")
    print(f"  Wall H : {meta['wall_height_m']} m")
    print(f"  Objects: {len(meta['objects'])}")
    print(f"{'─'*50}")

    by_class: dict[str, dict] = {}
    for obj in meta["objects"]:
        lbl = obj["label"]
        if lbl not in by_class:
            by_class[lbl] = {"count": 0, "total_m2": 0.0, "material": obj["sionna_material"]}
        by_class[lbl]["count"]    += 1
        by_class[lbl]["total_m2"] += obj["area_m2"]

    print(f"\n  {'Label':<22} {'Count':>6} {'Area (m²)':>12} {'Sionna Mat':<20}")
    print(f"  {'─'*22} {'─'*6} {'─'*12} {'─'*20}")
    for lbl, info in sorted(by_class.items()):
        print(f"  {lbl:<22} {info['count']:>6} {info['total_m2']:>12.2f} {info['material']:<20}")

# ── 3. 3-D viewer ─────────────────────────────────────────────────────────
meshes = []
colors = []

if os.path.exists(MESH_DIR):
    for fname in sorted(os.listdir(MESH_DIR)):
        if not fname.endswith(".obj"):
            continue
        path  = os.path.join(MESH_DIR, fname)
        mesh  = trimesh.load(path, force="mesh")
        col   = list(color_for(fname)) + [200]   # RGBA
        mesh.visual.face_colors = col
        meshes.append(mesh)

if not meshes and os.path.exists(COMBINED):
    print(f"\nLoading combined scene from {COMBINED}")
    meshes = [trimesh.load(COMBINED, force="mesh")]

if meshes:
    scene = trimesh.Scene(meshes)
    b = scene.bounds
    d = b[1] - b[0]
    print(f"\n3-D scene extents:")
    print(f"  X: {b[0,0]:.2f} → {b[1,0]:.2f}  ({d[0]:.2f} m)")
    print(f"  Y: {b[0,1]:.2f} → {b[1,1]:.2f}  ({d[1]:.2f} m)")
    print(f"  Z: {b[0,2]:.2f} → {b[1,2]:.2f}  ({d[2]:.2f} m)")
    scene.show()
else:
    print("No meshes to display.")