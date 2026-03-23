"""
Facade Segmentation Pipeline — windows and doors only, building surfaces only.
"""

import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from skimage import measure, morphology
from shapely.geometry import Polygon
from shapely.ops import unary_union
import trimesh

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
IMAGE_PATH     = "input/building.jpg"
SAM_CHECKPOINT = "checkpoints/sam_vit_h_4b8939.pth"
OUTPUT_DIR     = "output"
MODEL_ID       = "IDEA-Research/grounding-dino-base"

BOX_THRESHOLD  = 0.22
TEXT_THRESHOLD = 0.18

USE_TILING    = True
TILE_SIZE     = 1024
TILE_OVERLAP  = 256
NMS_IOU_THRESH = 0.4

# ── Crop: ignore bottom N% of image (parking lot, cars, ground) ──────────────
# Set to 0.0 to disable. 0.35 means ignore bottom 35% of image.
CROP_BOTTOM_FRACTION = 0.30   # lower floor windows above garage
CROP_TOP_FRACTION    = 0.05   # just sky — brick top row starts ~10% down
CROP_LEFT_FRACTION   = 0.27   # exclude glass atrium + tree
CROP_RIGHT_FRACTION  = 0.88   # exclude concrete stairwell block on right (keep left 88%)

# ── Box size limits as fraction of image area ─────────────────────────────────
MIN_BOX_AREA_FRACTION = 0.0005
MAX_BOX_AREA_FRACTION = 0.02

# ── Aspect ratio (width/height) ───────────────────────────────────────────────
MIN_ASPECT = 0.4
MAX_ASPECT = 2.5

# ── Only keep these classes — everything else is discarded ───────────────────
KEEP_CLASSES = {
    "window",
    "glass window",
    "window frame",
    "window pane",
    "door",
    "building window",
    "building door",
}

# Focused prompts — NO walls, facades, cars, trees, ground
TEXT_LABELS = [[
    "window",
    "glass window",
    "window frame",
    "window pane",
    "door",
    "building window",
    "building door",
]]

KNOWN_WALL_HEIGHT_M = 15.0   # ~5 storey building visible in photo

EXTRUDE_DEPTH_M = {
    "window":         0.05,
    "glass window":   0.05,
    "window frame":   0.08,
    "window pane":    0.05,
    "building window":0.05,
    "door":           0.10,
    "building door":  0.10,
    "default":        0.05,
}

SIONNA_MATERIAL = {
    "window":         "itu_glass",
    "glass window":   "itu_glass",
    "window frame":   "itu_concrete",
    "window pane":    "itu_glass",
    "building window":"itu_glass",
    "door":           "itu_wood",
    "building door":  "itu_wood",
    "default":        "itu_glass",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for sub in ["masks", "meshes", "per_class"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def compute_pixel_to_meter(image_h_px, known_height_m):
    return known_height_m / image_h_px

def clean_binary_mask(mask, min_area=300):
    mask = mask.astype(bool)
    mask = morphology.remove_small_objects(mask, min_size=min_area)
    mask = morphology.remove_small_holes(mask, area_threshold=min_area)
    mask = morphology.binary_closing(mask, morphology.disk(3))
    return mask.astype(np.uint8)

def mask_to_polygons(mask, px2m, min_area_m2=0.05, simplify_tol_px=2.0):
    h_px = mask.shape[0]
    polygons = []
    for contour in measure.find_contours(mask, level=0.5):
        xy = np.flip(contour, axis=1)
        xy[:, 0] *= px2m
        xy[:, 1] = (h_px - xy[:, 1]) * px2m
        if len(xy) < 6:
            continue
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        poly = poly.simplify(simplify_tol_px * px2m, preserve_topology=True)
        if poly.area < min_area_m2:
            continue
        polygons.append(poly)
    return polygons

def polygon_to_mesh(poly, extrude_m):
    if poly.is_empty or poly.area <= 0:
        return None
    try:
        return trimesh.creation.extrude_polygon(poly, height=extrude_m)
    except Exception as e:
        print(f"  ⚠ extrude failed: {e}")
        return None

def phrase_to_safe(phrase):
    return str(phrase).strip().lower().replace(" ", "_").replace("/", "_")

def get_extrude(phrase):
    return EXTRUDE_DEPTH_M.get(phrase.lower(), EXTRUDE_DEPTH_M["default"])

def get_sionna_mat(phrase):
    return SIONNA_MATERIAL.get(phrase.lower(), SIONNA_MATERIAL["default"])

def box_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def nms(boxes, phrases, scores, iou_thresh=0.4):
    if len(boxes) == 0:
        return boxes, phrases, scores
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        order = np.array([j for j in order[1:] if box_iou(boxes[i], boxes[j]) <= iou_thresh])
    keep = np.array(keep)
    return boxes[keep], [phrases[k] for k in keep], scores[keep]

def get_tiles(W, H, tile_size, overlap):
    tiles = []
    y = 0
    while y < H:
        y1 = min(y + tile_size, H)
        x = 0
        while x < W:
            x1 = min(x + tile_size, W)
            tiles.append((x, y, x1, y1))
            if x1 == W: break
            x += tile_size - overlap
        if y1 == H: break
        y += tile_size - overlap
    return tiles

def detect_on_image(image_pil, processor, model, text_labels, device):
    inputs = processor(images=image_pil, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"],
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image_pil.size[::-1]],
    )[0]
    return (results["boxes"].detach().cpu().numpy(),
            results["labels"],
            results["scores"].detach().cpu().numpy())

def is_valid_box(box, W, H, x_min_px, x_max_px, y_min_px, y_max_px, min_area_frac, max_area_frac, min_asp, max_asp):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = x2-x1, y2-y1

    if cx < x_min_px or cx > x_max_px:
        return False, "outside crop zone"
    if cy < y_min_px or cy > y_max_px:
        return False, "outside crop zone"

    box_area = bw * bh
    if box_area < min_area_frac * W * H:
        return False, "too small"
    if box_area > max_area_frac * W * H:
        return False, "too large"

    aspect = bw / bh if bh > 0 else 999
    if aspect < min_asp or aspect > max_asp:
        return False, "bad aspect"

    return True, ""

# ─────────────────────────────────────────────
# Load image
# ─────────────────────────────────────────────
print(f"\n{'─'*52}")
print(f" Facade Pipeline  |  device={DEVICE}")
print(f"{'─'*52}\n")

image_pil    = Image.open(IMAGE_PATH).convert("RGB")
image_source = np.array(image_pil)
H_px, W_px   = image_source.shape[:2]
px2m         = compute_pixel_to_meter(H_px, KNOWN_WALL_HEIGHT_M)

y_min_px = int(H_px * CROP_TOP_FRACTION)
y_max_px = int(H_px * (1.0 - CROP_BOTTOM_FRACTION))
x_min_px = int(W_px * CROP_LEFT_FRACTION)
x_max_px = int(W_px * CROP_RIGHT_FRACTION)

print(f"Image  : {W_px}×{H_px} px  |  {px2m*100:.3f} cm/px  (wall={KNOWN_WALL_HEIGHT_M} m)")
print(f"ROI    : x {x_min_px}–{x_max_px}  y {y_min_px}–{y_max_px} px")
print(f"         (top {int(CROP_TOP_FRACTION*100)}%  bottom {int(CROP_BOTTOM_FRACTION*100)}%  left {int(CROP_LEFT_FRACTION*100)}%  right crops at {int(CROP_RIGHT_FRACTION*100)}%)")
print(f"Filters: aspect {MIN_ASPECT}–{MAX_ASPECT}  |  box area {MIN_BOX_AREA_FRACTION*100:.2f}%–{MAX_BOX_AREA_FRACTION*100:.1f}% of image")
print(f"Classes: {sorted(KEEP_CLASSES)}\n")

# ─────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────
print("Loading Grounding DINO …")
processor       = AutoProcessor.from_pretrained(MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(DEVICE)

print("Loading SAM …")
sam           = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)
sam_predictor.set_image(image_source)

# ─────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────
all_boxes, all_phrases, all_scores = [], [], []

# Crop image to building zone (excluding sky, ground, and left curtain wall)
building_crop = image_pil.crop((x_min_px, y_min_px, x_max_px, y_max_px))
crop_W = x_max_px - x_min_px

print("Running DINO on building crop (full res) …")
b, p, s = detect_on_image(building_crop, processor, grounding_model, TEXT_LABELS, DEVICE)
for box in b:
    all_boxes.append([box[0]+x_min_px, box[1]+y_min_px, box[2]+x_min_px, box[3]+y_min_px])
all_phrases.extend(p); all_scores.extend(s)
print(f"  → {len(b)} detections")

if USE_TILING:
    tiles = get_tiles(crop_W, y_max_px - y_min_px, TILE_SIZE, TILE_OVERLAP)
    print(f"\nRunning DINO on {len(tiles)} tiles of building zone …")
    for tx0, ty0, tx1, ty1 in tiles:
        tile_pil = image_pil.crop((tx0+x_min_px, ty0+y_min_px, tx1+x_min_px, ty1+y_min_px))
        tb, tp, ts = detect_on_image(tile_pil, processor, grounding_model, TEXT_LABELS, DEVICE)
        for box in tb:
            all_boxes.append([box[0]+tx0+x_min_px, box[1]+ty0+y_min_px,
                               box[2]+tx0+x_min_px, box[3]+ty0+y_min_px])
        all_phrases.extend(tp); all_scores.extend(ts)
    print(f"  → {len(all_boxes)} raw detections before filtering")

# ── Filter: class whitelist + crop zone + size ──────────────────────────────
filtered_boxes, filtered_phrases, filtered_scores = [], [], []
rejected = {"class": 0, "crop": 0, "size": 0, "aspect": 0}

for box, phrase, score in zip(all_boxes, all_phrases, all_scores):
    phrase = str(phrase).lower().strip()

    if phrase not in KEEP_CLASSES:
        rejected["class"] += 1
        continue

    ok, reason = is_valid_box(box, W_px, H_px, x_min_px, x_max_px, y_min_px, y_max_px,
                               MIN_BOX_AREA_FRACTION, MAX_BOX_AREA_FRACTION,
                               MIN_ASPECT, MAX_ASPECT)
    if not ok:
        key = "crop" if "crop" in reason else ("aspect" if "aspect" in reason else "size")
        rejected[key] += 1
        continue

    filtered_boxes.append(box)
    filtered_phrases.append(phrase)
    filtered_scores.append(score)

print(f"\nFiltering: kept {len(filtered_boxes)}  |  "
      f"rejected class={rejected['class']}  crop={rejected['crop']}  "
      f"size={rejected['size']}  aspect={rejected['aspect']}")

# NMS
if filtered_boxes:
    fb = np.array(filtered_boxes)
    fs = np.array(filtered_scores)
    fb, filtered_phrases, fs = nms(fb, filtered_phrases, fs, NMS_IOU_THRESH)
    print(f"After NMS: {len(fb)} detections\n")
else:
    fb, fs = np.zeros((0,4)), np.array([])
    print("⚠  No detections survived filtering. Try lowering BOX_THRESHOLD.\n")

# ─────────────────────────────────────────────
# SAM + mesh export
# ─────────────────────────────────────────────
scene_meshes, metadata, class_polygons = [], [], {}

for idx, (box, phrase, score) in enumerate(zip(fb, filtered_phrases, fs)):
    phrase = str(phrase)
    print(f"[{idx:03d}] {phrase:<22}  score={score:.2f}")

    box = np.clip(np.array(box, dtype=np.float32), [0,0,0,0], [W_px,H_px,W_px,H_px])
    masks, mask_scores, _ = sam_predictor.predict(box=box, multimask_output=True)
    best = int(np.argmax(mask_scores))
    mask = clean_binary_mask(masks[best], min_area=300)

    # Zero out anything outside the ROI (SAM may bleed into excluded zones)
    mask[y_max_px:, :] = 0
    mask[:y_min_px, :] = 0
    mask[:, :x_min_px] = 0
    mask[:, x_max_px:] = 0

    safe = phrase_to_safe(phrase)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "masks", f"{idx:03d}_{safe}.png"),
                (mask*255).astype(np.uint8))

    polys = mask_to_polygons(mask, px2m)
    if not polys:
        print(f"  ↳ no valid polygons, skipping"); continue

    merged = unary_union(polys)
    geoms  = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)
    class_polygons.setdefault(phrase, []).extend(geoms)

    extrude_m  = get_extrude(phrase)
    sionna_mat = get_sionna_mat(phrase)

    for j, poly in enumerate(geoms):
        mesh = polygon_to_mesh(poly, extrude_m)
        if mesh is None: continue
        obj_path = os.path.join(OUTPUT_DIR, "meshes", f"{idx:03d}_{j:02d}_{safe}.obj")
        mesh.export(obj_path)
        scene_meshes.append((mesh, sionna_mat, phrase))
        metadata.append({"id": f"{idx:03d}_{j:02d}", "label": phrase,
                         "sionna_material": sionna_mat, "extrude_m": extrude_m,
                         "area_m2": round(poly.area, 4),
                         "mesh_file": os.path.basename(obj_path)})

    print(f"  ↳ {len(geoms)} polygon(s)  mat={sionna_mat}  depth={extrude_m}m")

# ─────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────
print("\nExporting …")
if scene_meshes:
    combined = trimesh.util.concatenate([m for m,_,_ in scene_meshes])
    combined.export(os.path.join(OUTPUT_DIR, "combined_scene.obj"))
    b2 = combined.bounds; d = b2[1]-b2[0]
    print(f"  Scene: X={d[0]:.2f}m  Y={d[1]:.2f}m  Z={d[2]:.2f}m")

for cls, polys in class_polygons.items():
    safe = phrase_to_safe(cls); ext = get_extrude(cls)
    ms = [m for p in polys if (m := polygon_to_mesh(p, ext)) is not None]
    if not ms: continue
    trimesh.util.concatenate(ms).export(
        os.path.join(OUTPUT_DIR, "per_class", f"{safe}.obj"))
    print(f"  → per_class/{safe}.obj  ({len(ms)} meshes)")

json.dump({"scene_name":"ut_campus_facade","px_to_meter":px2m,
           "image_size_px":[W_px,H_px],"wall_height_m":KNOWN_WALL_HEIGHT_M,
           "crop_bottom_fraction":CROP_BOTTOM_FRACTION,
           "objects":metadata},
          open(os.path.join(OUTPUT_DIR,"sionna_scene.json"),"w"), indent=2)

print(f"\n✓ done  —  {len(metadata)} mesh regions exported\n")