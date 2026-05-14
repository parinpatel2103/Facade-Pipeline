# Facade Segmentation Pipeline

Photogrammetry pipeline for building a digital twin of the UT campus. Takes photos of building facades and produces segmented, scaled 3D meshes of windows and doors — ready for Blender inspection and Sionna RF ray tracing simulation.

```
photo → Grounding DINO → SAM → scaled .obj meshes → Blender / Sionna RT
```

---

## Related Project Documentation

A higher-level overview of the project architecture, motivation, experiments, and system design can be found here:

[AI-Native-5G-Digital-Twin](https://github.com/parinpatel2103/ai-native-5G-digital-twin)

## How it works

1. **Grounding DINO** detects bounding boxes for windows and doors using open-vocabulary text prompts. Runs on the full building crop plus a tiled pass to catch small repeated elements like individual window panes.
2. **Filtering** removes false positives using a class whitelist, spatial crop zones (sky, ground, trees, adjacent structures), aspect ratio bounds, and min/max area thresholds.
3. **NMS** (non-maximum suppression) deduplicates overlapping detections across tiles.
4. **SAM** (Segment Anything) takes each surviving bounding box and produces a precise pixel-level mask.
5. **Scale conversion** maps pixel coordinates to real-world meters using a known wall height reference.
6. **Mesh export** extrudes each polygon into a 3D mesh with per-class depth (window glass = 5 cm, door = 10 cm) and exports `.obj` files grouped by semantic class.
7. **Sionna descriptor** writes a JSON scene file mapping each mesh to an ITU electromagnetic material for RF simulation.

---

## Project structure

```
facade-sam/
├── app/
│   ├── run_facade_pipeline.py   # main pipeline
│   ├── view_meshes.py           # mask overlay + scale summary
│   └── sionna_scene_loader.py   # loads output into Sionna RT
├── checkpoints/
│   ├── groundingdino_swint_ogc.pth
│   └── sam_vit_h_4b8939.pth
├── input/
│   └── building.jpg             # your photo goes here
├── output/
│   ├── masks/                   # binary PNG mask per detection
│   ├── meshes/                  # per-object .obj files
│   ├── per_class/               # one merged .obj per semantic class
│   ├── combined_scene.obj       # everything in one file
│   ├── mask_overlay.png         # visual QA — masks drawn on original image
│   └── sionna_scene.json        # scale + material metadata
├── scripts/
│   └── download_models.sh
├── .hf_cache/                   # HuggingFace model cache (auto-created)
├── Dockerfile
├── Makefile
└── requirements.txt
```

---

## Quickstart

### 1. Prerequisites

- Docker Desktop running with WSL2 integration enabled
- NVIDIA GPU with drivers installed (CPU fallback works but is slow)
- ~15 GB disk space for model checkpoints and Docker image

### 2. Download model checkpoints

```bash
make download-models
```

Downloads `sam_vit_h_4b8939.pth` (~2.4 GB) and `groundingdino_swint_ogc.pth` (~700 MB) into `checkpoints/`.

### 3. Build the Docker image

```bash
make build
```

### 4. Add your photo

```bash
cp /path/to/your/photo.jpg input/building.jpg
```

**Photo tips for best results:**
- Shoot straight-on to the facade, not at an angle
- Full building height should be visible — this sets the scale
- Good even lighting, avoid harsh shadows across windows
- Keep cars, trees, and adjacent buildings to a minimum in frame

### 5. Set the crop zones and wall height

Open `app/run_facade_pipeline.py` and update the config section at the top:

```python
KNOWN_WALL_HEIGHT_M  = 15.0   # estimated facade height in meters (3-4m per floor)

CROP_TOP_FRACTION    = 0.03   # fraction of image height to ignore from top (sky)
CROP_BOTTOM_FRACTION = 0.30   # fraction to ignore from bottom (ground, cars)
CROP_LEFT_FRACTION   = 0.27   # fraction to ignore from left (adjacent structures)
CROP_RIGHT_FRACTION  = 0.92   # fraction of width to keep (crop right edge)
```

These crop values are tuned per-photo. See the tuning guide below.

### 6. Run the pipeline

```bash
make run
```

### 7. Inspect results

```bash
make view
```

Saves `output/mask_overlay.png` showing detected regions on the original image. Open on Windows:

```bash
explorer.exe output/mask_overlay.png
```

---

## Importing into Blender

1. `File → Import → Wavefront (.obj)`
2. Select `output/combined_scene.obj` (everything) or files from `output/per_class/` (one object per class — recommended)
3. Import settings:
   - Forward: **Y**
   - Up: **Z**
   - Scale: **1.0** (already in meters)
4. Press `Numpad 1` for front view, `Numpad .` to zoom to selection

---

## Sionna RF simulation

After verifying geometry in Blender, load the scene for ray tracing.

### How it works

The Sionna workflow has two stages:

**1. Scene loader (`app/sionna_scene_loader.py`)**

Reads `output/sionna_scene.json` (written by the main pipeline) and builds a Mitsuba 3 XML scene file (`output/sionna_scene.xml`). Each per-class `.obj` mesh (e.g. `output/per_class/window.obj`) is registered as a shape, and each shape is assigned an ITU electromagnetic BSDF material. The XML is then loaded into Sionna's `Scene` object using `load_scene()`.

ITU electromagnetic material assignments:

| Semantic class   | Sionna material   |
|------------------|-------------------|
| window           | `itu_glass`       |
| glass_window     | `itu_glass`       |
| window_pane      | `itu_glass`       |
| door             | `itu_wood`        |
| brick_wall       | `itu_brick`       |
| concrete_wall    | `itu_concrete`    |
| pma_building     | `itu_concrete`    |

Run the loader to generate `output/sionna_scene.xml`:

```bash
pip install sionna tensorflow
python3 app/sionna_scene_loader.py
```


**2. Ray tracing experiments (`run_sionna.py`)**

Runs four experiments on the loaded scene at 3.5 GHz (mid-band 5G). TX and RX are single-element isotropic vertical-polarization arrays. Paths are solved with `PathSolver`.

| Experiment | What it measures |
|---|---|
| **1 — Non-LoS path validation** | Places TX at `[10, -20, 5]` m and RX at `[0, 20, 2]` m with no line-of-sight. Counts multipath components and prints the path coefficient tensor shape. Confirms the facade geometry is producing reflected/diffracted paths. |
| **2 — TX height sweep** | Sweeps TX height through 1.5 m, 10 m, 20 m, 35 m while keeping RX fixed. Reports path count and total received power at each height. Shows how elevation above the facade changes multipath richness. |
| **3 — Reflections on vs off** | Compares `max_depth=4` (reflections enabled) against `max_depth=0` (LoS only). Reports path count and power for both. Quantifies how much the facade contributes to received signal beyond direct path. |
| **4 — Reflection depth analysis** | Steps `max_depth` from 0 to 5 and records cumulative path count and power at each bounce level. Shows the marginal contribution of each additional reflection order. |

Run all experiments:

```bash
python3 run_sionna.py
```

**Scene setup (shared across all experiments)**

```python
scene.frequency = 3.5e9          # 3.5 GHz
scene.tx_array  = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array  = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
```

Power reported in experiments 2–4 is the sum of squared real and imaginary path coefficients: `Σ |a|²`.

---


 ## My Contributions

This project was completed as part of my senior design capstone at UT Austin. I contributed to the development, testing, and validation of the overall digital twin and RF simulation pipeline.

My main contributions included:

- RF ray-tracing simulation.
- Helping test and validate segmented building outputs for use in Blender and Sionna RT.
- Working with Sionna RT/Mitsuba scene setup to simulate wireless propagation in a campus digital twin environment.
- Supporting experiments comparing line-of-sight and reflected RF paths.
- Debugging environment/setup issues related to Python dependencies, Docker/Linux workflows, and simulation execution.
- Helping document the project workflow, limitations, and future improvements.


## Configuration reference

All parameters are at the top of `app/run_facade_pipeline.py`:

### Detection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BOX_THRESHOLD` | `0.22` | DINO confidence for bounding boxes. Lower = more detections, more noise. |
| `TEXT_THRESHOLD` | `0.18` | DINO text-match confidence. |
| `USE_TILING` | `True` | Run DINO on overlapping tiles to catch small windows. |
| `TILE_SIZE` | `1024` | Tile size in pixels. |
| `TILE_OVERLAP` | `256` | Overlap between tiles — prevents missing windows at tile edges. |
| `NMS_IOU_THRESH` | `0.4` | IoU threshold for deduplication across tiles. |

### Crop zones

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KNOWN_WALL_HEIGHT_M` | `15.0` | Estimated facade height. Sets pixel→meter scale. |
| `CROP_TOP_FRACTION` | `0.03` | Ignore top N% of image (sky). |
| `CROP_BOTTOM_FRACTION` | `0.30` | Ignore bottom N% (ground, cars, parking lot). |
| `CROP_LEFT_FRACTION` | `0.27` | Ignore left N% (adjacent buildings, trees). |
| `CROP_RIGHT_FRACTION` | `0.92` | Keep only left N% of width (cuts right edge structures). |

### Filters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_BOX_AREA_FRACTION` | `0.0005` | Minimum box area as fraction of image. Removes noise. |
| `MAX_BOX_AREA_FRACTION` | `0.02` | Maximum box area. Removes large false positives (walls, slabs). |
| `MIN_ASPECT` | `0.4` | Minimum width/height ratio. Rejects very tall thin strips. |
| `MAX_ASPECT` | `2.5` | Maximum width/height ratio. Rejects wide slabs. |

---

## Tuning guide

### Too few windows detected
- Lower `BOX_THRESHOLD` to `0.18`, `TEXT_THRESHOLD` to `0.14`
- Check crop fractions aren't cutting into the facade
- Open `output/mask_overlay.png` to visualize exactly what's being found

### False positives (cars, trees, walls getting through)
- Tighten crop fractions to exclude problem areas
- Lower `MAX_BOX_AREA_FRACTION` to kill large slab detections
- Lower `MAX_ASPECT` to kill wide horizontal strips

### Top row of windows missing
- Reduce `CROP_TOP_FRACTION` (try `0.03`)

### Bottom floor windows missing
- Reduce `CROP_BOTTOM_FRACTION` (try `0.25`)

### Adjacent building / structure on left getting detected
- Increase `CROP_LEFT_FRACTION`

### Adjacent structure on right getting detected
- Decrease `CROP_RIGHT_FRACTION`

### Geometry wrong size in Blender
- Adjust `KNOWN_WALL_HEIGHT_M` — measure the actual building if possible

### Meshes are flat / paper thin
- Increase `EXTRUDE_DEPTH_M` values in config (currently 5 cm for glass, 10 cm for doors)

---

## Makefile commands

| Command | Description |
|---------|-------------|
| `make build` | Build the Docker image |
| `make run` | Run the full pipeline |
| `make view` | Print detection summary, save mask overlay |
| `make shell` | Open bash shell inside the container |
| `make download-models` | Download SAM and DINO checkpoints |
| `make clean` | Clear all output files |

---

## Troubleshooting

**`Permission denied` on `make clean`**
Output files are owned by root (Docker runs as root). The Makefile uses `sudo rm`. Run `make clean` with your password.

**`Cannot connect to Docker daemon`**
Docker Desktop is not running. Start it from the Windows Start menu and wait for "Engine running".

**`exec format error` on docker-credential-desktop.exe**
WSL2 credential helper issue:
```bash
echo '{}' > ~/.docker/config.json
```

**`PermissionError: /.cache` when loading models**
The HuggingFace cache dir isn't writable. Make sure `.hf_cache/` exists in the project root:
```bash
mkdir -p .hf_cache
make run
```

---

## Models used

| Model | Source | Size |
|-------|--------|------|
| Grounding DINO Base | `IDEA-Research/grounding-dino-base` on HuggingFace | ~700 MB |
| SAM ViT-H | Meta AI / facebookresearch/segment-anything | ~2.4 GB |

---

## Known limitations

- **Occluded windows** (behind trees, signs, cars) cannot be detected from a single photo. Take a second photo from a slightly different angle to fill gaps, then merge both `sionna_scene.json` outputs.
- **Perspective distortion** — photos taken at an angle produce skewed geometry. Shooting straight-on minimizes this. True correction requires camera calibration (focal length + GPS distance).
- **Crop zones are per-photo** — each new building photo needs its own crop fraction tuning. See the tuning guide above.

---

## Roadmap

- [ ] Replace `KNOWN_WALL_HEIGHT_M` with proper camera calibration (focal length + GPS + compass)
- [ ] Multi-image merging — stitch detections from multiple photos of the same facade
- [ ] Auto crop zone detection — use sky/ground segmentation to set fractions automatically
- [ ] Mitsuba XML export for direct Sionna `load_scene()` ingestion
- [ ] Depth estimation (MiDaS) to improve Z positioning of window elements
