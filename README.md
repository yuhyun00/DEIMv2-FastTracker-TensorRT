# DEIMv2-FastTracker-TensorRT

*Read this in other languages: [한국어](README_KR.md)*

A detection-and-tracking pipeline that combines the
[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) object detector (TensorRT-accelerated)
with the [FastTracker](https://github.com/Hamidreza-Hashempoor/FastTracker) C++ multi-object tracker.

- **Detection**: DEIMv2 is converted to a TensorRT engine for inference.
- **Tracking**: the ByteTrack-style FastTracker C++ tracker is wrapped with pybind11 and called from Python (OpenCV dependency removed — only Eigen + pybind11 are needed).
- **Per-class confidence**: a different confidence threshold can be applied per class id, right before the final output.
- **Visualization**: a standalone function (separate from the pipeline class) draws boxes, classes, and track ids.

```
RGB image → preprocess → DEIMv2 TensorRT inference → (optional) FastTracker → per-class confidence → output → visualization
```

## Directory layout

```
DEIMv2-FastTracker-TensorRT/
├── main.py            # pipeline class + visualization function + main()
├── pth2onnx.py        # DEIMv2 .pth -> .onnx
├── onnx2trt.py        # .onnx -> .engine (TensorRT)
├── requirements.txt
└── FastTracker/       # C++ tracker + pybind11 bindings
    ├── include/  FastTracker.h, STrack.h, kalmanFilter.h, dataType.h, lapjv.h
    ├── src/      FastTracker.cpp, STrack.cpp, kalmanFilter.cpp, lapjv.cpp, utils.cpp
    ├── bindings.cpp
    ├── setup.py
    └── README.md      # tracker build/usage details
```

## Installation

Requirements: Python ≥ 3.7, a C++14 compiler, CUDA + TensorRT, Eigen3.

```bash
# 1) Eigen headers (needed to build the C++ tracker)
sudo apt install libeigen3-dev

# 2) Python packages
pip install -r requirements.txt

# 3) Install TensorRT (separately, matching your CUDA / platform)
pip install tensorrt

# 4) Build the C++ FastTracker module
cd FastTracker
python setup.py build_ext --inplace   # produces fasttracker*.so in this folder
cd ..
# or install into the environment: pip install ./FastTracker
```

If Eigen is not in a standard location, set the `EIGEN_INCLUDE_DIR` environment variable
to the directory that contains the `Eigen/` folder.

## Usage

### 1. Model conversion (.pth → .onnx → .engine)

ONNX export must be run **from inside the DEIMv2 repository** (it needs
`engine.core.YAMLConfig` and the `.deploy()` methods). Copy `pth2onnx.py` into the
DEIMv2 repo root, then run it.

```bash
# (inside the DEIMv2 repo) pth -> onnx
python pth2onnx.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_s.pth --check --simplify

# onnx -> tensorrt engine
python3 onnx2trt.py --onnx deimv2_s.onnx --saveEngine deimv2_s.engine --fp16 --size 640
```

### 2. Run the pipeline

Reads every image in the input folder, runs detection/tracking, and saves the
visualized results to the output folder.

```bash
# detection + tracking
python3 main.py \
    --input ./images --output ./results \
    --trt ./deimv2_s.engine --size 640 --model-size s \
    --track --conf 0.4 0.5 0.3 \
    --names person,bike,car

# detection only (tracking disabled)
python3 main.py -i ./images -o ./results -trt ./deimv2_s.engine -s 640 -ms s --conf 0.4 0.5 0.3
```

#### Main arguments

| Argument | Description |
|----------|-------------|
| `-i, --input` | input image folder |
| `-o, --output` | folder to save results |
| `-trt, --trt` | path to the TensorRT engine (`.engine`) |
| `--track` | enable FastTracker tracking (detection only if omitted) |
| `--conf` | per-class confidence thresholds (list index = class id) |
| `-s, --size` | model input size (e.g. 640) |
| `-ms, --model-size` | `atto/femto/pico/n/s/m/l/x` (decides whether normalization is applied) |
| `--names` | class names (comma-separated string or a file, one name per line) |
| `-d, --device` | inference device (default `cuda:0`) |
| `--track-buffer` | frames a lost track survives (default 30) |
| `--frame-rate` | input frame rate (default 30) |

> `--conf` is indexed by class id. Example: `--conf 0.4 0.5 0.3` → class 0 uses 0.4, class 1 uses 0.5, class 2 uses 0.3.
> Class ids beyond the list fall back to the last value. Even with tracking enabled, the confidence threshold is applied **after** tracking, right before the final output.

## Python API

You can also use the `DEIMv2FastTracker` class from `main.py` directly.

```python
import cv2
from main import DEIMv2FastTracker, visualize

pipe = DEIMv2FastTracker(
    engine_path="deimv2_s.engine",
    input_size=640,
    model_size="s",
    device="cuda:0",
)

bgr = cv2.imread("frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# (K, 7) float32: [x1, y1, x2, y2, score, class_id, track_id]
out = pipe.run(rgb, track=True, class_conf=[0.4, 0.5, 0.3])

vis = visualize(rgb, out, names=["person", "bike", "car"])  # BGR image, ready to save
cv2.imwrite("result.jpg", vis)
```

### Methods

- `preprocess(rgb_image)` — convert an RGB numpy image into the engine input blob
- `infer(blob)` — TensorRT inference, returns `(labels, boxes, scores)`
- `track(dets)` — feed `(N,6)` detections into the C++ FastTracker, returns `(M,7)`
- `apply_class_conf(arr, class_conf)` — per-class confidence filter
- `run(rgb_image, track=True, class_conf=None)` — run all of the above at once

The tracker keeps state across frames inside the class instance. Create a new
`DEIMv2FastTracker` per folder (sequence) to restart track ids.

## Output format

Each row of the array returned by `run()` is:

```
[x1, y1, x2, y2, score, class_id, track_id]
```

- Coordinates are `xyxy` in original-image space.
- `track_id` is `-1` when tracking is disabled (`track=False`).

## License / Sources

- DEIMv2: https://github.com/Intellindust-AI-Lab/DEIMv2
- FastTracker: https://github.com/Hamidreza-Hashempoor/FastTracker
