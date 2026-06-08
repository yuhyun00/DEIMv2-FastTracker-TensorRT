# FastTracker (C++ + pybind11)

A trimmed-down build of the [FastTracker](https://github.com/Hamidreza-Hashempoor/FastTracker)
C++ multi-object tracker, exposed to Python via pybind11. OpenCV is **not** required
(visualization is handled on the Python side); only **Eigen3** and **pybind11** are needed.

The tracker is a ByteTrack-style detection-based tracker with occlusion handling.
Compared to the original C++ source, this version drops the OpenCV/demo code and
propagates the detection class id (`label`) through to the output.

## Layout
```
FastTracker/
├── include/   FastTracker.h, STrack.h, kalmanFilter.h, dataType.h, lapjv.h
├── src/       FastTracker.cpp, STrack.cpp, kalmanFilter.cpp, lapjv.cpp, utils.cpp
├── bindings.cpp   # pybind11 module
└── setup.py
```

## Build
```bash
sudo apt install libeigen3-dev        # Eigen headers
pip install pybind11

# in-place build (module lands in this folder):
cd FastTracker
python setup.py build_ext --inplace

# or install into the environment:
pip install ./FastTracker             # from the project root
```
If Eigen is in a non-standard location, set `EIGEN_INCLUDE_DIR` to the directory
containing the `Eigen/` folder.

## Usage
```python
import numpy as np
import fasttracker

tracker = fasttracker.FastTracker(frame_rate=30, track_buffer=30)

# Per frame: detections as an (N, 6) float array.
#   columns = [x1, y1, x2, y2, score, class_id]   (absolute image coordinates)
dets = np.array([[100, 100, 140, 200, 0.9, 0]], dtype=np.float32)

tracks = tracker.update(dets)
#   returns (M, 7) float array:
#   columns = [x1, y1, x2, y2, score, class_id, track_id]
```
Call `update()` once per frame in temporal order; the tracker keeps state between
calls. Pass an empty `(0, 6)` array for frames with no detections.

## Notes
- Internal `track_thresh` (0.5) splits detections into high/low score for the
  two-stage association — this is independent of any display confidence threshold
  you apply afterwards.
- Track ids come from a process-global counter; restart the process (or
  reinstantiate in a fresh process) to reset ids between independent sequences.
