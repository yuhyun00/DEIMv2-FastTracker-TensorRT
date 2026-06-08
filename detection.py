"""
DEIMv2 (TensorRT) + FastTracker (C++) pipeline.

Pipeline:
    RGB image -> preprocess -> DEIMv2 TensorRT inference -> (optional) FastTracker
              -> per-class confidence threshold -> output

The detection/tracking logic lives in the `DEIMv2FastTracker` class. Visualization
is a separate, standalone function. `main()` globs an input folder, runs the
pipeline on each image, and saves the visualized results.

Example:
    python3 detection.py \
        --input ./images --output ./results \
        --trt ./deimv2_s.engine --size 640 --model-size s \
        --track --conf 0.4 0.5 0.3 \
        --names person,bike,car
"""

import argparse
import collections
import glob
import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import tensorrt as trt


# --------------------------------------------------------------------------- #
# TensorRT inference wrapper
# --------------------------------------------------------------------------- #
class TRTInference(object):
    def __init__(self, engine_path, device="cuda:0", backend="torch", max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

            assert self.bindings[n].data.dtype == blob[n].dtype, "{} dtype mismatch".format(n)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    def __call__(self, blob):
        if self.backend == "torch":
            return self.run_torch(blob)
        raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        if self.backend == "torch" and torch.cuda.is_available():
            torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# Import the compiled C++ FastTracker (built via FastTracker/setup.py)
# --------------------------------------------------------------------------- #
def _import_fasttracker():
    try:
        import fasttracker  # noqa: F401
        return fasttracker
    except ImportError:
        # Fall back to the in-place build inside the FastTracker/ folder.
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(here, "FastTracker"))
        import fasttracker  # noqa: F401
        return fasttracker


# Output column layout (used everywhere downstream): one row per detection/track.
X1, Y1, X2, Y2, SCORE, CLASS, TRACK_ID = range(7)

# Model sizes that DO NOT use ImageNet normalization (matches DEIMv2 trt_inf.py).
_NO_NORM_SIZES = {"atto", "femto", "pico", "n"}


class DEIMv2FastTracker:
    """End-to-end DEIMv2 (TensorRT) detector with an optional C++ FastTracker.

    Output of `run()` is an (K, 7) float32 array with columns:
        [x1, y1, x2, y2, score, class_id, track_id]
    where track_id is -1 when tracking is disabled.
    """

    def __init__(self, engine_path, input_size, model_size, device="cuda:0",
                 track_buffer=30, frame_rate=30):
        self.device = device
        self.input_size = (int(input_size), int(input_size))
        self.model_size = model_size

        self.model = TRTInference(engine_path, device=device)

        normalize = model_size not in _NO_NORM_SIZES
        self.transforms = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize else T.Lambda(lambda x: x),
        ])

        # The tracker keeps state across frames, so create it once and reuse it
        # for the whole image sequence / folder.
        ft = _import_fasttracker()
        self.tracker = ft.FastTracker(frame_rate=frame_rate, track_buffer=track_buffer)

    # ---- 1) preprocessing -------------------------------------------------- #
    def preprocess(self, rgb_image):
        """RGB uint8 numpy image -> blob dict for the TRT engine."""
        h, w = rgb_image.shape[:2]
        im_pil = Image.fromarray(rgb_image)
        im_data = self.transforms(im_pil)[None]
        orig_size = torch.tensor([w, h])[None]
        return {
            "images": im_data.to(self.device),
            "orig_target_sizes": orig_size.to(self.device),
        }

    # ---- 2) detection (TensorRT) ------------------------------------------ #
    def infer(self, blob):
        """Run the engine; return (labels, boxes, scores) numpy arrays for batch 0."""
        output = self.model(blob)
        labels = output["labels"][0].detach().cpu().numpy()
        boxes = output["boxes"][0].detach().cpu().numpy()   # xyxy in original image coords
        scores = output["scores"][0].detach().cpu().numpy()
        return labels, boxes, scores

    # ---- 3) tracking (C++ FastTracker) ------------------------------------ #
    def track(self, dets):
        """dets: (N, 6) [x1,y1,x2,y2,score,class] -> (M, 7) with track_id appended."""
        dets = np.ascontiguousarray(dets, dtype=np.float32)
        return self.tracker.update(dets)

    # ---- 4) per-class confidence threshold -------------------------------- #
    @staticmethod
    def apply_class_conf(arr, class_conf):
        """Keep rows whose score >= class_conf[class_id].

        class_conf is a list indexed by class id. Classes without an entry use
        the last value as a fallback. If class_conf is None, nothing is filtered.
        """
        if class_conf is None or len(arr) == 0:
            return arr
        class_conf = np.asarray(class_conf, dtype=np.float32)
        cls = arr[:, CLASS].astype(int)
        # Clamp class ids that fall outside the provided list to the last threshold.
        idx = np.clip(cls, 0, len(class_conf) - 1)
        thr = class_conf[idx]
        keep = arr[:, SCORE] >= thr
        return arr[keep]

    # ---- 5) everything together ------------------------------------------- #
    def run(self, rgb_image, track=True, class_conf=None):
        blob = self.preprocess(rgb_image)
        labels, boxes, scores = self.infer(blob)

        dets = np.concatenate(
            [
                boxes.astype(np.float32),
                scores.reshape(-1, 1).astype(np.float32),
                labels.reshape(-1, 1).astype(np.float32),
            ],
            axis=1,
        )  # (N, 6)

        if track:
            out = self.track(dets)  # (M, 7)
        else:
            track_col = np.full((dets.shape[0], 1), -1.0, dtype=np.float32)
            out = np.concatenate([dets, track_col], axis=1)  # (N, 7)

        # Confidence threshold is applied right before returning the final output.
        out = self.apply_class_conf(out, class_conf)
        return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Visualization (separate from the pipeline class)
# --------------------------------------------------------------------------- #
def _color_for(idx):
    """Deterministic BGR color for a track id / class id."""
    idx = int(idx) + 3
    return (37 * idx % 255, 17 * idx % 255, 29 * idx % 255)


def visualize(rgb_image, outputs, names=None):
    """Draw boxes/labels/ids on a copy of the image. Returns a BGR image (for cv2.imwrite).

    outputs: (K, 7) [x1,y1,x2,y2,score,class_id,track_id]
    names: optional list of class names indexed by class id.
    """
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR).copy()

    for row in outputs:
        x1, y1, x2, y2 = (int(round(v)) for v in row[:4])
        score = float(row[SCORE])
        cls = int(row[CLASS])
        tid = int(row[TRACK_ID])

        color = _color_for(tid if tid >= 0 else cls)

        cls_name = names[cls] if (names is not None and 0 <= cls < len(names)) else str(cls)
        label = f"{cls_name} {score:.2f}"
        if tid >= 0:
            label = f"ID:{tid} " + label

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 2)
        cv2.rectangle(img, (x1, ty - th - baseline), (x1 + tw, ty + baseline), color, -1)
        cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return img


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _parse_names(value):
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return [n.strip() for n in value.split(",") if n.strip()]


def _collect_images(input_dir):
    files = []
    for ext in _IMG_EXTS:
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="DEIMv2 (TensorRT) + FastTracker (C++) pipeline")
    parser.add_argument("-i", "--input", required=True, help="folder containing input images")
    parser.add_argument("-o", "--output", required=True, help="folder to save visualized results")
    parser.add_argument("-trt", "--trt", required=True, help="path to the TensorRT engine (.engine)")
    parser.add_argument("--track", action="store_true", help="enable FastTracker tracking")
    parser.add_argument("--conf", type=float, nargs="+", default=None,
                        help="per-class confidence thresholds (index = class id)")
    parser.add_argument("-s", "--size", type=int, required=True, help="model input size, e.g. 640")
    parser.add_argument("-ms", "--model-size", required=True,
                        choices=["atto", "femto", "pico", "n", "s", "m", "l", "x"])
    parser.add_argument("--names", default=None,
                        help="class names: comma-separated string or a file (one name per line)")
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--track-buffer", type=int, default=30, help="frames a lost track survives")
    parser.add_argument("--frame-rate", type=int, default=30)
    args = parser.parse_args()

    names = _parse_names(args.names)
    os.makedirs(args.output, exist_ok=True)

    images = _collect_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        return

    pipeline = DEIMv2FastTracker(
        engine_path=args.trt,
        input_size=args.size,
        model_size=args.model_size,
        device=args.device,
        track_buffer=args.track_buffer,
        frame_rate=args.frame_rate,
    )

    print(f"Found {len(images)} images. track={args.track}")
    for path in images:
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"  [skip] could not read {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        outputs = pipeline.run(rgb, track=args.track, class_conf=args.conf)

        vis = visualize(rgb, outputs, names=names)
        out_path = os.path.join(args.output, os.path.basename(path))
        cv2.imwrite(out_path, vis)
        print(f"  {os.path.basename(path)}: {outputs.shape[0]} objects -> {out_path}")


if __name__ == "__main__":
    main()
