"""
Export a trained DEIMv2 checkpoint (.pth) to ONNX.

This follows the official DEIMv2 / D-FINE export pattern (tools/deployment/export_onnx.py):
the exported graph takes `images` + `orig_target_sizes` and returns `labels`, `boxes`,
`scores` (boxes are already in original-image coordinates).

IMPORTANT: run this from *inside the DEIMv2 repository* so that `engine.core.YAMLConfig`
and the model/postprocessor `.deploy()` methods are importable. Copy this file into the
DEIMv2 repo root (or add the repo to PYTHONPATH) before running.

Example:
    python pth2onnx.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_s.pth --check --simplify
"""

import argparse
import os

import torch
import torch.nn as nn

from engine.core import YAMLConfig  # provided by the DEIMv2 repository


def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)

    # Disable backbone pretrained download during export.
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
        cfg.model.load_state_dict(state)
    else:
        raise AttributeError("Provide a checkpoint with -r/--resume to export weights.")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model()

    size = args.size
    data = torch.rand(1, 3, size, size)
    orig_size = torch.tensor([[size, size]])

    # Sanity forward pass before tracing.
    _ = model(data, orig_size)

    dynamic_axes = {
        "images": {0: "N"},
        "orig_target_sizes": {0: "N"},
    }

    output_file = args.output
    if output_file is None:
        output_file = (
            args.resume.replace(".pth", ".onnx") if args.resume else "model.onnx"
        )

    torch.onnx.export(
        model,
        (data, orig_size),
        output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"Exported ONNX -> {output_file}")

    if args.check:
        import onnx

        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX check passed.")

    if args.simplify:
        import onnx
        import onnxsim

        onnx_model = onnx.load(output_file)
        onnx_model, ok = onnxsim.simplify(onnx_model)
        assert ok, "onnxsim simplification failed"
        onnx.save(onnx_model, output_file)
        print(f"Simplified ONNX -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="DEIMv2 config yaml")
    parser.add_argument("-r", "--resume", type=str, required=True, help="checkpoint .pth")
    parser.add_argument("-o", "--output", type=str, default=None, help="output .onnx path")
    parser.add_argument("-s", "--size", type=int, default=640, help="square input size")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--check", action="store_true", help="validate the exported model")
    parser.add_argument("--simplify", action="store_true", help="simplify with onnxsim")
    main(parser.parse_args())
