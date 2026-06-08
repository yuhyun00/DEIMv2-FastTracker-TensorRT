"""
Build a TensorRT engine (.engine) from a DEIMv2 ONNX model.

Uses the TensorRT Python API with an explicit-batch network and a dynamic-batch
optimization profile (inputs: `images` [N,3,H,W] and `orig_target_sizes` [N,2]).

Example:
    python3 onnx2trt.py --onnx deimv2_s.onnx --saveEngine deimv2_s.engine --fp16 --size 640
    # custom batch range:
    python3 onnx2trt.py --onnx m.onnx --saveEngine m.engine --min-batch 1 --opt-batch 1 --max-batch 8
"""

import argparse

import tensorrt as trt


def build_engine(onnx_path, engine_path, size=640, fp16=False,
                 min_batch=1, opt_batch=1, max_batch=1, workspace_gb=4):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled.")
        else:
            print("FP16 requested but not supported on this platform; using FP32.")

    # Dynamic-batch optimization profile.
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        (min_batch, 3, size, size),
        (opt_batch, 3, size, size),
        (max_batch, 3, size, size),
    )
    profile.set_shape(
        "orig_target_sizes",
        (min_batch, 2),
        (opt_batch, 2),
        (max_batch, 2),
    )
    config.add_optimization_profile(profile)

    print(f"Building engine from {onnx_path} ... (this can take a while)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed.")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Saved engine -> {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="input ONNX path")
    parser.add_argument("--saveEngine", required=True, help="output .engine path")
    parser.add_argument("--size", type=int, default=640, help="square input size (must match ONNX)")
    parser.add_argument("--fp16", action="store_true", help="enable FP16 precision")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument("--workspace", type=int, default=4, help="workspace size in GiB")
    args = parser.parse_args()

    build_engine(
        args.onnx,
        args.saveEngine,
        size=args.size,
        fp16=args.fp16,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_gb=args.workspace,
    )
