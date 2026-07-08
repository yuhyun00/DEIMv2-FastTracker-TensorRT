"""
Build a TensorRT engine (.engine) from a DEIMv2 ONNX model.

Uses the TensorRT Python API with an explicit-batch network and a dynamic-batch
optimization profile (inputs: `images` [N,3,H,W] and `orig_target_sizes` [N,2]).

Precision handling (DINOv3 backbone). Two independent sources of accuracy loss
are avoided here so the engine matches the ONNX/PyTorch outputs:

  * FP32 case -- TF32 is ON by default on Ampere+ GPUs (A6000/A100/30xx+) and
    truncates the matmul mantissa to 10 bits. The DINOv3 backbone has "massive
    activations" (a few very large values); TF32 rounds those, giving a small
    mean but large, input-dependent max error. We clear BuilderFlag.TF32 to get
    true FP32 (bit-exact with ONNX-Runtime). Pass --keep-tf32 to reproduce the bug.

  * FP16 case -- plain FP16 is catastrophic (cosine ~0.91). Two transformer ops
    overflow FP16 and must stay FP32; which one appears depends on how the ONNX
    was exported, so we guard both:
      - Softmax: attention logits are large and exp() saturates to inf. This is
        the decisive one on opset>=17 exports (measured: Softmax->FP32 alone
        recovers cosine 0.91 -> 0.9999; LayerNorm->FP32 alone does nothing).
      - LayerNorm: the variance x^2 overflows FP16 after attention. On opset>=17
        it is fused into a single Normalization layer; on opset<=16 it is
        decomposed into Reduce + Pow, so we force Normalization, Reduce and Pow.
    Pass --no-mixed to reproduce the broken result.

Example:
    # Accurate FP32 (TF32 disabled):
    python3 onnx2trt.py --onnx deimv2_s.onnx --saveEngine deimv2_s.engine --size 640
    # Fast, accurate FP16 (mixed precision):
    python3 onnx2trt.py --onnx deimv2_s.onnx --saveEngine deimv2_s_fp16.engine --fp16 --size 640
    # custom batch range:
    python3 onnx2trt.py --onnx m.onnx --saveEngine m.engine --min-batch 1 --opt-batch 1 --max-batch 8
"""

import argparse

import tensorrt as trt

# Layer types kept in FP32 inside an FP16 engine for the DINOv3/ViT backbone.
# SOFTMAX is the decisive one (attention exp() overflow). NORMALIZATION is the
# fused LayerNorm (opset>=17); REDUCE covers the decomposed LayerNorm variance
# (opset<=16). POW is the decomposed square and is matched separately below.
FP16_SENSITIVE_TYPES = {
    trt.LayerType.SOFTMAX,
    trt.LayerType.NORMALIZATION,
    trt.LayerType.REDUCE,
}


def _force_fp16_sensitive_fp32(network):
    """Force the FP16-sensitive layers (Softmax + fused/decomposed LayerNorm) to
    FP32. Covers both the fused Normalization op and the Reduce + Pow form.
    OBEY_PRECISION_CONSTRAINTS must be set on the config for this to take effect.
    Returns the count of layers forced to FP32.
    """
    forced = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        sensitive = layer.type in FP16_SENSITIVE_TYPES

        # Decomposed LayerNorm variance: elementwise x^2.
        if not sensitive and layer.type == trt.LayerType.ELEMENTWISE:
            try:
                sensitive = layer.op == trt.ElementWiseOperation.POW
            except AttributeError:
                pass

        if sensitive:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            forced += 1
    return forced


def build_engine(onnx_path, engine_path, size=640, fp16=False, mixed=True,
                 keep_tf32=False, min_batch=1, opt_batch=1, max_batch=1,
                 workspace_gb=4):
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
            if mixed:
                # Keep Softmax and LayerNorm in FP32 (the FP16 fix).
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
                forced = _force_fp16_sensitive_fp32(network)
                print(f"FP16 mixed precision: {forced} sensitive layers "
                      f"(Softmax/LayerNorm) forced to FP32.")
            else:
                print("Plain FP16 (--no-mixed): Softmax overflow, expect broken accuracy.")
        else:
            print("FP16 requested but not supported on this platform; using FP32.")
    else:
        # FP32 path: disable TF32 for bit-exact match with ONNX (the FP32 fix).
        if keep_tf32:
            print("TF32 kept (--keep-tf32): expect accuracy loss on DINOv3 backbone.")
        else:
            config.clear_flag(trt.BuilderFlag.TF32)
            print("TF32 disabled: true FP32, matches ONNX/PyTorch.")

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
    parser.add_argument("--fp16", action="store_true", help="enable FP16 precision (mixed by default)")
    parser.add_argument("--no-mixed", dest="mixed", action="store_false",
                        help="with --fp16: plain FP16, reproduces the broken result")
    parser.add_argument("--keep-tf32", action="store_true",
                        help="FP32 only: keep TF32 on, reproduces the FP32 accuracy loss")
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
        mixed=args.mixed,
        keep_tf32=args.keep_tf32,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_gb=args.workspace,
    )
