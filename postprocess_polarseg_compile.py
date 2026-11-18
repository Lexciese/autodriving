from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

import torch
import torch.nn as nn

from common.polarseg.network.BEV_Unet import BEV_Unet, double_conv_circular
from common.polarseg.network.ptBEV import ptBEVnet
from config import GlobalConfig
from postprocess_model_compile import build_engine, preflight, simplify_onnx


def circular_pad_width(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[..., -1:], x, x[..., :1]], dim=-1)


def circular_forward(self, x: torch.Tensor) -> torch.Tensor:
    x = circular_pad_width(x)
    x = self.conv1(x)
    x = circular_pad_width(x)
    x = self.conv2(x)
    return x


def replace_dropblock(module: nn.Module) -> None:
    for name, child in module.named_children():
        if child.__class__.__name__ == "DropBlock2D":
            setattr(module, name, nn.Identity())
        else:
            replace_dropblock(child)


def build_polarseg(config: GlobalConfig, pth: Path) -> ptBEVnet:
    bev_model = BEV_Unet(
        n_class=config.n_class_kitti - 1,
        n_height=config.grid_size[2],
        input_batch_norm=True,
        dropout=0.5,
        circular_padding=True,
    )
    model = ptBEVnet(
        bev_model,
        pt_model='pointnet',
        grid_size=config.grid_size,
        fea_dim=9,
        max_pt_per_encode=256,
        out_pt_fea_dim=512,
        kernal_size=1,
        pt_selection='random',
        fea_compre=config.grid_size[2],
    )
    model.load_state_dict(torch.load(pth, map_location="cpu"))
    model.to(config.gpu_device)
    model.eval()
    return model


def export_bev_onnx(bev: nn.Module, config: GlobalConfig, args, output_dir: Path, stem: str) -> Path:
    import onnx

    replace_dropblock(bev)
    double_conv_circular.forward = circular_forward
    bev = bev.to(config.gpu_device, dtype=config.dtype).eval()

    input_shape = (
        1,
        int(config.grid_size[2]),
        int(config.grid_size[0]),
        int(config.grid_size[1]),
    )
    dummy_input = torch.randn(input_shape, device=config.gpu_device, dtype=config.dtype)
    onnx_path = output_dir / f"{stem}.onnx"

    print(f"Exporting ONNX: {onnx_path}")
    export_kwargs = {
        "export_params": True,
        "opset_version": args.opset,
        "do_constant_folding": True,
        "input_names": ["out_data"],
        "output_names": ["logits"],
        "verbose": args.verbose,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(bev, dummy_input, str(onnx_path), **export_kwargs)
        logits = bev(dummy_input)

    expected_shape = (
        1,
        int(config.n_class_kitti) - 1,
        int(config.grid_size[0]),
        int(config.grid_size[1]),
        int(config.grid_size[2]),
    )
    if tuple(logits.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected BEV_Unet output shape: {tuple(logits.shape)} "
            f"(expected {expected_shape})"
        )

    print(f"ONNX input: out_data {tuple(input_shape)}")
    print(f"PyTorch output: logits {tuple(logits.shape)}")

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    return onnx_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile the PolarSeg BEV_Unet to ONNX and TensorRT."
    )
    parser.add_argument(
        "--pth",
        type=Path,
        default=None,
        help="PolarSeg weights. Defaults to config.polarseg_weight_path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the directory containing the pth.",
    )
    parser.add_argument(
        "--mode",
        choices=["onnx", "tensorrt", "both"],
        default="both",
        help="Convert to ONNX only, TensorRT only, or both (default: both).",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="TensorRT engine precision (default: fp16). TensorRT 11+ FP16 "
        "uses NVIDIA ModelOpt AutoCast.",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=4096,
        help="TensorRT workspace in MiB (default: 4096).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--trtexec",
        default=None,
        help="Path to trtexec. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--trt-args",
        default="",
        help="Additional trtexec arguments, e.g. --trt-args='--useCudaGraph'. "
        "Do not pass removed precision flags on TensorRT 11+.",
    )
    parser.add_argument(
        "--autocast-args",
        default="",
        help="Additional NVIDIA ModelOpt AutoCast arguments, e.g. "
        "--autocast-args='--calibration_data calib.npz --data_max 256'.",
    )
    parser.add_argument(
        "--no-autocast",
        action="store_true",
        help="Skip ModelOpt AutoCast for FP16. Useful when the ONNX is already "
        "FP16 or was converted elsewhere.",
    )
    parser.add_argument(
        "--skip-power-setup",
        action="store_true",
        help="Skip sudo nvpmodel -m 0 and sudo jetson_clocks.",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip onnxsim simplification.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose ONNX and trtexec output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preflight(args)

    config = GlobalConfig()
    pth = (args.pth.expanduser() if args.pth else Path(config.polarseg_weight_path)).resolve()
    if not pth.is_file():
        raise SystemExit(f"PolarSeg weights not found: {pth}")

    output_dir = (
        args.output_dir.expanduser() if args.output_dir else pth.parent
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{pth.stem}_bev"

    print(f"PolarSeg weights: {pth}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {output_dir}")
    print("Compiling BEV_Unet only; pointnet/voxelization/scatter stay in PyTorch.")

    model = build_polarseg(config, pth)
    bev = model.BEV_model

    if args.mode in ("onnx", "both"):
        onnx_path = export_bev_onnx(bev, config, args, output_dir, stem)
    else:
        onnx_path = output_dir / f"{stem}.onnx"
        if not onnx_path.is_file():
            raise SystemExit(f"ONNX file not found: {onnx_path}")

    if args.no_simplify:
        simplified_path = onnx_path
    else:
        simplified_path = simplify_onnx(onnx_path, output_dir / f"{stem}_sim.onnx")

    if args.mode in ("tensorrt", "both"):
        engine_path = output_dir / f"{stem}_{args.precision}.engine"
        build_engine(simplified_path, engine_path, args)
        print(f"TensorRT engine: {engine_path}")

    print(f"ONNX: {onnx_path}")
    if not args.no_simplify:
        print(f"Simplified ONNX: {simplified_path}")


if __name__ == "__main__":
    main()
