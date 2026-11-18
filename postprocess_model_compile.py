import argparse
import importlib
import importlib.util
import inspect
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


MODEL_CLASSES = {
    "ai23": ("ai23_model", "ai23"),
    "ai33": ("ai33_model", "ai33"),
    "xr20": ("xr20_model", "xr20"),
}

INPUT_NAMES = [
    "bev_segs",
    "bev_deps",
    "front_segs",
    "front_deps",
    "rp1",
    "rp2",
    "velocity",
]


class TraceableAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = tuple(output_size)

    def forward(self, x):
        out_h, out_w = self.output_size
        in_h, in_w = x.shape[2], x.shape[3]
        rows = []
        for oh in range(out_h):
            h_start = (oh * in_h) // out_h
            h_end = -(-((oh + 1) * in_h) // out_h)
            cols = []
            for ow in range(out_w):
                w_start = (ow * in_w) // out_w
                w_end = -(-((ow + 1) * in_w) // out_w)
                cols.append(
                    x[:, :, h_start:h_end, w_start:w_end].mean(
                        dim=(2, 3), keepdim=True
                    )
                )
            rows.append(torch.cat(cols, dim=3))
        return torch.cat(rows, dim=2)


class TraceableGRUCell(nn.Module):
    def __init__(self, gru: nn.GRUCell):
        super().__init__()
        self.input_size = gru.input_size
        self.hidden_size = gru.hidden_size
        self.weight_ih = gru.weight_ih
        self.weight_hh = gru.weight_hh
        self.bias_ih = gru.bias_ih
        self.bias_hh = gru.bias_hh

    def forward(self, x, hx):
        gates_x = F.linear(x, self.weight_ih, self.bias_ih)
        gates_h = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gates_x.chunk(3, 1)
        h_r, h_z, h_n = gates_h.chunk(3, 1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * hx


class CompileWrapper(nn.Module):
    def __init__(self, model: nn.Module, seq_len: int):
        super().__init__()
        self.model = model
        self.seq_len = seq_len

    def _unstack(self, x: torch.Tensor):
        return [x[:, i] for i in range(self.seq_len)]

    def forward(self, bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, velo):
        return self.model(
            self._unstack(bev_segs),
            self._unstack(bev_deps),
            self._unstack(front_segs),
            self._unstack(front_deps),
            rp1,
            rp2,
            velo,
        )


def check_conda_env():
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env != "dl":
        print(
            f"Warning: CONDA_DEFAULT_ENV={env!r}, expected 'dl'. "
            f"Run with: conda run --no-capture-output -n dl python {Path(sys.argv[0]).name} ..."
        )


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def check_onnxruntime_compat():
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:
        details = " ".join(
            str(item) for item in (exc, exc.__cause__, exc.__context__) if item
        )
        if "_ARRAY_API" in details or "NumPy 1.x" in details:
            raise SystemExit(
                "onnxruntime cannot run with the installed NumPy.\n"
                f"  numpy: {_package_version('numpy')}\n"
                f"  onnxruntime: {_package_version('onnxruntime')}\n"
                "Fix one of:\n"
                f'  "{sys.executable}" -m pip install "numpy<2"\n'
                f'  "{sys.executable}" -m pip install -U onnxruntime onnxsim\n'
                "Then verify with:\n"
                f'  "{sys.executable}" -c "import numpy, onnxruntime, onnxsim; '
                'print(numpy.__version__, onnxruntime.__version__)"'
            ) from exc
        raise SystemExit(
            f"onnxruntime failed to import: {type(exc).__name__}: {exc}\n"
            f'Install with: "{sys.executable}" -m pip install onnxruntime'
        ) from exc


def preflight(args):
    required = [("onnx", "onnx")]
    if not args.no_simplify:
        check_onnxruntime_compat()
        required.append(("onnxsim", "onnxsim"))

    failures = []
    for module_name, package_name in required:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append((package_name, exc))

    if failures:
        lines = [
            f"{package_name} failed to import: {type(exc).__name__}: {exc}"
            for package_name, exc in failures
        ]
        packages = " ".join(package_name for package_name, _ in failures)
        lines.append(f'Install with: "{sys.executable}" -m pip install {packages}')
        raise SystemExit("\n".join(lines))


def load_run_config(logdir: Path):
    config_path = logdir / "config.py"
    if not config_path.is_file():
        raise SystemExit(f"Missing config file: {config_path}")

    spec = importlib.util.spec_from_file_location("log_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load config from: {config_path}")

    log_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(log_config)
    config = log_config.GlobalConfig()
    config.logdir = logdir
    return config


def resolve_model_name(args_model, logdir: Path) -> str:
    model_name = args_model if args_model else logdir.name.split("_", 1)[0]
    if model_name not in MODEL_CLASSES:
        raise SystemExit(
            f"Unknown model '{model_name}'. Choose one of: {', '.join(sorted(MODEL_CLASSES))}"
        )
    return model_name


def load_model_class(model_name: str):
    module_name, class_name = MODEL_CLASSES[model_name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_model(model_name: str, config) -> nn.Module:
    model_cls = load_model_class(model_name)
    model = model_cls(config, device=config.gpu_device).to(
        config.gpu_device, dtype=config.dtype
    )
    return model


def build_dummy_inputs(config, batch_size: int):
    seq_len = int(config.seq_len)
    n_class = int(config.n_class)

    shapes = {
        "bev_segs": (batch_size, seq_len, n_class, int(config.bev_h), int(config.bev_w)),
        "bev_deps": (batch_size, seq_len, 1, int(config.bev_h), int(config.bev_w)),
        "front_segs": (
            batch_size,
            seq_len,
            n_class,
            int(config.front_h),
            int(config.front_w),
        ),
        "front_deps": (
            batch_size,
            seq_len,
            1,
            int(config.front_h),
            int(config.front_w),
        ),
        "rp1": (batch_size, 2),
        "rp2": (batch_size, 2),
        "velocity": (batch_size, 1),
    }

    dummy_inputs = tuple(
        torch.randn(shape, device=config.gpu_device, dtype=config.dtype)
        for shape in shapes.values()
    )
    return dummy_inputs, shapes


def replace_gru_cells(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GRUCell):
            setattr(module, name, TraceableGRUCell(child))
        else:
            replace_gru_cells(child)


def replace_adaptive_avg_pools(module):
    for name, child in module.named_children():
        if isinstance(child, nn.AdaptiveAvgPool2d):
            output_size = child.output_size
            if isinstance(output_size, int):
                output_size = (output_size, output_size)
            output_size = tuple(output_size)
            if output_size != (1, 1):
                setattr(module, name, TraceableAdaptiveAvgPool2d(output_size))
        else:
            replace_adaptive_avg_pools(child)


def export_onnx(model, config, args, output_dir: Path, stem: str):
    import onnx

    replace_gru_cells(model)
    replace_adaptive_avg_pools(model)
    wrapper = CompileWrapper(model, int(config.seq_len))
    wrapper = wrapper.to(config.gpu_device, dtype=config.dtype).eval()
    dummy_inputs, input_shapes = build_dummy_inputs(config, args.batch_size)
    onnx_path = output_dir / f"{stem}.onnx"

    print(f"Exporting ONNX: {onnx_path}")
    export_kwargs = {
        "export_params": True,
        "opset_version": args.opset,
        "do_constant_folding": True,
        "input_names": INPUT_NAMES,
        "output_names": ["pred_wp"],
        "verbose": args.verbose,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(wrapper, dummy_inputs, str(onnx_path), **export_kwargs)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    print("ONNX input bindings:")
    for name, shape in input_shapes.items():
        print(f"  {name}: {tuple(shape)}")

    return onnx_path


def describe_onnx_inputs(onnx_path: Path):
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    print("ONNX input bindings:")
    for inp in onnx_model.graph.input:
        dims = []
        for dim in inp.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
        print(f"  {inp.name}: {tuple(dims)}")


def simplify_onnx(onnx_path: Path, sim_path: Path) -> Path:
    import onnx
    import onnxsim

    if onnx_path.resolve() == sim_path.resolve():
        print(f"ONNX is already the simplified target: {onnx_path}")
        return onnx_path

    print(f"Simplifying ONNX with onnxsim: {onnx_path}")
    try:
        simplified, ok = onnxsim.simplify(str(onnx_path))
        if not ok:
            raise RuntimeError("onnxsim validation failed")
        onnx.save(simplified, str(sim_path))
        onnx.checker.check_model(simplified)
        print(f"Simplified ONNX: {sim_path}")
        return sim_path
    except Exception as exc:
        print(f"Warning: onnxsim failed: {exc}")
        print("Using unsimplified ONNX.")
        return onnx_path


def setup_max_performance():
    commands = [
        ["sudo", "nvpmodel", "-m", "0"],
        ["sudo", "jetson_clocks"],
    ]

    for command in commands:
        print(f"Running: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"Command not found: {command[0]}. "
                "Run this on a Jetson or pass --skip-power-setup."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Power setup failed with exit code {exc.returncode}: {' '.join(command)}"
            ) from exc

    try:
        subprocess.run(["nvpmodel", "-q"], check=False)
    except FileNotFoundError:
        pass


def find_trtexec(explicit_path):
    if explicit_path:
        resolved = shutil.which(explicit_path)
        if resolved:
            return resolved
        path = Path(explicit_path).expanduser()
        if path.is_file():
            return str(path)
        raise SystemExit(f"trtexec not found at: {explicit_path}")

    candidates = [
        shutil.which("trtexec"),
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate

    raise SystemExit(
        "trtexec not found. Install TensorRT or pass --trtexec /path/to/trtexec."
    )


def get_trtexec_major_version(trtexec: str):
    try:
        result = subprocess.run(
            [trtexec, "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception:
        return None

    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(
        r"TensorRT(?:\s+version)?[\s:-]*v?(\d+)", output, re.IGNORECASE
    )
    if match:
        return int(match.group(1))

    match = re.search(r"\bv?(\d+)\.\d+(?:\.\d+)?\b", output)
    if match:
        return int(match.group(1))
    return None


def trtexec_has_fp16_flag(trtexec: str):
    try:
        result = subprocess.run(
            [trtexec, "--help"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return None

    output = f"{result.stdout}\n{result.stderr}"
    if re.search(r"(?<![\w-])--fp16(?![\w-])", output):
        return True
    if "--onnx" in output or "--saveEngine" in output:
        return False
    return None


def trtexec_requires_modelopt_fp16(trtexec: str) -> bool:
    has_fp16 = trtexec_has_fp16_flag(trtexec)
    if has_fp16 is not None:
        return not has_fp16

    major = get_trtexec_major_version(trtexec)
    if major is not None:
        return major >= 11

    raise SystemExit(
        "Could not determine whether trtexec supports --fp16. "
        "Pass --trtexec /path/to/trtexec explicitly."
    )


def onnx_inputs_are_fp16(onnx_path: Path) -> bool:
    import onnx

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    input_types = {
        inp.type.tensor_type.elem_type
        for inp in onnx_model.graph.input
        if inp.type.HasField("tensor_type")
    }
    return bool(input_types) and input_types == {onnx.TensorProto.FLOAT16}


def autocast_onnx_fp16(onnx_path: Path, output_path: Path, args) -> Path:
    if onnx_inputs_are_fp16(onnx_path):
        print(f"ONNX inputs are already FP16; skipping ModelOpt AutoCast: {onnx_path}")
        return onnx_path

    check_onnxruntime_compat()

    command = [
        sys.executable,
        "-m",
        "modelopt.onnx.autocast",
        "--onnx_path",
        str(onnx_path),
        "--output_path",
        str(output_path),
        "--low_precision_type",
        "fp16",
    ]
    if getattr(args, "verbose", False):
        command.extend(["--log_level", "DEBUG"])
    if getattr(args, "autocast_args", ""):
        command.extend(shlex.split(args.autocast_args))

    print(f"Converting ONNX to mixed FP16 with NVIDIA ModelOpt AutoCast: {onnx_path}")
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(f"Failed to run ModelOpt: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"ModelOpt AutoCast failed with exit code {exc.returncode}.\n"
            'Install ModelOpt with: pip install -U "nvidia-modelopt[onnx]"\n'
            "If your environment pins PyTorch/ONNX Runtime, use a compatible "
            "ModelOpt version or convert the ONNX on another machine and pass --no-autocast."
        ) from exc

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise SystemExit(f"ModelOpt AutoCast did not create: {output_path}")

    import onnx

    onnx.checker.check_model(onnx.load(str(output_path)))
    print(f"FP16 ONNX: {output_path}")
    return output_path


def build_engine(onnx_path: Path, engine_path: Path, args):
    trtexec = find_trtexec(args.trtexec)

    if not args.skip_power_setup:
        setup_max_performance()

    no_autocast = getattr(args, "no_autocast", False)
    fp16_requires_modelopt = (
        args.precision == "fp16" and trtexec_requires_modelopt_fp16(trtexec)
    )
    use_modelopt = fp16_requires_modelopt and not no_autocast

    build_onnx = onnx_path
    if use_modelopt:
        build_onnx = autocast_onnx_fp16(
            onnx_path,
            engine_path.parent / f"{onnx_path.stem}_autocast_fp16.onnx",
            args,
        )
    elif fp16_requires_modelopt and no_autocast:
        print(
            "Warning: --no-autocast is set with a TensorRT version that no longer "
            "supports --fp16; building the ONNX as-is. Ensure the ONNX was already "
            "converted to FP16."
        )

    command = [
        trtexec,
        f"--onnx={build_onnx}",
        f"--saveEngine={engine_path}",
    ]

    if args.precision == "fp16" and not fp16_requires_modelopt:
        command.append("--fp16")

    if args.workspace_mb > 0:
        major = get_trtexec_major_version(trtexec)
        if major is not None and major < 9:
            command.append(f"--workspace={args.workspace_mb}")
        else:
            command.append(f"--memPoolSize=workspace:{args.workspace_mb}MiB")

    if args.verbose:
        command.append("--verbose")

    if args.trt_args:
        command.extend(shlex.split(args.trt_args))

    print(f"Building TensorRT {args.precision.upper()} engine: {engine_path}")
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"trtexec failed with exit code {exc.returncode}") from exc

    if not engine_path.is_file() or engine_path.stat().st_size == 0:
        raise SystemExit(f"TensorRT engine was not created: {engine_path}")

    print(f"TensorRT engine: {engine_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert best_model.pth to ONNX and TensorRT FP16."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CLASSES),
        default=None,
        help="Model architecture. Inferred from the log directory name if omitted.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="Training log directory containing config.py and best_model.pth.",
    )
    parser.add_argument(
        "--pth",
        type=Path,
        default=None,
        help="Path to the .pth model. Defaults to <logdir>/best_model.pth.",
    )
    parser.add_argument(
        "--mode",
        choices=["onnx", "tensorrt", "both"],
        default="both",
        help="Convert to ONNX only, TensorRT only, or both (default: both).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <logdir>/compiled.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Static batch size used for export (default: 1).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
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
        "--onnx-path",
        type=Path,
        default=None,
        help="Existing ONNX file to use with --mode tensorrt.",
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
    check_conda_env()
    preflight(args)

    if args.logdir:
        logdir = args.logdir.expanduser()
    else:
        from config import select_logdir

        logdir = select_logdir()
    logdir = logdir.resolve()
    if not logdir.is_dir():
        raise SystemExit(f"Log directory not found: {logdir}")

    config = load_run_config(logdir)
    model_name = resolve_model_name(args.model, logdir)
    pth = (args.pth.expanduser() if args.pth else logdir / "best_model.pth").resolve()
    if not pth.is_file():
        raise SystemExit(f"Model weights not found: {pth}")

    output_dir = (
        args.output_dir.expanduser() if args.output_dir else logdir / "compiled"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pth.stem

    print(f"Log directory: {logdir}")
    print(f"Model: {model_name}")
    print(f"Weights: {pth}")
    print(f"Mode: {args.mode}")
    print(f"Output directory: {output_dir}")

    model = build_model(model_name, config)
    model.load_state_dict(torch.load(pth, map_location=config.gpu_device))
    model.eval()

    if args.mode in ("onnx", "both") or (
        args.mode == "tensorrt" and args.onnx_path is None
    ):
        onnx_path = export_onnx(model, config, args, output_dir, stem)
    else:
        onnx_path = args.onnx_path.expanduser().resolve()
        if not onnx_path.is_file():
            raise SystemExit(f"ONNX file not found: {onnx_path}")
        import onnx

        onnx.checker.check_model(onnx.load(str(onnx_path)))
        describe_onnx_inputs(onnx_path)

    if args.no_simplify:
        simplified_path = onnx_path
    else:
        simplified_path = simplify_onnx(onnx_path, output_dir / f"{stem}_sim.onnx")

    if args.mode in ("tensorrt", "both"):
        engine_path = output_dir / f"{stem}_{args.precision}.engine"
        build_engine(simplified_path, engine_path, args)

    print("Done.")
    print(f"ONNX: {onnx_path}")
    if not args.no_simplify:
        print(f"Simplified ONNX: {simplified_path}")
    if args.mode in ("tensorrt", "both"):
        print(f"TensorRT engine: {output_dir / f'{stem}_{args.precision}.engine'}")


if __name__ == "__main__":
    main()
