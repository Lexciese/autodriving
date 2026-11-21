from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

import jetson_deployment_test as jdt
from jetson_deployment_test import _read_gpu_id, _select_logdir


class TensorRTEngineBase:
    INPUT_NAMES: tuple[str, ...] = ()
    OUTPUT_NAME: str = ""

    def __init__(self, engine_path: Path, config: Any):
        import tensorrt as trt
        import torch

        self._trt = trt
        self._torch = torch
        self._device = config.gpu_device
        self.engine_path = engine_path

        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as engine_file:
            engine_bytes = engine_file.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {engine_path}")

        self.input_buffers: dict[str, Any] = {}
        self.output_buffers: dict[str, Any] = {}
        self._binding_names: dict[str, int] = {}
        self._binding_order: list[str] = []
        self._load_bindings()

        if self.OUTPUT_NAME not in self.output_buffers:
            raise RuntimeError(
                f"TensorRT engine output '{self.OUTPUT_NAME}' not found; "
                f"outputs: {sorted(self.output_buffers)}"
            )

    def _iter_io(self):
        engine = self.engine
        trt = self._trt
        if hasattr(engine, "num_io_tensors"):
            for index in range(engine.num_io_tensors):
                name = engine.get_tensor_name(index)
                mode = engine.get_tensor_mode(name)
                yield (
                    name,
                    mode == trt.TensorIOMode.INPUT,
                    tuple(engine.get_tensor_shape(name)),
                    engine.get_tensor_dtype(name),
                    index,
                )
        else:
            for index in range(engine.num_bindings):
                name = engine.get_binding_name(index)
                yield (
                    name,
                    engine.binding_is_input(index),
                    tuple(engine.get_binding_shape(index)),
                    engine.get_binding_dtype(index),
                    index,
                )

    def _torch_dtype(self, trt_dtype):
        torch = self._torch
        dtype_name = str(trt_dtype).split(".")[-1].upper()
        mapping = {
            "FLOAT": torch.float32,
            "HALF": torch.float16,
            "INT32": torch.int32,
            "INT8": torch.int8,
            "BOOL": torch.bool,
            "UINT8": torch.uint8,
            "INT64": torch.int64,
        }
        if dtype_name in mapping:
            return mapping[dtype_name]
        try:
            numpy_dtype = self._trt.nptype(trt_dtype)
            return torch.from_numpy(np.empty(0, dtype=numpy_dtype)).dtype
        except Exception as exc:
            raise RuntimeError(f"Unsupported TensorRT dtype: {trt_dtype}") from exc

    def _load_bindings(self):
        torch = self._torch
        for name, is_input, shape, dtype, binding_index in self._iter_io():
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"TensorRT engine has dynamic shape for '{name}': {shape}. "
                    "Rebuild with postprocess_model_compile.py using static shapes."
                )
            buffer = torch.empty(
                shape,
                dtype=self._torch_dtype(dtype),
                device=self._device,
            )
            if is_input:
                self.input_buffers[name] = buffer
            else:
                self.output_buffers[name] = buffer
            self._binding_names[name] = binding_index
            print(
                f"[tensorrt] {'input' if is_input else 'output'} "
                f"{name}: shape={shape} dtype={buffer.dtype}"
            )
        self._binding_order = [
            name for name, _ in sorted(self._binding_names.items(), key=lambda item: item[1])
        ]

    def _binding_buffers(self):
        return [
            self.input_buffers[name]
            if name in self.input_buffers
            else self.output_buffers[name]
            for name in self._binding_order
        ]

    def eval(self):
        return self

    def _run(self, inputs: dict[str, Any]):
        torch = self._torch

        for name, buffer in self.input_buffers.items():
            source = inputs.get(name)
            if source is None:
                raise RuntimeError(f"Missing input tensor for TensorRT binding '{name}'")
            if tuple(source.shape) != tuple(buffer.shape):
                raise RuntimeError(
                    f"Input shape mismatch for '{name}': "
                    f"engine={tuple(buffer.shape)} data={tuple(source.shape)}"
                )
            buffer.copy_(source.to(device=buffer.device, dtype=buffer.dtype))

        bindings = [int(buffer.data_ptr()) for buffer in self._binding_buffers()]
        stream = torch.cuda.current_stream(self._device).cuda_stream

        if hasattr(self.context, "set_tensor_address"):
            for name, buffer in self.input_buffers.items():
                self.context.set_tensor_address(name, int(buffer.data_ptr()))
            for name, buffer in self.output_buffers.items():
                self.context.set_tensor_address(name, int(buffer.data_ptr()))

        if hasattr(self.context, "execute_async_v3"):
            success = self.context.execute_async_v3(stream_handle=stream)
        else:
            success = self.context.execute_async_v2(bindings=bindings, stream_handle=stream)

        if success is False:
            raise RuntimeError("TensorRT execution failed")
        return self.output_buffers[self.OUTPUT_NAME]


class TensorRTModel(TensorRTEngineBase):
    INPUT_NAMES = (
        "bev_segs",
        "bev_deps",
        "front_segs",
        "front_deps",
        "rp1",
        "rp2",
        "velocity",
    )
    OUTPUT_NAME = "pred_wp"

    def __call__(self, bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, velo):
        return self.forward(bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, velo)

    def forward(self, bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, velo):
        torch = self._torch
        inputs = {
            "bev_segs": torch.stack(bev_segs, dim=1),
            "bev_deps": torch.stack(bev_deps, dim=1),
            "front_segs": torch.stack(front_segs, dim=1),
            "front_deps": torch.stack(front_deps, dim=1),
            "rp1": rp1,
            "rp2": rp2,
            "velocity": velo,
        }
        return self._run(inputs)


class TensorRTBEVModel(TensorRTEngineBase):
    INPUT_NAMES = ("out_data",)
    OUTPUT_NAME = "logits"

    def __call__(self, out_data):
        return self.forward(out_data)

    def forward(self, out_data):
        return self._run({"out_data": out_data})


def _load_log_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    assert spec is not None and spec.loader is not None, f"Failed to load spec/loader from {config_path}"
    log_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(log_config)
    config = cast(Any, log_config.GlobalConfig())
    config.logdir = config_path.parent
    return config


def _find_engine(logdir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        engine_path = explicit.expanduser().resolve()
        if not engine_path.is_file():
            raise SystemExit(f"TensorRT engine not found: {engine_path}")
        return engine_path

    compiled_dir = logdir / "compiled"
    for candidate in (
        compiled_dir / "best_model_fp16.engine",
        compiled_dir / "best_model_fp32.engine",
    ):
        if candidate.is_file():
            return candidate

    for pattern in ("*_fp16.engine", "*_fp32.engine", "*.engine"):
        engines = sorted(compiled_dir.glob(pattern)) if compiled_dir.is_dir() else []
        if len(engines) == 1:
            return engines[0]
        if len(engines) > 1:
            candidates = ", ".join(str(path) for path in engines)
            raise SystemExit(
                f"Multiple TensorRT engines found under {compiled_dir}: {candidates}. "
                "Select one with --engine PATH."
            )

    raise SystemExit(
        f"No TensorRT engine found under {compiled_dir}. Build one with: "
        f"python postprocess_model_compile.py --logdir {logdir} --mode tensorrt"
    )


def _find_polarseg_engine(pth: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        engine_path = explicit.expanduser().resolve()
        if not engine_path.is_file():
            raise SystemExit(f"PolarSeg TensorRT engine not found: {engine_path}")
        return engine_path

    parent = pth.parent
    stem = pth.stem
    for candidate in (
        parent / f"{stem}_bev_fp16.engine",
        parent / f"{stem}_bev_fp32.engine",
        parent / f"{stem}_bev.engine",
    ):
        if candidate.is_file():
            return candidate

    for pattern in (f"{stem}_bev*.engine", "*_bev*.engine", "*.engine"):
        engines = sorted(parent.glob(pattern)) if parent.is_dir() else []
        if len(engines) == 1:
            return engines[0]
        if len(engines) > 1:
            candidates = ", ".join(str(path) for path in engines)
            raise SystemExit(
                f"Multiple PolarSeg engines found under {parent}: {candidates}. "
                "Select one with --polarseg-engine PATH."
            )

    raise SystemExit(
        f"No PolarSeg BEV engine found under {parent}. Build one with: "
        f"python postprocess_polarseg_compile.py --pth {pth} --mode tensorrt"
    )


def main():
    parser = argparse.ArgumentParser(description="TensorRT benchmarking in Jetson board")
    parser.add_argument("--route", action='append', type=str, help='set routes, you can combine multiple routes. e.g: --route "a" --route "b" ...')
    parser.add_argument("--warmup", type=int, default=3, help='number of warm-up batches excluded from summary stats (default: 3)')
    parser.add_argument("--mode", choices=["model", "pipeline", "both"], default="both", help='benchmark scope: model only, pipeline (PolarSeg+projection+model), or both (default: both)')
    parser.add_argument("--engine", type=Path, default=None, help="TensorRT engine path. Auto-detected under <logdir>/compiled if omitted.")
    parser.add_argument("--polarseg-engine", type=Path, default=None, help="PolarSeg BEV TensorRT engine path. Auto-detected beside the PolarSeg pth if omitted.")
    args = parser.parse_args()

    logdir = _select_logdir()
    gpu_id = _read_gpu_id(logdir / "config.py") or os.environ.get("CUDA_VISIBLE_DEVICES") or "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        from jtop import jtop  # type: ignore
    except ImportError as e:
        print(f"[jtop] import failed, system metrics will be NaN: {e}")
        jtop = None

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from jetson_deployment_dataloader import KarrDataset
    from preprocess_lidar import GenerateLidSegDep

    try:
        import tensorrt as trt
    except ImportError as exc:
        raise SystemExit(
            "TensorRT Python bindings not found. On Jetson, install python3-libnvinfer "
            "or run with the system Python that provides the tensorrt module."
        ) from exc

    jdt.torch = torch
    jdt.F = F

    config = _load_log_config(logdir / "config.py")

    available_route = ["ugm_baru", "ringroad-711", "utara", "selatan"]
    if len(sys.argv) == 1:
        print("\nAvailable route: ")
        for i, route in enumerate(available_route, 1):
            print(f"  [{i}] {route}")
        choices = input("Select routes [e.g. 1 3]: ").split()
        config.select_route = [
            available_route[int(c) - 1]
            for c in dict.fromkeys(choices)
            if c.isdigit() and 0 < int(c) <= len(available_route)
        ]
    elif args.route:
        config.select_route = args.route
    print("Selected route: ", config.select_route)

    class TensorRTBEVModule(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, out_data):
            return self.model(out_data)

    class TensorRTLidSegDep(GenerateLidSegDep):
        def __init__(self, config, bev_engine_path):
            self._bev_engine_path = bev_engine_path
            super().__init__(config)

        def _init_model(self):
            model = super()._init_model()
            model.BEV_model = TensorRTBEVModule(
                TensorRTBEVModel(self._bev_engine_path, self.config)
            )
            return model

    class KarrDatasetTensorRT(KarrDataset):
        def __init__(self, config, bev_engine_path, **kwargs):
            self._bev_engine_path = bev_engine_path
            super().__init__(config=config, **kwargs)

        def _get_lidar_pipeline(self):
            if not hasattr(self, "_lidar_pipeline"):
                self._lidar_pipeline = TensorRTLidSegDep(
                    self.config, self._bev_engine_path
                )
            return self._lidar_pipeline

    modes = ["model", "pipeline"] if args.mode == "both" else [args.mode]
    needs_polarseg = "pipeline" in modes

    engine_path = _find_engine(config.logdir, args.engine)
    print("TensorRT version: ", getattr(trt, "__version__", "unknown"))
    print("TensorRT engine: ", engine_path)
    model = TensorRTModel(engine_path, config)

    polarseg_engine_path = None
    if needs_polarseg:
        polarseg_engine_path = _find_polarseg_engine(
            Path(config.polarseg_weight_path), args.polarseg_engine
        )
        print("PolarSeg TensorRT engine: ", polarseg_engine_path)

    for mode in modes:
        if mode == "pipeline":
            karr_dataset = KarrDatasetTensorRT(
                config=config,
                bev_engine_path=polarseg_engine_path,
                phase="test",
                test_lidar_from_scratch=True,
            )
        else:
            karr_dataset = KarrDataset(
                config=config,
                phase="test",
                test_lidar_from_scratch=False,
            )

        dataloader_test = DataLoader(karr_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        jetson = None
        if jtop is not None:
            try:
                jetson = jtop()
                jetson.start()
            except Exception as e:
                print(f"[jtop] unavailable, energy stats will be NaN: {e}")
                if "Mismatch version" in str(e):
                    print("[jtop] pin the conda client to the running service version, e.g. "
                          "pip install 'jetson-stats==7.2.2'; if the system package was upgraded, "
                          "also run: sudo systemctl restart jtop.service")
                jetson = None

        if jetson is not None:
            try:
                if jetson.ok(spin=True):
                    board = jetson.board.get('hardware', {})
                    print(f"[jtop] board={board.get('Model', 'unknown')} "
                          f"nvpmodel={jetson.nvpmodel} jetson_clocks={jetson.jetson_clocks} "
                          f"interval={jetson.interval}s")
            except Exception as e:
                print(f"[jtop] failed to read board metadata: {e}")

        try:
            jdt.test(
                dataloader_test,
                model,
                config,
                jetson,
                args.warmup,
                scope=mode,
                output_dir=config.logdir / "offline_test_tensorrt",
            )
        finally:
            if jetson is not None:
                jetson.close()

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
