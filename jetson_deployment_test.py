from __future__ import annotations

import os
import sys
import ast
import traceback
import argparse
from collections import OrderedDict
import time
from pathlib import Path
from typing import Any, cast
import importlib.util

import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2
import yaml

torch: Any = None
F: Any = None
DataLoader: Any = None
KarrDataset: Any = None
ai23: Any = None
ai33: Any = None
xr20: Any = None


def _select_logdir(log_root=None):
    log_root = Path(log_root) if log_root else (Path(__file__).resolve().parent / "log")
    runs = [d for d in log_root.iterdir() if d.is_dir()] if log_root.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"No log runs found under {log_root}")
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    print(f"Available log runs under {log_root}:")
    for i, d in enumerate(runs, 1):
        incomplete = []
        if not (d / "config.py").is_file():
            incomplete.append("missing config.py")
        if not (d / "best_model.pth").is_file():
            incomplete.append("missing best_model.pth")
        suffix = f"  <-- incomplete: {', '.join(incomplete)}" if incomplete else ""
        print(f"  [{i}] {d.name}{suffix}")
    while True:
        choice = input(f"Select log run [1-{len(runs)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(runs):
            return runs[int(choice) - 1]
        print("Invalid selection, please try again.")


def _read_gpu_id(config_path: Path, default: str | None = None) -> str | None:
    try:
        tree = ast.parse(config_path.read_text())
    except Exception:
        return default
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "gpu_id":
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    return default

#Class untuk penyimpanan dan perhitungan update metric
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

SYSTEM_METRIC_KEYS = [
    'cpu_util_pct', 'gpu_util_pct',
    'power_w', 'power_avg_w',
    'ram_used_mb', 'vram_shared_mb', 'vram_torch_mb',
    'gpu_freq_mhz', 'gpu_temp_c', 'cpu_temp_c',
]

NUMERIC_LOG_KEYS = [
    'test_metric', 'test_wp_metric',
    'model_elapsed_time', 'model_fps', 'model_energy_j', 'model_fps_per_w',
    'pipeline_elapsed_time', 'pipeline_fps', 'pipeline_energy_j', 'pipeline_fps_per_w',
    'power_w', 'power_avg_w', 'cpu_util_pct', 'gpu_util_pct',
    'ram_used_mb', 'vram_shared_mb', 'vram_torch_mb',
    'gpu_freq_mhz', 'gpu_temp_c', 'cpu_temp_c',
]

def scoped_dataframe(log, scope: str):
    df = pd.DataFrame(log)
    if scope == "model":
        df = df.drop(columns=['pipeline_elapsed_time', 'pipeline_fps', 'pipeline_energy_j', 'pipeline_fps_per_w'], errors='ignore')
        rename = {key: f"model_{key}" for key in SYSTEM_METRIC_KEYS}
    else:
        df = df.drop(columns=['model_elapsed_time', 'model_fps', 'model_energy_j', 'model_fps_per_w'], errors='ignore')
        rename = {key: f"pipeline_{key}" for key in SYSTEM_METRIC_KEYS}
    return df.rename(columns=rename)

def jetson_stat(jetson, config: Any):
    stat = {
        'power_w': float('nan'),
        'power_avg_w': float('nan'),
        'cpu_util_pct': float('nan'),
        'gpu_util_pct': float('nan'),
        'ram_used_mb': float('nan'),
        'vram_shared_mb': float('nan'),
        'vram_torch_mb': float('nan'),
        'gpu_freq_mhz': float('nan'),
        'gpu_temp_c': float('nan'),
        'cpu_temp_c': float('nan'),
    }
    if jetson is not None:
        try:
            if jetson.ok(spin=True):
                power = jetson.power['tot']
                stat['power_w'] = power['power'] / 1000.0
                stat['power_avg_w'] = power['avg'] / 1000.0 if 'avg' in power else stat['power_w']

                cpu = jetson.cpu['total']
                stat['cpu_util_pct'] = min(100.0, max(0.0, cpu['user'] + cpu['nice'] + cpu['system']))

                for name in jetson.gpu:
                    gpu_info = jetson.gpu[name]
                    if 'status' in gpu_info and 'load' in gpu_info['status']:
                        stat['gpu_util_pct'] = float(gpu_info['status']['load'])
                    if 'freq' in gpu_info and 'cur' in gpu_info['freq']:
                        freq = gpu_info['freq']['cur']
                        stat['gpu_freq_mhz'] = freq / 1000.0 if freq > 10000 else float(freq)
                    break

                ram = jetson.memory['RAM']
                stat['ram_used_mb'] = ram['used'] / 1024.0
                stat['vram_shared_mb'] = ram['shared'] / 1024.0 if 'shared' in ram else 0.0

                temps = jetson.temperature
                if 'gpu' in temps and temps['gpu'].get('online', False):
                    stat['gpu_temp_c'] = float(temps['gpu']['temp'])
                if 'cpu' in temps and temps['cpu'].get('online', False):
                    stat['cpu_temp_c'] = float(temps['cpu']['temp'])
        except Exception as e:
            print(f"[jtop] failed to read stats: {e}")
            traceback.print_exc()

    try:
        stat['vram_torch_mb'] = torch.cuda.memory_allocated(config.gpu_device) / (1024.0 ** 2)
    except Exception:
        pass
    return stat

def measure_baseline_power(jetson, samples=5, interval=0.2):
    if jetson is None:
        return float('nan')
    readings = []
    for _ in range(samples):
        try:
            if jetson.ok(spin=True):
                power = jetson.power['tot']
                readings.append((power['avg'] if 'avg' in power else power['power']) / 1000.0)
        except Exception as e:
            print(f"[jtop] failed to read baseline power: {e}")
            traceback.print_exc()
        time.sleep(interval)
    return float(np.mean(readings)) if readings else float('nan')

def test(data_loader, model, config: Any, jetson=None, warmup_batches=3, scope="pipeline", output_dir: Path | None = None):
    score = {
        'total_metric': AverageMeter(),
        'wp_metric': AverageMeter()
    }

    log: dict[str, list[Any]] = OrderedDict([
        ('batch', []),
        ('warmup', []),
        ('test_metric', []),
        ('test_wp_metric', []),
        ('model_elapsed_time', []),
        ('model_fps', []),
        ('model_energy_j', []),
        ('model_fps_per_w', []),
        ('pipeline_elapsed_time', []),
        ('pipeline_fps', []),
        ('pipeline_energy_j', []),
        ('pipeline_fps_per_w', []),
        ('power_w', []),
        ('power_avg_w', []),
        ('cpu_util_pct', []),
        ('gpu_util_pct', []),
        ('ram_used_mb', []),
        ('vram_shared_mb', []),
        ('vram_torch_mb', []),
        ('gpu_freq_mhz', []),
        ('gpu_temp_c', []),
        ('cpu_temp_c', []),
    ])
    batch_ke = 1

    save_dir = output_dir if output_dir is not None else config.logdir / "offline_test"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir_log = save_dir
    save_path = save_dir_log / f'test_log_{config.select_route}_{scope}.csv'
    save_dir_meta = save_dir / 'pred_meta' / scope

    model.eval()
    baseline_power_w = measure_baseline_power(jetson)
    if np.isfinite(baseline_power_w):
        print(f"[jtop] baseline power: {baseline_power_w:.2f} W")

    meta_preds = []
    prog_bar = tqdm(total=len(data_loader))

    try:
        with torch.no_grad():
            data_iter = iter(data_loader)

            for _ in range(len(data_loader)):
                data = next(data_iter)
                fetch_end_time = time.perf_counter()
                lidar_elapsed_time = data['lidar_elapsed_time'].item()

                bev_segs = []
                bev_deps = []
                front_segs = []
                front_deps = []
                for i in range(0, config.seq_len):
                    bev_segs.append(data['bev_segs'][i].to(config.gpu_device, dtype=config.dtype))
                    bev_deps.append(data['bev_deps'][i].to(config.gpu_device, dtype=config.dtype))
                    front_segs.append(data['front_segs'][i].to(config.gpu_device, dtype=config.dtype))
                    front_deps.append(data['front_deps'][i].to(config.gpu_device, dtype=config.dtype))

                rp1 = torch.stack(data['rp1'], dim=1).to(config.gpu_device, dtype=config.dtype)
                rp2 = torch.stack(data['rp2'], dim=1).to(config.gpu_device, dtype=config.dtype)
                gt_velocity = data['velocity'].to(config.gpu_device, dtype=config.dtype)
                if gt_velocity.dim() == 1:
                    gt_velocity = gt_velocity.unsqueeze(1)
                gt_waypoints = [torch.stack(data['waypoints'][j], dim=1).to(config.gpu_device, dtype=config.dtype) for j in range(0, config.pred_len)]
                gt_waypoints = torch.stack(gt_waypoints, dim=1).to(config.gpu_device, dtype=config.dtype)

                #forward pass
                torch.cuda.synchronize(config.gpu_device)
                model_start_time = time.perf_counter()
                pred_wp = model(bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, gt_velocity)
                torch.cuda.synchronize(config.gpu_device)
                model_elapsed_time = time.perf_counter() - model_start_time
                pipeline_elapsed_time = lidar_elapsed_time + (time.perf_counter() - fetch_end_time)

                #baca statistik jetson (daya, cpu, gpu, ram, vram)
                stat = jetson_stat(jetson, config)

                #hitung metrik deploy turunan (energi neto memakai baseline daya idle)
                is_warmup = batch_ke <= warmup_batches
                model_fps = 1.0 / model_elapsed_time if model_elapsed_time > 0 else float('nan')
                pipeline_fps = 1.0 / pipeline_elapsed_time if pipeline_elapsed_time > 0 else float('nan')
                net_power_w = stat['power_avg_w'] - baseline_power_w
                if np.isfinite(net_power_w):
                    model_energy_j = net_power_w * model_elapsed_time
                    pipeline_energy_j = net_power_w * pipeline_elapsed_time
                else:
                    model_energy_j = float('nan')
                    pipeline_energy_j = float('nan')
                model_fps_per_w = model_fps / net_power_w if np.isfinite(net_power_w) and net_power_w > 0 else float('nan')
                pipeline_fps_per_w = pipeline_fps / net_power_w if np.isfinite(net_power_w) and net_power_w > 0 else float('nan')

                #compute metric
                metric_wp = F.l1_loss(pred_wp.to(gt_waypoints.dtype), gt_waypoints)
                total_metric = metric_wp.item()

                score['total_metric'].update(total_metric)
                score['wp_metric'].update(metric_wp.item())

                #update visualisasi progress bar
                postfix = OrderedDict([
                    ('te_total_m', score['total_metric'].avg),
                    ('te_wp_m', score['wp_metric'].avg),
                    ('model_fps', model_fps),
                    ('pipeline_fps', pipeline_fps),
                    ('power_avg_w', stat['power_avg_w']),
                    ('model_j', model_energy_j),
                    ('model_fps_w', model_fps_per_w),
                    ('pipeline_j', pipeline_energy_j),
                    ('pipeline_fps_w', pipeline_fps_per_w),
                    ('cpu_%', stat['cpu_util_pct']),
                    ('gpu_%', stat['gpu_util_pct']),
                ])

                #simpan history test ke file csv, ambil dari hasil kalkulasi metric langsung, jangan dari averagemeter
                log['batch'].append(batch_ke)
                log['warmup'].append(is_warmup)
                log['test_metric'].append(total_metric)
                log['test_wp_metric'].append(metric_wp.item())
                log['model_elapsed_time'].append(model_elapsed_time)
                log['model_fps'].append(model_fps)
                log['model_energy_j'].append(model_energy_j)
                log['model_fps_per_w'].append(model_fps_per_w)
                log['pipeline_elapsed_time'].append(pipeline_elapsed_time)
                log['pipeline_fps'].append(pipeline_fps)
                log['pipeline_energy_j'].append(pipeline_energy_j)
                log['pipeline_fps_per_w'].append(pipeline_fps_per_w)
                log['power_w'].append(stat['power_w'])
                log['power_avg_w'].append(stat['power_avg_w'])
                log['cpu_util_pct'].append(stat['cpu_util_pct'])
                log['gpu_util_pct'].append(stat['gpu_util_pct'])
                log['ram_used_mb'].append(stat['ram_used_mb'])
                log['vram_shared_mb'].append(stat['vram_shared_mb'])
                log['vram_torch_mb'].append(stat['vram_torch_mb'])
                log['gpu_freq_mhz'].append(stat['gpu_freq_mhz'])
                log['gpu_temp_c'].append(stat['gpu_temp_c'])
                log['cpu_temp_c'].append(stat['cpu_temp_c'])

                #isikan beberapa data
                meta_pred = {}
                meta_pred['rp1_pos_local'] = rp1[0].cpu().detach().numpy().tolist()
                meta_pred['rp2_pos_local'] = rp2[0].cpu().detach().numpy().tolist()
                meta_pred['rp1_pos_global'] = np.array([data['rp1_lat'].item(), data['rp1_lon'].item()]).tolist()
                meta_pred['rp2_pos_global'] = np.array([data['rp2_lat'].item(), data['rp2_lon'].item()]).tolist()
                meta_pred['robot_bearing'] = float(data['bearing_robot'].item())
                meta_pred['robot_pos_global'] = np.array([data['lat_robot'].item(), data['lon_robot'].item()]).tolist()
                meta_pred['scope'] = scope
                if scope == "model":
                    meta_pred['model_fps'] = float(model_fps)
                    meta_pred['model_energy_j'] = float(model_energy_j)
                    meta_pred['model_fps_per_w'] = float(model_fps_per_w)
                else:
                    meta_pred['model_fps'] = float(model_fps)
                    meta_pred['pipeline_fps'] = float(pipeline_fps)
                    meta_pred['pipeline_energy_j'] = float(pipeline_energy_j)
                    meta_pred['pipeline_fps_per_w'] = float(pipeline_fps_per_w)
                frame_name = data['filename'][0] if isinstance(data['filename'], (list, tuple)) else data['filename']
                meta_preds.append((frame_name, meta_pred))

                batch_ke += 1
                prog_bar.set_postfix(postfix) # type: ignore
                prog_bar.update(1)

        #batas sampel setelah warm-up
        warm = min(warmup_batches, max(len(log['batch']) - 1, 0))
        clean = {key: np.asarray(log[key][warm:], dtype=float) for key in NUMERIC_LOG_KEYS}

        def _mean(key):
            values = clean[key]
            return float(np.nanmean(values)) if values.size else float('nan')

        def _std(key):
            values = clean[key]
            return float(np.nanstd(values)) if values.size else float('nan')

        def _pct(key, q):
            values = clean[key]
            return float(np.nanpercentile(values, q)) if values.size else float('nan')

        def _max(key):
            values = clean[key]
            return float(np.nanmax(values)) if values.size else float('nan')

        def _safe_fps(elapsed):
            return 1.0 / elapsed if np.isfinite(elapsed) and elapsed > 0 else float('nan')

        def _append_summary(label, values):
            log['batch'].append(label)
            log['warmup'].append(float('nan'))
            for key in NUMERIC_LOG_KEYS:
                log[key].append(values.get(key, float('nan')))

        avg_model_elapsed = _mean('model_elapsed_time')
        avg_pipeline_elapsed = _mean('pipeline_elapsed_time')
        _append_summary("avg", {
            'test_metric': _mean('test_metric'),
            'test_wp_metric': _mean('test_wp_metric'),
            'model_elapsed_time': avg_model_elapsed,
            'model_fps': _safe_fps(avg_model_elapsed),
            'model_energy_j': _mean('model_energy_j'),
            'model_fps_per_w': _mean('model_fps_per_w'),
            'pipeline_elapsed_time': avg_pipeline_elapsed,
            'pipeline_fps': _safe_fps(avg_pipeline_elapsed),
            'pipeline_energy_j': _mean('pipeline_energy_j'),
            'pipeline_fps_per_w': _mean('pipeline_fps_per_w'),
            'power_w': _mean('power_w'),
            'power_avg_w': _mean('power_avg_w'),
            'cpu_util_pct': _mean('cpu_util_pct'),
            'gpu_util_pct': _mean('gpu_util_pct'),
            'ram_used_mb': _mean('ram_used_mb'),
            'vram_shared_mb': _mean('vram_shared_mb'),
            'vram_torch_mb': _mean('vram_torch_mb'),
            'gpu_freq_mhz': _mean('gpu_freq_mhz'),
            'gpu_temp_c': _mean('gpu_temp_c'),
            'cpu_temp_c': _mean('cpu_temp_c'),
        })

        _append_summary("stddev", {
            'test_metric': _std('test_metric'),
            'test_wp_metric': _std('test_wp_metric'),
            'model_elapsed_time': _std('model_elapsed_time'),
            'model_fps': _std('model_fps'),
            'model_energy_j': _std('model_energy_j'),
            'model_fps_per_w': _std('model_fps_per_w'),
            'pipeline_elapsed_time': _std('pipeline_elapsed_time'),
            'pipeline_fps': _std('pipeline_fps'),
            'pipeline_energy_j': _std('pipeline_energy_j'),
            'pipeline_fps_per_w': _std('pipeline_fps_per_w'),
            'power_w': _std('power_w'),
            'power_avg_w': _std('power_avg_w'),
            'cpu_util_pct': _std('cpu_util_pct'),
            'gpu_util_pct': _std('gpu_util_pct'),
            'ram_used_mb': _std('ram_used_mb'),
            'vram_shared_mb': _std('vram_shared_mb'),
            'vram_torch_mb': _std('vram_torch_mb'),
            'gpu_freq_mhz': _std('gpu_freq_mhz'),
            'gpu_temp_c': _std('gpu_temp_c'),
            'cpu_temp_c': _std('cpu_temp_c'),
        })

        for label, q in (("p50", 50), ("p95", 95), ("p99", 99)):
            model_pct = _pct('model_elapsed_time', q)
            pipeline_pct = _pct('pipeline_elapsed_time', q)
            _append_summary(label, {
                'model_elapsed_time': model_pct,
                'model_fps': _safe_fps(model_pct),
                'model_energy_j': _pct('model_energy_j', q),
                'model_fps_per_w': _pct('model_fps_per_w', q),
                'pipeline_elapsed_time': pipeline_pct,
                'pipeline_fps': _safe_fps(pipeline_pct),
                'pipeline_energy_j': _pct('pipeline_energy_j', q),
                'pipeline_fps_per_w': _pct('pipeline_fps_per_w', q),
            })

        _append_summary("max", {
            'ram_used_mb': _max('ram_used_mb'),
            'vram_shared_mb': _max('vram_shared_mb'),
            'vram_torch_mb': _max('vram_torch_mb'),
        })
    finally:
        prog_bar.close()
        try:
            scoped_dataframe(log, scope).to_csv(save_path, index=False)
        except Exception as e:
            print(f"failed to save csv log: {e}")
            traceback.print_exc()
        try:
            save_dir_meta.mkdir(parents=True, exist_ok=True)
            for frame_name, meta_pred in meta_preds:
                with open(save_dir_meta / f"{frame_name}.yml", 'w') as dict_file:
                    yaml.dump(meta_pred, dict_file)
        except Exception as e:
            print(f"failed to save prediction metadata: {e}")
            traceback.print_exc()

    return log

def main():
    global torch, F, DataLoader, KarrDataset, ai23, ai33, xr20

    parser = argparse.ArgumentParser(description="Model benchmarking in Jetson board")
    parser.add_argument("--route", action='append', type=str, help='set routes, you can combine multiple routes. e.g: --route "a" --route "b" ...')
    parser.add_argument("--warmup", type=int, default=3, help='number of warm-up batches excluded from summary stats (default: 3)')
    parser.add_argument("--mode", choices=["model", "pipeline", "both"], default="both", help='benchmark scope: model only, pipeline (PolarSeg+projection+model), or both (default: both)')
    parser.add_argument("--logdir", type=Path, default=None, help='path to a specific log run directory containing config.py and best_model.pth; when omitted, the interactive selector is used')
    args = parser.parse_args()

    if args.logdir is not None:
        logdir = args.logdir.expanduser().resolve()
        if not logdir.is_dir():
            parser.error(f"--logdir is not a directory: {logdir}")
        missing = [name for name in ("config.py", "best_model.pth") if not (logdir / name).is_file()]
        if missing:
            parser.error(f"--logdir {logdir} is incomplete, missing: {', '.join(missing)}")
    else:
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
    from xr20_model import xr20
    from ai23_model import ai23
    from ai33_model import ai33
    from jetson_deployment_dataloader import KarrDataset
    from config import set_seed

    torch.backends.cudnn.benchmark = True

    # Load config from the selected log run
    config_path = logdir / "config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    assert spec is not None and spec.loader is not None, f"Failed to load spec/loader from {config_path}"
    log_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(log_config)
    config = cast(Any, log_config.GlobalConfig())
    config.logdir = logdir

    set_seed(getattr(config, "seed", 42))

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

    #IMPORT MODEL dan load bobot
    model_name = config.logdir.name.split('_', 1)[0] # ambil nama model dari nama folder log
    print("IMPORT ARSITEKTUR DL DAN COMPILE")
    print("current model of Logdir: ", model_name)
    model = None
    if model_name == "ai23":
        model = ai23(config, device=config.gpu_device).to(config.gpu_device, dtype=config.dtype)
    elif model_name == "ai33":
        model = ai33(config, device=config.gpu_device).to(config.gpu_device, dtype=config.dtype)
    elif model_name == "xr20":
        model = xr20(config, device=config.gpu_device).to(config.gpu_device, dtype=config.dtype)
    assert model is not None, f"Unknown model architecture: {model_name}"
    model.load_state_dict(torch.load(config.logdir / 'best_model.pth'))

    modes = ["model", "pipeline"] if args.mode == "both" else [args.mode]

    for mode in modes:
        test_lidar_from_scratch = (mode == "pipeline")

        karr_dataset = KarrDataset(config=config, phase="test", test_lidar_from_scratch=test_lidar_from_scratch)

        dataloader_test = DataLoader(karr_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        #mulai jtop untuk log daya, cpu, gpu, ram, dan vram
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

        #tampilkan info board dan mode daya
        if jetson is not None:
            try:
                if jetson.ok(spin=True):
                    board = jetson.board.get('hardware', {})
                    print(f"[jtop] board={board.get('Model', 'unknown')} "
                          f"nvpmodel={jetson.nvpmodel} jetson_clocks={jetson.jetson_clocks} "
                          f"interval={jetson.interval}s")
            except Exception as e:
                print(f"[jtop] failed to read board metadata: {e}")

        #test
        try:
            test_log = test(dataloader_test, model, config, jetson, args.warmup, scope=mode)
        finally:
            if jetson is not None:
                jetson.close()

        #kosongkan cuda chace
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
