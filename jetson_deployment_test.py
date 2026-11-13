import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
import cv2
from torch import torch
import yaml
from pathlib import Path
try:
    from jtop import jtop  # type: ignore
except ImportError:
    jtop = None

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from xr20_model import xr20
from ai23_model import ai23
from ai33_model import ai33
from jetson_deployment_dataloader import KarrDataset

# use the config from the log directory
from config import GlobalConfig, select_logdir
import importlib.util
from typing import cast

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

def scoped_dataframe(log, scope: str):
    df = pd.DataFrame(log)
    if scope == "model":
        df = df.drop(columns=['pipeline_elapsed_time', 'pipeline_fps', 'pipeline_energy_j', 'pipeline_fps_per_w'], errors='ignore')
        rename = {key: f"model_{key}" for key in SYSTEM_METRIC_KEYS}
    else:
        df = df.drop(columns=['model_energy_j', 'model_fps_per_w'], errors='ignore')
        rename = {key: f"pipeline_{key}" for key in SYSTEM_METRIC_KEYS}
    return df.rename(columns=rename)

def jetson_stat(jetson, config: GlobalConfig):
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
                stat['power_avg_w'] = power.get('avg', power['power']) / 1000.0
                cpu = jetson.cpu['total']
                stat['cpu_util_pct'] = min(100.0, max(0.0, cpu['user'] + cpu['nice'] + cpu['system']))
                stat['gpu_util_pct'] = float(jetson.stats.get('GPU', 0.0))
                ram = jetson.memory['RAM']
                stat['ram_used_mb'] = ram['used'] / 1024.0
                stat['vram_shared_mb'] = ram.get('shared', 0) / 1024.0
                for name in jetson.gpu:
                    freq = jetson.gpu[name].get('freq', {}).get('cur')
                    if freq:
                        stat['gpu_freq_mhz'] = freq / 1000.0 if freq > 10000 else float(freq)
                    break
                temps = jetson.temperature
                gpu_temp = temps.get('GPU', {}).get('temp')
                if gpu_temp is not None:
                    stat['gpu_temp_c'] = float(gpu_temp)
                cpu_temp = temps.get('CPU', {}).get('temp')
                if cpu_temp is not None:
                    stat['cpu_temp_c'] = float(cpu_temp)
        except Exception as e:
            print(f"[jtop] failed to read stats: {e}")
    try:
        stat['vram_torch_mb'] = torch.cuda.memory_allocated(config.gpu_device) / (1024.0 ** 2)
    except Exception:
        pass
    return stat

def test(data_loader, model, config: GlobalConfig, jetson=None, warmup_batches=3, scope="pipeline"):
    score = {
        'total_metric': AverageMeter(),
        'wp_metric': AverageMeter()
    }

    log = OrderedDict([
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

    save_dir = config.logdir / "offline_test"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir_log = save_dir
    save_path = save_dir_log / f'test_log_{config.select_route}_{scope}.csv'

    model.eval()

    with torch.no_grad():
        prog_bar = tqdm(total=len(data_loader))
        data_iter = iter(data_loader)

        for _ in range(len(data_loader)):
            data = next(data_iter)
            fetch_end_time = time.time()
            lidar_elapsed_time = data['lidar_elapsed_time'].item()

            bev_segs = []
            bev_deps = []
            front_segs = []
            front_deps = []
            for i in range(0, config.seq_len):
                bev_segs.append(torch.tensor(data['bev_segs'][i]).to(config.gpu_device, dtype=config.dtype))
                bev_deps.append(torch.tensor(data['bev_deps'][i]).to(config.gpu_device, dtype=config.dtype))
                front_segs.append(torch.tensor(data['front_segs'][i]).to(config.gpu_device, dtype=config.dtype))
                front_deps.append(torch.tensor(data['front_deps'][i]).to(config.gpu_device, dtype=config.dtype))

            rp1 = torch.stack(data['rp1'], dim=1).to(config.gpu_device, dtype=config.dtype)
            rp2 = torch.stack(data['rp2'], dim=1).to(config.gpu_device, dtype=config.dtype)
            gt_velocity = data['velocity'].to(config.gpu_device, dtype=config.dtype)
            if gt_velocity.dim() == 1:
                gt_velocity = gt_velocity.unsqueeze(1)
            gt_waypoints = [torch.stack(data['waypoints'][j], dim=1).to(config.gpu_device, dtype=config.dtype) for j in range(0, config.pred_len)]
            gt_waypoints = torch.stack(gt_waypoints, dim=1).to(config.gpu_device, dtype=config.dtype)

            #forward pass
            torch.cuda.synchronize(config.gpu_device)
            model_start_time = time.time()
            pred_wp = model(bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, gt_velocity)
            torch.cuda.synchronize(config.gpu_device)
            model_elapsed_time = time.time() - model_start_time
            pipeline_elapsed_time = lidar_elapsed_time + (time.time() - fetch_end_time)

            #baca statistik jetson (daya, cpu, gpu, ram, vram)
            stat = jetson_stat(jetson, config)

            #hitung metrik deploy turunan (energi memakai power_avg_w)
            is_warmup = batch_ke <= warmup_batches
            model_fps = 1.0 / model_elapsed_time if model_elapsed_time > 0 else float('nan')
            pipeline_fps = 1.0 / pipeline_elapsed_time if pipeline_elapsed_time > 0 else float('nan')
            model_energy_j = stat['power_avg_w'] * model_elapsed_time
            model_fps_per_w = model_fps / stat['power_avg_w'] if stat['power_avg_w'] > 0 else float('nan')
            pipeline_energy_j = stat['power_avg_w'] * pipeline_elapsed_time
            pipeline_fps_per_w = pipeline_fps / stat['power_avg_w'] if stat['power_avg_w'] > 0 else float('nan')

            #compute metric
            metric_wp = F.l1_loss(pred_wp, gt_waypoints)
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
            scoped_dataframe(log, scope).to_csv(save_path, index=False)

            #save metadata prediksi
            save_dir_meta = save_dir / 'pred_meta' / scope
            save_dir_meta.mkdir(parents=True, exist_ok=True)
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
            with open(save_dir_meta / f"{frame_name}.yml", 'w') as dict_file:
                yaml.dump(meta_pred, dict_file)

            batch_ke += 1
            prog_bar.set_postfix(postfix) # type: ignore
            prog_bar.update(1)
        prog_bar.close()

        #batas sampel setelah warm-up
        warm = min(warmup_batches, max(len(log['batch']) - 1, 0))

        #ketika semua sudah selesai, hitung rata2 performa pada log
        log['batch'].append("avg")
        log['warmup'].append(float('nan'))
        log['test_metric'].append(np.mean(log['test_metric'][warm:]))
        log['test_wp_metric'].append(np.mean(log['test_wp_metric'][warm:]))
        log['model_elapsed_time'].append(np.mean(log['model_elapsed_time'][warm:]))
        log['model_fps'].append(np.mean(log['model_fps'][warm:]))
        log['model_energy_j'].append(np.nanmean(log['model_energy_j'][warm:]))
        log['model_fps_per_w'].append(np.nanmean(log['model_fps_per_w'][warm:]))
        log['pipeline_elapsed_time'].append(np.mean(log['pipeline_elapsed_time'][warm:]))
        log['pipeline_fps'].append(np.mean(log['pipeline_fps'][warm:]))
        log['pipeline_energy_j'].append(np.nanmean(log['pipeline_energy_j'][warm:]))
        log['pipeline_fps_per_w'].append(np.nanmean(log['pipeline_fps_per_w'][warm:]))
        log['power_w'].append(np.nanmean(log['power_w'][warm:]))
        log['power_avg_w'].append(np.nanmean(log['power_avg_w'][warm:]))
        log['cpu_util_pct'].append(np.nanmean(log['cpu_util_pct'][warm:]))
        log['gpu_util_pct'].append(np.nanmean(log['gpu_util_pct'][warm:]))
        log['ram_used_mb'].append(np.nanmean(log['ram_used_mb'][warm:]))
        log['vram_shared_mb'].append(np.nanmean(log['vram_shared_mb'][warm:]))
        log['vram_torch_mb'].append(np.nanmean(log['vram_torch_mb'][warm:]))
        log['gpu_freq_mhz'].append(np.nanmean(log['gpu_freq_mhz'][warm:]))
        log['gpu_temp_c'].append(np.nanmean(log['gpu_temp_c'][warm:]))
        log['cpu_temp_c'].append(np.nanmean(log['cpu_temp_c'][warm:]))

        #ketika semua sudah selesai, hitung VARIANCE performa pada log
        log['batch'].append("stddev")
        log['warmup'].append(float('nan'))
        log['test_metric'].append(np.std(log['test_metric'][warm:]))
        log['test_wp_metric'].append(np.std(log['test_wp_metric'][warm:]))
        log['model_elapsed_time'].append(np.std(log['model_elapsed_time'][warm:]))
        log['model_fps'].append(np.std(log['model_fps'][warm:]))
        log['model_energy_j'].append(np.nanstd(log['model_energy_j'][warm:]))
        log['model_fps_per_w'].append(np.nanstd(log['model_fps_per_w'][warm:]))
        log['pipeline_elapsed_time'].append(np.std(log['pipeline_elapsed_time'][warm:]))
        log['pipeline_fps'].append(np.std(log['pipeline_fps'][warm:]))
        log['pipeline_energy_j'].append(np.nanstd(log['pipeline_energy_j'][warm:]))
        log['pipeline_fps_per_w'].append(np.nanstd(log['pipeline_fps_per_w'][warm:]))
        log['power_w'].append(np.nanstd(log['power_w'][warm:]))
        log['power_avg_w'].append(np.nanstd(log['power_avg_w'][warm:]))
        log['cpu_util_pct'].append(np.nanstd(log['cpu_util_pct'][warm:]))
        log['gpu_util_pct'].append(np.nanstd(log['gpu_util_pct'][warm:]))
        log['ram_used_mb'].append(np.nanstd(log['ram_used_mb'][warm:]))
        log['vram_shared_mb'].append(np.nanstd(log['vram_shared_mb'][warm:]))
        log['vram_torch_mb'].append(np.nanstd(log['vram_torch_mb'][warm:]))
        log['gpu_freq_mhz'].append(np.nanstd(log['gpu_freq_mhz'][warm:]))
        log['gpu_temp_c'].append(np.nanstd(log['gpu_temp_c'][warm:]))
        log['cpu_temp_c'].append(np.nanstd(log['cpu_temp_c'][warm:]))

        #hitung persentil tail latency dan efisiensi
        log['batch'].append("p50")
        log['warmup'].append(float('nan'))
        log['test_metric'].append(float('nan'))
        log['test_wp_metric'].append(float('nan'))
        log['model_elapsed_time'].append(np.nanpercentile(log['model_elapsed_time'][warm:], 50))
        log['model_fps'].append(1.0 / np.nanpercentile(log['model_elapsed_time'][warm:], 50))
        log['pipeline_elapsed_time'].append(np.nanpercentile(log['pipeline_elapsed_time'][warm:], 50))
        log['pipeline_fps'].append(1.0 / np.nanpercentile(log['pipeline_elapsed_time'][warm:], 50))
        log['power_w'].append(float('nan'))
        log['power_avg_w'].append(float('nan'))
        log['model_energy_j'].append(np.nanpercentile(log['model_energy_j'][warm:], 50))
        log['model_fps_per_w'].append(np.nanpercentile(log['model_fps_per_w'][warm:], 50))
        log['pipeline_energy_j'].append(np.nanpercentile(log['pipeline_energy_j'][warm:], 50))
        log['pipeline_fps_per_w'].append(np.nanpercentile(log['pipeline_fps_per_w'][warm:], 50))
        log['cpu_util_pct'].append(float('nan'))
        log['gpu_util_pct'].append(float('nan'))
        log['ram_used_mb'].append(float('nan'))
        log['vram_shared_mb'].append(float('nan'))
        log['vram_torch_mb'].append(float('nan'))
        log['gpu_freq_mhz'].append(float('nan'))
        log['gpu_temp_c'].append(float('nan'))
        log['cpu_temp_c'].append(float('nan'))

        log['batch'].append("p95")
        log['warmup'].append(float('nan'))
        log['test_metric'].append(float('nan'))
        log['test_wp_metric'].append(float('nan'))
        log['model_elapsed_time'].append(np.nanpercentile(log['model_elapsed_time'][warm:], 95))
        log['model_fps'].append(1.0 / np.nanpercentile(log['model_elapsed_time'][warm:], 95))
        log['pipeline_elapsed_time'].append(np.nanpercentile(log['pipeline_elapsed_time'][warm:], 95))
        log['pipeline_fps'].append(1.0 / np.nanpercentile(log['pipeline_elapsed_time'][warm:], 95))
        log['power_w'].append(float('nan'))
        log['power_avg_w'].append(float('nan'))
        log['model_energy_j'].append(np.nanpercentile(log['model_energy_j'][warm:], 95))
        log['model_fps_per_w'].append(np.nanpercentile(log['model_fps_per_w'][warm:], 95))
        log['pipeline_energy_j'].append(np.nanpercentile(log['pipeline_energy_j'][warm:], 95))
        log['pipeline_fps_per_w'].append(np.nanpercentile(log['pipeline_fps_per_w'][warm:], 95))
        log['cpu_util_pct'].append(float('nan'))
        log['gpu_util_pct'].append(float('nan'))
        log['ram_used_mb'].append(float('nan'))
        log['vram_shared_mb'].append(float('nan'))
        log['vram_torch_mb'].append(float('nan'))
        log['gpu_freq_mhz'].append(float('nan'))
        log['gpu_temp_c'].append(float('nan'))
        log['cpu_temp_c'].append(float('nan'))

        log['batch'].append("p99")
        log['warmup'].append(float('nan'))
        log['test_metric'].append(float('nan'))
        log['test_wp_metric'].append(float('nan'))
        log['model_elapsed_time'].append(np.nanpercentile(log['model_elapsed_time'][warm:], 99))
        log['model_fps'].append(1.0 / np.nanpercentile(log['model_elapsed_time'][warm:], 99))
        log['pipeline_elapsed_time'].append(np.nanpercentile(log['pipeline_elapsed_time'][warm:], 99))
        log['pipeline_fps'].append(1.0 / np.nanpercentile(log['pipeline_elapsed_time'][warm:], 99))
        log['power_w'].append(float('nan'))
        log['power_avg_w'].append(float('nan'))
        log['model_energy_j'].append(np.nanpercentile(log['model_energy_j'][warm:], 99))
        log['model_fps_per_w'].append(np.nanpercentile(log['model_fps_per_w'][warm:], 99))
        log['pipeline_energy_j'].append(np.nanpercentile(log['pipeline_energy_j'][warm:], 99))
        log['pipeline_fps_per_w'].append(np.nanpercentile(log['pipeline_fps_per_w'][warm:], 99))
        log['cpu_util_pct'].append(float('nan'))
        log['gpu_util_pct'].append(float('nan'))
        log['ram_used_mb'].append(float('nan'))
        log['vram_shared_mb'].append(float('nan'))
        log['vram_torch_mb'].append(float('nan'))
        log['gpu_freq_mhz'].append(float('nan'))
        log['gpu_temp_c'].append(float('nan'))
        log['cpu_temp_c'].append(float('nan'))

        log['batch'].append("max")
        log['warmup'].append(float('nan'))
        log['test_metric'].append(float('nan'))
        log['test_wp_metric'].append(float('nan'))
        log['model_elapsed_time'].append(float('nan'))
        log['model_fps'].append(float('nan'))
        log['model_energy_j'].append(float('nan'))
        log['model_fps_per_w'].append(float('nan'))
        log['pipeline_elapsed_time'].append(float('nan'))
        log['pipeline_fps'].append(float('nan'))
        log['pipeline_energy_j'].append(float('nan'))
        log['pipeline_fps_per_w'].append(float('nan'))
        log['power_w'].append(float('nan'))
        log['power_avg_w'].append(float('nan'))
        log['cpu_util_pct'].append(float('nan'))
        log['gpu_util_pct'].append(float('nan'))
        log['ram_used_mb'].append(np.nanmax(log['ram_used_mb'][warm:]))
        log['vram_shared_mb'].append(np.nanmax(log['vram_shared_mb'][warm:]))
        log['vram_torch_mb'].append(np.nanmax(log['vram_torch_mb'][warm:]))
        log['gpu_freq_mhz'].append(float('nan'))
        log['gpu_temp_c'].append(float('nan'))
        log['cpu_temp_c'].append(float('nan'))

        #paste ke csv file
        scoped_dataframe(log, scope).to_csv(save_path, index=False)

    return log

def main():
    parser = argparse.ArgumentParser(description="Model benchmarking in Jetson board")
    parser.add_argument("--route", action='append', type=str, help='set routes, you can combine multiple routes. e.g: --route "a" --route "b" ...')
    parser.add_argument("--warmup", type=int, default=3, help='number of warm-up batches excluded from summary stats (default: 3)')
    parser.add_argument("--mode", choices=["model", "pipeline", "both"], default="both", help='benchmark scope: model only, pipeline (PolarSeg+projection+model), or both (default: both)')
    args = parser.parse_args()
    # Load default config
    config: GlobalConfig = GlobalConfig()

    # Load config from the selected log run
    logdir = select_logdir()
    config_path = logdir / "config.py"
    spec = importlib.util.spec_from_file_location("config", config_path)
    assert spec is not None and spec.loader is not None, f"Failed to load spec/loader from {config_path}"
    log_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(log_config)
    config = cast(GlobalConfig, log_config.GlobalConfig())
    config.logdir = logdir

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

    #SET GPU YANG AKTIF
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id

    #IMPORT MODEL dan load bobot
    model_name = config.logdir.name.split('_', 1)[0] # ambil nama model dari nama folder log
    print("IMPORT ARSITEKTUR DL DAN COMPILE")
    print("current model of Logdir: ", model_name)
    if model_name == "ai23":
        model = ai23(config, device=config.gpu_device).to(config.gpu_device, dtype=config.dtype)
    elif model_name == "ai33":
            model = ai33(config, device=config.gpu_device).to(config.gpu_device, dtype=config.dtype)
    elif model_name == "xr20":
            model = xr20(config, device=config.gpu_device).to(config.gpu_device, dtype=config.dtype)
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
