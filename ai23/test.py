import pandas as pd
import os
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
import cv2
from torch import torch
import yaml
from pathlib import Path

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from ai23.model import xr20
from ai23.dataloader import KarrDataset

# use the config from the log directory
from ai23.config import GlobalConfig, select_logdir
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

def test(data_loader, model, config: GlobalConfig):
    score = {
        'total_metric': AverageMeter(),
        'wp_metric': AverageMeter()
    }

    log = OrderedDict([
        ('batch', []),
        ('test_metric', []),
        ('test_wp_metric', []),
        ('model_elapsed_time', []),
    ])
    batch_ke = 1

    save_dir = config.logdir + "/offline_test/"
    os.makedirs(save_dir, exist_ok=True)
    save_dir_log = save_dir
    os.makedirs(save_dir_log, exist_ok=True)

    model.eval()

    with torch.no_grad():
        prog_bar = tqdm(total=len(data_loader))

        for data in data_loader:
            start_time = time.time()
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
            model_start_time = time.time()
            pred_wp = model(bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, gt_velocity)
            model_elapsed_time = time.time() - model_start_time

            #compute metric
            metric_wp = F.l1_loss(pred_wp, gt_waypoints)
            total_metric = metric_wp.item()

            score['total_metric'].update(total_metric)
            score['wp_metric'].update(metric_wp.item())

            #update visualisasi progress bar
            postfix = OrderedDict([
                ('te_total_m', score['total_metric'].avg),
                ('te_wp_m', score['wp_metric'].avg)
            ])

            #simpan history test ke file csv, ambil dari hasil kalkulasi metric langsung, jangan dari averagemeter
            log['batch'].append(batch_ke)
            log['test_metric'].append(total_metric)
            log['test_wp_metric'].append(metric_wp.item())
            log['model_elapsed_time'].append(model_elapsed_time)
            pd.DataFrame(log).to_csv(save_dir_log+'/test_log.csv', index=False)

            #save metadata prediksi
            save_dir_meta = save_dir+'/pred_meta/'
            os.makedirs(save_dir_meta, exist_ok=True)
            #isikan beberapa data
            meta_pred = {}
            meta_pred['rp1_pos_local'] = rp1[0].cpu().detach().numpy().tolist()
            meta_pred['rp2_pos_local'] = rp2[0].cpu().detach().numpy().tolist()
            meta_pred['rp1_pos_global'] = np.array([data['rp1_lat'].item(), data['rp1_lon'].item()]).tolist()
            meta_pred['rp2_pos_global'] = np.array([data['rp2_lat'].item(), data['rp2_lon'].item()]).tolist()
            meta_pred['robot_bearing'] = float(data['bearing_robot'].item())
            meta_pred['robot_pos_global'] = np.array([data['lat_robot'].item(), data['lon_robot'].item()]).tolist()
            meta_pred['model_fps'] = float(1/model_elapsed_time)
            elapsed_time = time.time() - start_time #hitung elapsedtime
            meta_pred['fps'] = float(1/elapsed_time)
            with open(save_dir_meta+data['filename'][-1]+".yml", 'w') as dict_file:
                yaml.dump(meta_pred, dict_file)

            batch_ke += 1
            prog_bar.set_postfix(postfix)
            prog_bar.update(1)
        prog_bar.close()

        #ketika semua sudah selesai, hitung rata2 performa pada log
        log['batch'].append("avg")
        log['test_metric'].append(np.mean(log['test_metric']))
        log['test_wp_metric'].append(np.mean(log['test_wp_metric']))
        log['model_elapsed_time'].append(np.mean(log['model_elapsed_time']))

        #ketika semua sudah selesai, hitung VARIANCE performa pada log
        log['batch'].append("stddev")
        log['test_metric'].append(np.std(log['test_metric'][:-1]))
        log['test_wp_metric'].append(np.std(log['test_wp_metric'][:-1]))
        log['model_elapsed_time'].append(np.std(log['model_elapsed_time'][:-1]))

        #paste ke csv file
        pd.DataFrame(log).to_csv(save_dir_log+'/test_log.csv', index=False)

    return log


def main():
    # Load default config
    config: GlobalConfig = GlobalConfig()

    # Load config from the selected log run
    logdir = select_logdir()
    config_path = os.path.join(logdir, "config.py")
    spec = importlib.util.spec_from_file_location("config", config_path)
    log_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(log_config)
    config = cast(GlobalConfig, log_config.GlobalConfig())
    config.logdir = logdir


    #SET GPU YANG AKTIF
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id

    #IMPORT MODEL dan load bobot
    print("IMPORT ARSITEKTUR DL DAN COMPILE")
    model = xr20(config, device=config.gpu_device).to(config.gpu_device, dtype=config.dtype)
    model.load_state_dict(torch.load(os.path.join(config.logdir, 'best_model.pth')))

    karr_dataset = KarrDataset(config=config)
    total_len = len(karr_dataset)
    # train: 80%, validation: 10%, test: 10%
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_indices = list(range(0, train_len))
    val_indices = list(range(train_len, train_len + val_len))
    test_indices = list(range(train_len + val_len, total_len))
    test_set = Subset(karr_dataset, test_indices)
    dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    #test
    test_log = test(dataloader_test, model, config)

    #kosongkan cuda chace
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()