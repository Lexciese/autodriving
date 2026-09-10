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
from PIL import Image, ImageDraw

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True

from ai23.model import xr20
from ai23.dataloader import KarrDataset

from preprocessing.data_util import plot_lidbev_rpwp, plot_lidfront_rpwp

# use the config from the log directory
from ai23.config import GlobalConfig
import importlib.util

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

def test(data_loader, model, config):
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

    out_video = None

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

            pred_wp_np = pred_wp[0].cpu().detach().numpy()
            wp_lidbev_frame = []
            wp_lidfront_frame = []
            for j in range(pred_wp_np.shape[0]):
                x_frame, y_frame = plot_lidbev_rpwp(config, pred_wp_np[j][0], pred_wp_np[j][1])
                wp_lidbev_frame.append(np.array([x_frame, y_frame]))
                x_frame, y_frame = plot_lidfront_rpwp(config, pred_wp_np[j][0], pred_wp_np[j][1])
                wp_lidfront_frame.append(np.array([x_frame, y_frame]))

            rp1_np = rp1[0].cpu().detach().numpy()
            rp2_np = rp2[0].cpu().detach().numpy()
            rp_lidbev_frame = []
            rp_lidfront_frame = []
            for rp_xy in [rp1_np, rp2_np]:
                x_frame, y_frame = plot_lidbev_rpwp(config, rp_xy[0], rp_xy[1])
                rp_lidbev_frame.append(np.array([x_frame, y_frame]))
                x_frame, y_frame = plot_lidfront_rpwp(config, rp_xy[0], rp_xy[1])
                rp_lidfront_frame.append(np.array([x_frame, y_frame]))

            filename_base = data['filename'][-1]
            if filename_base.endswith('.yml'):
                filename_base = filename_base[:-4]
            if '/' in filename_base:
                fn_parts = filename_base.split('/')
                filenum = fn_parts[-1]
                route_path = '/'.join(fn_parts[:-1])
                base_dir = config.datadir + route_path + "/"
            else:
                filenum = filename_base
                base_dir = config.datadir

            base_dir = "/media/mf/SATA4TB/autodriving/datasetx/2026-09-07_route00"
            ddir_lidseg_bev = base_dir + "lidar/img/bev_seg/"
            ddir_lidseg_fro = base_dir + "lidar/img/front_seg/"
            ddir_liddep_bev = base_dir + "lidar/img/bev_dep/"
            ddir_liddep_fro = base_dir + "lidar/img/front_dep/"
            ddir_rgb_front = base_dir + "camera/rgb/"

            lidar_bev_segcol = cv2.imread(ddir_lidseg_bev + filenum + ".png")
            lidar_bev_depcol = cv2.imread(ddir_liddep_bev + filenum + ".png")
            lidar_front_segcol = cv2.imread(ddir_lidseg_fro + filenum + ".png")
            lidar_front_depcol = cv2.imread(ddir_liddep_fro + filenum + ".png")
            rgb_front = cv2.imread(ddir_rgb_front + filenum + ".png")

            if rgb_front is not None and lidar_bev_segcol is not None and lidar_bev_depcol is not None and lidar_front_segcol is not None and lidar_front_depcol is not None:
                lidar_bev_segcol_wprp = lidar_bev_segcol.copy()
                lidar_front_segcol_wprp = lidar_front_segcol.copy()

                for k in range(len(rp_lidbev_frame)):
                    lidar_bev_segcol_wprp = cv2.circle(lidar_bev_segcol_wprp, (rp_lidbev_frame[k][0], rp_lidbev_frame[k][1]), radius=3, color=(255, 255, 255), thickness=2)
                    lidar_front_segcol_wprp = cv2.circle(lidar_front_segcol_wprp, (rp_lidfront_frame[k][0], rp_lidfront_frame[k][1]), radius=3, color=(255, 255, 255), thickness=2)

                for k in range(len(wp_lidbev_frame)):
                    lidar_bev_segcol_wprp = cv2.circle(lidar_bev_segcol_wprp, (wp_lidbev_frame[k][0], wp_lidbev_frame[k][1]), radius=2, color=(255, 255, 255), thickness=-1)
                    lidar_front_segcol_wprp = cv2.circle(lidar_front_segcol_wprp, (wp_lidfront_frame[k][0], wp_lidfront_frame[k][1]), radius=2, color=(255, 255, 255), thickness=-1)

                left_column_w = 1024
                rgb_h = int(rgb_front.shape[0] * (left_column_w / rgb_front.shape[1]))
                rgb_front_scaled = cv2.resize(rgb_front, (left_column_w, rgb_h), interpolation=cv2.INTER_LINEAR)

                front_lidar_h = int(lidar_front_depcol.shape[0] * (left_column_w / lidar_front_depcol.shape[1]))
                front_dep_scaled = cv2.resize(lidar_front_depcol, (left_column_w, front_lidar_h), interpolation=cv2.INTER_LINEAR)
                front_seg_scaled = cv2.resize(lidar_front_segcol_wprp, (left_column_w, front_lidar_h), interpolation=cv2.INTER_LINEAR)

                left_column = np.concatenate((rgb_front_scaled, front_dep_scaled, front_seg_scaled), axis=0)
                total_h = left_column.shape[0]

                vel_val = float(data['velocity'].flatten()[0].item())
                bearing_val = float(data['bearing_robot'].item())
                lat_robot_val = float(data['lat_robot'].item())
                lon_robot_val = float(data['lon_robot'].item())
                rp1_lat_val = float(data['rp1_lat'].item())
                rp1_lon_val = float(data['rp1_lon'].item())
                rp2_lat_val = float(data['rp2_lat'].item())
                rp2_lon_val = float(data['rp2_lon'].item())

                telemetry_lines = [
                    ("INPUT", ""),
                    (f"File Name: {filenum}.yml", ""),
                    (f"Speed: {format(np.round(vel_val, 3), '.3f')} km/h", ""),
                    (f"Bearing: {format(np.round(bearing_val, 3), '.3f')}", ""),
                    (f"Robot Lat: {format(np.round(lat_robot_val, 6), '.6f')}", ""),
                    (f"Robot Lon: {format(np.round(lon_robot_val, 6), '.6f')}", ""),
                    (f"Rp1 Lat: {format(np.round(rp1_lat_val, 6), '.6f')}", ""),
                    (f"Rp1 Lon: {format(np.round(rp1_lon_val, 6), '.6f')}", ""),
                    (f"Rp2 Lat: {format(np.round(rp2_lat_val, 6), '.6f')}", ""),
                    (f"Rp2 Lon: {format(np.round(rp2_lon_val, 6), '.6f')}", ""),
                    ("", ""),
                    ("OUTPUT", ""),
                ]

                for wp_idx in range(min(3, pred_wp_np.shape[0])):
                    txt_wp = f"Wp{wp_idx+1} Loc: x: {format(np.round(pred_wp_np[wp_idx][0], 3), '.3f')} | y: {format(np.round(pred_wp_np[wp_idx][1], 3), '.3f')}"
                    telemetry_lines.append((txt_wp, ""))

                telemetry_lines.extend([
                    ("", ""),
                    ("METRIC", ""),
                    (f"L1 Loss: {format(np.round(total_metric, 4), '.4f')}", ""),
                    (f"Model FPS: {format(np.round(1/model_elapsed_time, 3), '.3f')}", "")
                ])

                line_gap = min(22, int(rgb_h / (len(telemetry_lines) + 2)))
                overlay_w = 420
                overlay_h = (len(telemetry_lines) + 1) * line_gap

                overlay_pil = Image.new('RGBA', (overlay_w, overlay_h), (0, 0, 0, 128))
                draw = ImageDraw.Draw(overlay_pil)

                x_offset = 12
                for idx, (line_text, _) in enumerate(telemetry_lines):
                    if line_text:
                        draw.text((x_offset, 8 + idx * line_gap), line_text, font=config.fontx, fill=(255, 255, 255, 255))

                left_column_pil = Image.fromarray(cv2.cvtColor(left_column, cv2.COLOR_BGR2RGB)).convert('RGBA')
                left_column_pil.paste(overlay_pil, (10, 10), overlay_pil)
                left_column = cv2.cvtColor(np.array(left_column_pil.convert('RGB')), cv2.COLOR_RGB2BGR)

                bev_stack_raw = np.concatenate((lidar_bev_depcol, lidar_bev_segcol_wprp), axis=0)
                bev_target_w = int(bev_stack_raw.shape[1] * (total_h / bev_stack_raw.shape[0]))
                bev_column = cv2.resize(bev_stack_raw, (bev_target_w, total_h), interpolation=cv2.INTER_LINEAR)

                final_img = np.concatenate((left_column, bev_column), axis=1)

                if out_video is None:
                    out_video = cv2.VideoWriter(
                        save_dir + '/test_video.avi',
                        cv2.VideoWriter_fourcc(*'DIVX'),
                        config.fps,
                        (final_img.shape[1], final_img.shape[0])
                    )

                out_video.write(np.uint8(final_img))

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

    if out_video is not None:
        out_video.release()

    return log


def main():
    # Load default config
    config = GlobalConfig()

    # Load config from the saved log
    spec = importlib.util.spec_from_file_location("config", str(config.logdir))
    log_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(log_config)
    log_config = log_config.GlobalConfig()
    config = log_config


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
