import pandas as pd
import os
import cv2
from tqdm import tqdm
from collections import OrderedDict
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True

import shutil
from ai23.model import xr20
from ai23.dataloader import KarrDataset
from ai23.config import GlobalConfig
import common.config


# Class untuk penyimpanan dan perhitungan update loss
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # update kalkulasi
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# fungsi renormalize loss weights seperti di paper gradnorm
def renormalize_params_lw(current_lw, config: GlobalConfig):
    # detach dulu paramsnya dari torch, pindah ke CPU
    lw = np.array([tens.cpu().detach().numpy() for tens in current_lw])
    lws = np.array([lw[i][0] for i in range(len(lw))])
    # fungsi renormalize untuk algoritma 1 di paper gradnorm
    coef = np.array(config.loss_weights).sum() / lws.sum()
    new_lws = [coef * lwx for lwx in lws]
    # buat torch float tensor lagi dan masukkan ke cuda memory
    normalized_lws = [torch.cuda.FloatTensor([lw]).clone().detach().requires_grad_(True) for lw in new_lws]
    return normalized_lws


# FUNGSI TRAINING
def train(data_loader, model, config: GlobalConfig, writer, cur_epoch, optimizer, params_lw, optimizer_lw):
    # buat variabel untuk menyimpan kalkulasi loss
    score = {
        'total_loss': AverageMeter(),
        'wp_loss': AverageMeter()
    }

    # masuk ke mode training, pytorch
    model.train()

    # visualisasi progress training dengan tqdm
    prog_bar = tqdm(total=len(data_loader))

    total_batch = len(data_loader)
    batch_ke = 0
    for data in data_loader:
        cur_step = cur_epoch * total_batch + batch_ke

        # pindah ke torch gpu device
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

        # Velocity as a scalar per sample in the batch
        gt_velocity = data['velocity'].to(config.gpu_device, dtype=config.dtype)
        if gt_velocity.dim() == 1:
            gt_velocity = gt_velocity.unsqueeze(1)

        gt_waypoints = [torch.stack(data['waypoints'][j], dim=1).to(config.gpu_device, dtype=config.dtype) for j in range(0, config.pred_len)]
        gt_waypoints = torch.stack(gt_waypoints, dim=1).to(config.gpu_device, dtype=config.dtype)

        # forward pass
        pred_wp = model(bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, gt_velocity)

        # compute loss
        print("wp: ", gt_waypoints)
        loss_wp = F.l1_loss(pred_wp, gt_waypoints)
        total_loss = params_lw[0] * loss_wp

        # backprop, kalkulasi gradient, dan optimasi
        optimizer.zero_grad()

        if batch_ke == 0:  # batch pertama, hitung loss awal
            total_loss.backward()
            loss_wp_0 = torch.clone(loss_wp)

        elif 0 < batch_ke < total_batch - 1:
            total_loss.backward()

        elif batch_ke == total_batch - 1:  # batch terakhir
            if config.MGN:
                optimizer_lw.zero_grad()
                total_loss.backward(retain_graph=True)
                params = list(filter(lambda p: p.requires_grad, model.parameters()))
                G0R = torch.autograd.grad(loss_wp, params[config.bottleneck], retain_graph=True, create_graph=True)
                G0 = torch.norm(G0R[0], keepdim=True)

                G_avg = G0

                loss_wp_hat = loss_wp / loss_wp_0
                loss_hat_avg = loss_wp_hat

                inv_rate_wp = loss_wp_hat / loss_hat_avg

                C0 = (G_avg * inv_rate_wp).detach() ** config.lw_alpha

                Lgrad = F.l1_loss(G0, C0)
                Lgrad.backward()
                optimizer_lw.step()

                lgrad = Lgrad.item()
                new_param_lw = optimizer_lw.param_groups[0]['params']
            else:
                total_loss.backward()
                lgrad = 0
                new_param_lw = params_lw

        optimizer.step()

        # hitung rata-rata loss
        score['total_loss'].update(total_loss.item())
        score['wp_loss'].update(loss_wp.item())

        # update visualisasi progress bar
        postfix = OrderedDict([
            ('t_total_l', score['total_loss'].avg),
            ('t_wp_l', score['wp_loss'].avg)
        ])

        # tambahkan ke summary writer
        writer.add_scalar('t_total_l', total_loss.item(), cur_step)
        writer.add_scalar('t_wp_l', loss_wp.item(), cur_step)

        prog_bar.set_postfix(postfix)
        prog_bar.update(1)
        batch_ke += 1
    prog_bar.close()

    return postfix, new_param_lw, lgrad


# FUNGSI VALIDATION
def validate(data_loader, model, config: GlobalConfig, writer, cur_epoch):
    score = {
        'total_loss': AverageMeter(),
        'wp_loss': AverageMeter()
    }

    model.eval()

    with torch.no_grad():
        prog_bar = tqdm(total=len(data_loader))

        total_batch = len(data_loader)
        batch_ke = 0
        for data in data_loader:
            cur_step = cur_epoch * total_batch + batch_ke

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

            # forward pass
            pred_wp = model(bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, gt_velocity)

            # compute loss
            loss_wp = F.l1_loss(pred_wp, gt_waypoints)
            total_loss = loss_wp

            score['total_loss'].update(total_loss.item())
            score['wp_loss'].update(loss_wp.item())

            postfix = OrderedDict([
                ('v_total_l', score['total_loss'].avg),
                ('v_wp_l', score['wp_loss'].avg)
            ])

            writer.add_scalar('v_total_l', total_loss.item(), cur_step)
            writer.add_scalar('v_wp_l', loss_wp.item(), cur_step)

            prog_bar.set_postfix(postfix)
            prog_bar.update(1)
            batch_ke += 1
        prog_bar.close()

    return postfix


# MAIN FUNCTION
def main():
    config: GlobalConfig = GlobalConfig()

    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    print("IMPORT ARSITEKTUR DL DAN COMPILE")
    model = xr20(config, device=config.gpu_device).to(config.gpu_device, dtype=config.dtype)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total trainable parameters: ', params)

    optima = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optima, mode='min', factor=0.5, patience=4, min_lr=1e-6)

    karr_dataset = KarrDataset(config=config)

    # Calculate dataset lengths
    total_len = len(karr_dataset)
    # train: 80%, validation: 10%, test: 10%
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_indices = list(range(0, train_len))
    val_indices = list(range(train_len, train_len + val_len))

    train_set = Subset(karr_dataset, train_indices)
    val_set = Subset(karr_dataset, val_indices)


    drop_last = True if len(train_set) % config.batch_size == 1 else False

    dataloader_train = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)
    dataloader_val = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset split total: {total_len} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {test_len}")

    if not os.path.exists(config.logdir + "/trainval_log.csv"):
        print('TRAIN from the beginning!!!!!!!!!!!!!!!!')
        os.makedirs(config.logdir, exist_ok=True)
        print('Created dir:', config.logdir)

        # Initial loss weights (single task for waypoints)
        params_lw = [torch.cuda.FloatTensor([config.loss_weights[0]]).clone().detach().requires_grad_(True)]
        optima_lw = optim.SGD(params_lw, lr=config.lr)
        curr_ep = 0
        lowest_score = float('inf')
        stop_count = config.init_stop_counter
    else:
        print('Continue training!!!!!!!!!!!!!!!!')
        print('Loading checkpoint from ' + config.logdir)
        log_trainval = pd.read_csv(config.logdir + "/trainval_log.csv")
        curr_ep = int(log_trainval['epoch'][-1:]) + 1
        lowest_score = float(np.min(log_trainval['val_loss']))
        stop_count = int(log_trainval['stop_counter'][-1:])

        model.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_model.pth')))
        optima.load_state_dict(torch.load(os.path.join(config.logdir, 'recent_optim.pth')))

        latest_lw = [float(log_trainval['lw_wp'][-1:])]
        params_lw = [torch.cuda.FloatTensor([latest_lw[0]]).clone().detach().requires_grad_(True)]
        optima_lw = optim.SGD(params_lw, lr=float(log_trainval['lrate'][-1:]))

        config.logdir += "/retrain"
        os.makedirs(config.logdir, exist_ok=True)
        print('Created new retrain dir:', config.logdir)

    config_file_path = common.config.__file__
    shutil.copyfile(config_file_path, os.path.join(config.logdir, 'config.py'))

    log = OrderedDict([
        ('epoch', []),
        ('best_model', []),
        ('val_loss', []),
        ('val_wp_loss', []),
        ('train_loss', []),
        ('train_wp_loss', []),
        ('lrate', []),
        ('stop_counter', []),
        ('lgrad_loss', []),
        ('lw_wp', []),
        ('elapsed_time', []),
    ])
    writer = SummaryWriter(log_dir=config.logdir)

    epoch = curr_ep
    while True:
        print("Epoch: {:05d}------------------------------------------------".format(epoch))
        if config.MGN:
            curr_lw = optima_lw.param_groups[0]['params']
            lw = np.array([tens.cpu().detach().numpy() for tens in curr_lw])
            lws = np.array([lw[i][0] for i in range(len(lw))])
            print("current loss weights: ", lws)
        else:
            curr_lw = [config.loss_weights[0]]
            lws = [config.loss_weights[0]]
            print("current loss weights: ", lws)
        print("current lr untuk training: ", optima.param_groups[0]['lr'])

        start_time = time.time()
        train_log, new_params_lw, lgrad = train(dataloader_train, model, config, writer, epoch, optima, curr_lw, optima_lw)
        val_log = validate(dataloader_val, model, config, writer, epoch)

        if config.MGN:
            optima_lw.param_groups[0]['params'] = renormalize_params_lw(new_params_lw, config)
            print("total loss gradient: " + str(lgrad))

        scheduler.step(val_log['v_total_l'])
        optima_lw.param_groups[0]['lr'] = optima.param_groups[0]['lr']
        elapsed_time = time.time() - start_time

        log['epoch'].append(epoch)
        log['lrate'].append(optima.param_groups[0]['lr'])
        log['train_loss'].append(train_log['t_total_l'])
        log['val_loss'].append(val_log['v_total_l'])
        log['train_wp_loss'].append(train_log['t_wp_l'])
        log['val_wp_loss'].append(val_log['v_wp_l'])
        log['lgrad_loss'].append(lgrad)
        log['lw_wp'].append(lws[0])
        log['elapsed_time'].append(elapsed_time)

        print('| t_total_l: %.4f | t_wp_l: %.4f |' % (train_log['t_total_l'], train_log['t_wp_l']))
        print('| v_total_l: %.4f | v_wp_l: %.4f |' % (val_log['v_total_l'], val_log['v_wp_l']))
        print('elapsed time: %.4f sec' % (elapsed_time))

        torch.save(model.state_dict(), os.path.join(config.logdir, 'recent_model.pth'))
        torch.save(optima.state_dict(), os.path.join(config.logdir, 'recent_optim.pth'))

        if val_log['v_total_l'] < lowest_score:
            print("v_total_l: %.4f < lowest sebelumnya: %.4f" % (val_log['v_total_l'], lowest_score))
            print("model terbaik disave!")
            torch.save(model.state_dict(), os.path.join(config.logdir, 'best_model.pth'))
            torch.save(optima.state_dict(), os.path.join(config.logdir, 'best_optim.pth'))
            lowest_score = val_log['v_total_l']
            stop_count = config.init_stop_counter
            print("stop counter direset ke: ", stop_count)
            log['best_model'].append("BEST")
        else:
            print("v_total_l: %.4f >= lowest sebelumnya: %.4f" % (val_log['v_total_l'], lowest_score))
            print("model tidak disave!")
            stop_count -= 1
            print("stop counter : ", stop_count)
            log['best_model'].append("")

        log['stop_counter'].append(stop_count)
        pd.DataFrame(log).to_csv(os.path.join(config.logdir, 'trainval_log.csv'), index=False)

        torch.cuda.empty_cache()
        epoch += 1

        if stop_count == 0:
            print("TRAINING BERHENTI KARENA TIDAK ADA PENURUNAN TOTAL LOSS DALAM %d EPOCH TERAKHIR" % (config.init_stop_counter))
            break


if __name__ == "__main__":
    main()
