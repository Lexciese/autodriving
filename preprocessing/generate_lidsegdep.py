import numpy as np
import yaml
import cv2
import torch
import os
torch.backends.cudnn.benchmark = True

# polarseg
from pypcd import pypcd
from polarseg.network.BEV_Unet import BEV_Unet
from polarseg.network.ptBEV import ptBEVnet
from data_util import preproc_spherical, gen_bev_front_rear_seg_dep, colorize_seg, colorize_logdepth
from config import GlobalConfig

configx = GlobalConfig()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = configx.gpu_id

# init model
BEV_model = BEV_Unet(n_class=configx.n_class_kitti - 1,
                     n_height=configx.grid_size[2],
                     input_batch_norm=True,
                     dropout=0.5,
                     circular_padding=True)
polarseg = ptBEVnet(BEV_model,
                    pt_model='pointnet',
                    grid_size=configx.grid_size,
                    fea_dim=9,
                    max_pt_per_encode=256,
                    out_pt_fea_dim=512,
                    kernal_size=1,
                    pt_selection='random',
                    fea_compre=configx.grid_size[2])
polarseg.load_state_dict(torch.load(configx.polarseg_weight_path))
polarseg.to(configx.gpu_device)
polarseg.eval()

# loop over all routes
route_list = os.listdir(configx.datadir)
route_list.sort()
for route in route_list:
    if os.path.isfile(configx.datadir + route):
        continue
    print(route)
    ddir_meta = configx.datadir + route + "/meta/"
    ddir_lidar = configx.datadir + route + "/lidar/cld/"
    
    ddir_lidseg = configx.datadir + route + "/lidar/seg/"
    ddir_lidseg_bev = configx.datadir + route + "/lidar/img/bev_seg/"
    ddir_lidseg_fro = configx.datadir + route + "/lidar/img/front_seg/"
    ddir_lidseg_rea = configx.datadir + route + "/lidar/img/rear_seg/"
    ddir_liddep_bev = configx.datadir + route + "/lidar/img/bev_dep/"
    ddir_liddep_fro = configx.datadir + route + "/lidar/img/front_dep/"
    ddir_liddep_rea = configx.datadir + route + "/lidar/img/rear_dep/"
    os.makedirs(ddir_lidseg, exist_ok=True)
    os.makedirs(ddir_lidseg_bev, exist_ok=True)
    os.makedirs(ddir_lidseg_fro, exist_ok=True)
    os.makedirs(ddir_lidseg_rea, exist_ok=True)
    os.makedirs(ddir_liddep_bev, exist_ok=True)
    os.makedirs(ddir_liddep_fro, exist_ok=True)
    os.makedirs(ddir_liddep_rea, exist_ok=True)

    file_list = os.listdir(ddir_meta)
    file_list.sort()

    with torch.no_grad():
        for filex in file_list:
            filenum = filex[:-4]
            print(filenum)

            lid_pc = pypcd.PointCloud.from_path(ddir_lidar + filenum + ".pcd")
            lid_x = lid_pc.pc_data['x']
            lid_y = lid_pc.pc_data['y']
            lid_z = lid_pc.pc_data['z']
            lid_intensity = lid_pc.pc_data['intensity']

            in_velodyne = np.zeros((lid_x.shape[0], 4), dtype=np.float32)
            in_velodyne[:, 0] = lid_x
            in_velodyne[:, 1] = lid_y
            in_velodyne[:, 2] = lid_z
            in_velodyne[:, 3] = lid_intensity

            valid = np.isfinite(in_velodyne[:, 0]) & np.isfinite(in_velodyne[:, 1]) & \
                    np.isfinite(in_velodyne[:, 2]) & np.isfinite(in_velodyne[:, 3])
            in_velodyne = in_velodyne[valid]

            test_grid, test_pt_fea = preproc_spherical(in_velodyne, configx)

            grid = test_grid[0]                     # (N, 3)  [dim0, dim1, dim2]
            valid_idx = (grid[:, 0] >= 0) & (grid[:, 0] < configx.grid_size[0]) & \
                        (grid[:, 1] >= 0) & (grid[:, 1] < configx.grid_size[1]) & \
                        (grid[:, 2] >= 0) & (grid[:, 2] < configx.grid_size[2])
            if not np.all(valid_idx):
                print(f"  Warning: {np.sum(~valid_idx)} points outside grid, dropping them.")
                grid = grid[valid_idx]
                test_pt_fea = [test_pt_fea[0][valid_idx]]
                test_grid[0] = grid

            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(configx.gpu_device) for i in test_pt_fea]
            test_grid_ten = [torch.from_numpy(i[:, :2]).to(configx.gpu_device) for i in test_grid]

            predict_labels = polarseg(test_pt_fea_ten, test_grid_ten)
            predict_labels = torch.argmax(predict_labels, 1).type(torch.uint8)
            predict_labels = predict_labels.cpu().detach().numpy()

            test_pred_label = predict_labels[0, test_grid[0][:, 0], test_grid[0][:, 1], test_grid[0][:, 2]] + 1
            test_pred_label = np.expand_dims(test_pred_label, axis=1)
            np.save(ddir_lidseg + filenum + ".npy", test_pred_label)

            pred_lidseg = np.load(ddir_lidseg + filenum + ".npy")  # class id 1‑19

            if configx.lidar_sensor == "rs32":
                ptx_ten = torch.tensor(in_velodyne[:, 1], device=configx.gpu_device, dtype=configx.dtype) * -1
                pty_ten = torch.tensor(in_velodyne[:, 2], device=configx.gpu_device, dtype=configx.dtype)
                ptz_ten = torch.tensor(in_velodyne[:, 0], device=configx.gpu_device, dtype=configx.dtype)
            else:  # hdl32e
                ptx_ten = torch.tensor(in_velodyne[:, 0], device=configx.gpu_device, dtype=configx.dtype)
                pty_ten = torch.tensor(in_velodyne[:, 2], device=configx.gpu_device, dtype=configx.dtype)
                ptz_ten = torch.tensor(in_velodyne[:, 1], device=configx.gpu_device, dtype=configx.dtype)

            ptseg_ten = torch.tensor(pred_lidseg[:, 0], device=configx.gpu_device, dtype=configx.dtype)

            bev_seg, bev_dep, front_seg, front_dep, rear_seg, rear_dep = \
                gen_bev_front_rear_seg_dep(configx, ptx_ten, pty_ten, ptz_ten, ptseg_ten)

            # colourise and save
            bev_segcol = colorize_seg(bev_seg.cpu().detach().numpy(), configx.SEG_CLASSES['colors'])
            bev_depcol = colorize_logdepth(bev_dep.cpu().detach().numpy())
            front_segcol = colorize_seg(front_seg.cpu().detach().numpy(), configx.SEG_CLASSES['colors'])
            front_depcol = colorize_logdepth(front_dep.cpu().detach().numpy())
            rear_segcol = colorize_seg(rear_seg.cpu().detach().numpy(), configx.SEG_CLASSES['colors'])
            rear_depcol = colorize_logdepth(rear_dep.cpu().detach().numpy())

            cv2.imwrite(ddir_lidseg_bev + filenum + ".png", bev_segcol)
            cv2.imwrite(ddir_liddep_bev + filenum + ".png", bev_depcol)
            cv2.imwrite(ddir_lidseg_fro + filenum + ".png", front_segcol)
            cv2.imwrite(ddir_liddep_fro + filenum + ".png", front_depcol)
            cv2.imwrite(ddir_lidseg_rea + filenum + ".png", rear_segcol)
            cv2.imwrite(ddir_liddep_rea + filenum + ".png", rear_depcol)

print("Processing completed.")