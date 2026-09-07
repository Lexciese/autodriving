import numpy as np
import yaml
import cv2
import torch
import os
from pathlib import Path
from tqdm import tqdm
torch.backends.cudnn.benchmark = True

from pypcd4 import PointCloud
from pypcd4.pointcloud2 import sensor_msgs__msg__PointCloud2
from polarseg.network.BEV_Unet import BEV_Unet
from polarseg.network.ptBEV import ptBEVnet
from data_util import preproc_spherical, gen_bev_front_rear_seg_dep, colorize_seg, colorize_logdepth

from preprocessing import Preprocessing
from config import GlobalConfig

from preprocessing_lidar import PreprocessingLidar

def main():
    config = GlobalConfig()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    config.datadir = "experiment"

    file = "empty_gap_front"
    preproc_lidar = PreprocessingLidar(config)

    route = Path("experiment")
    dir_lidseg = route / "lidar" / "seg"
    # visualisasi
    dir_lidseg_bev = route / "lidar" / "img" / "bev_seg"
    dir_lidseg_fro = route / "lidar" / "img" / "front_seg"
    dir_lidseg_rea = route / "lidar" / "img" / "rear_seg"
    dir_liddep_bev = route / "lidar" / "img" / "bev_dep"
    dir_liddep_fro = route / "lidar" / "img" / "front_dep"
    dir_liddep_rea = route / "lidar" / "img" / "rear_dep"
    dir_lidseg.mkdir(parents=True, exist_ok=True)
    dir_lidseg_bev.mkdir(parents=True, exist_ok=True)
    dir_lidseg_fro.mkdir(parents=True, exist_ok=True)
    dir_lidseg_rea.mkdir(parents=True, exist_ok=True)
    dir_liddep_bev.mkdir(parents=True, exist_ok=True)
    dir_liddep_fro.mkdir(parents=True, exist_ok=True)
    dir_liddep_rea.mkdir(parents=True, exist_ok=True)

    preproc_lidar.paths = {
        'seg': dir_lidseg / f"{file}.npy",
        'bev_seg': dir_lidseg_bev / f"{file}.png",
        'bev_dep': dir_liddep_bev / f"{file}.png",
        'front_seg': dir_lidseg_fro / f"{file}.png",
        'front_dep': dir_liddep_fro / f"{file}.png",
        'rear_seg': dir_lidseg_rea / f"{file}.png",
        'rear_dep': dir_liddep_rea / f"{file}.png"
    }
    preproc_lidar.set_input(f"{file}.pcd")
    preproc_lidar.process()
    preproc_lidar.get_output(to_file=True)


if __name__ == "__main__":
    main()