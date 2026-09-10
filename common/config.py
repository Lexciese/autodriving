"""Shared configuration for the ai23 and preprocessing packages."""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFont

# Project paths are anchored to the installed ``preprocessing`` package so that a
# config copied into a log directory still resolves the repo root and pretrained weights.
try:
    import preprocessing
    _PREPROCESSING_DIR = Path(preprocessing.__file__).resolve().parent
    _REPO_ROOT = _PREPROCESSING_DIR.parent
except Exception:
    _PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "preprocessing"
    _REPO_ROOT = _PREPROCESSING_DIR.parent


class GlobalConfig:
    # General / device
    now = datetime.now()
    string_date = now.strftime("%d_%m_%Y-%H_%M")
    use_gpu = True
    gpu_id = '0'
    gpu_device = torch.device("cuda:0")
    dtype = torch.float32
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Data paths
    datadir = str(_REPO_ROOT / "datasetx") + "/"
    root_dir = str(_REPO_ROOT / 'dataset' / 'dataset')
    train_dir = root_dir + '/train_routes'
    val_dir = root_dir + '/val_routes'
    test_dir = root_dir + '/test_routes'
    polarseg_weight_path = str(_PREPROCESSING_DIR / "polarseg" / "SemKITTI_PolarSeg.pt")
    segformer_weight_path = str(_PREPROCESSING_DIR / "segformer" / "segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth")
    segformer_config_path = str(_PREPROCESSING_DIR / "segformer" / "configs" / "segformer" / "segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py")

    # Image / BEV dimensions
    bev_h = lidbev_h = 128
    bev_w = lidbev_w = 256
    front_h = lidfront_h = 64
    front_w = lidfront_w = 512
    cam_h = 128
    cam_w = 256
    dvs_h = cam_h
    dvs_w = cam_w

    # Recording / route
    hz = 4  # samples recorded per second
    bias_basic = 15
    bearing_bias = [-bias_basic, bias_basic, 2*bias_basic+5, bias_basic, -bias_basic+10, -bias_basic]  # per-sector bias (deg): 0-60, 60-120, 120-180, -180--120, -120--60, -60-0
    rp1_close = 4  # min distance (m) to advance to the next route point
    route_gap_distance = 6  # in meters
    n_buffer = 0  # moving-average buffer (seconds)

    # Sequence / waypoints
    seq_len = 1  # number of input frames
    pred_len = 3  # future waypoints predicted
    n_wp = pred_len  # waypoints
    wp_gap = int(hz*5)  # frames between waypoints
    gap_bearing = wp_gap  # frames used for bearing estimation

    # Training
    inputs = 'segdep'  # segdep | seg | dep
    logdir_name = 'log/xr20_' + inputs
    logdir = str(_REPO_ROOT / logdir_name) + "_seq" + str(seq_len) + f"_{string_date}"
    init_stop_counter = 30
    batch_size = 4
    lr = 1e-4  # learning rate (AdamW)
    weight_decay = 1e-3
    # MGN (loss-weighting) parameters
    MGN = True
    loss_weights = [1, 1, 1]  # wp, mlp steering, mlp throttle
    lw_alpha = 1.5
    bottleneck = 64  # see check_arch.py
    n_fmap = [48, 96, 192, 384]

    # Dataset conditions
    train_conditions = ['noon', 'evening', 'night']
    val_conditions = ['noon', 'evening', 'night']  # complement of the training conditions
    test_conditions = ['noon0', 'evening0', 'night0',
                       'noon1', 'evening1', 'night1',
                       'noon2', 'evening2', 'night2']

    # Controller
    # PID/MLP control weights from MGN tuning; order: steering, throttle.
    # Read trainval_log.csv after training and normalize the weights to 0-1.
    # LWS: lw_wp, lw_str, lw_thr at convergence.
    lws = [1, 1, 1]
    cw_pid = [lws[0]/(lws[0]+lws[1]), lws[0]/(lws[0]+lws[2])]  # steering, throttle
    cw_mlp = [1-cw_pid[0], 1-cw_pid[1]]  # steering, throttle, brake

    turn_KP = 0.5
    turn_KI = 0.25
    turn_KD = 0.15
    turn_n = 15  # buffer size

    speed_KP = 1.5
    speed_KI = 0.25
    speed_KD = 0.5
    speed_n = 15  # buffer size

    n_cmd = 3  # commands: 0 straight, 1 left, 2 right
    max_throttle = 1.0  # upper throttle limit in dataset
    wheel_radius = 0.15  # robot wheel radius (m)
    min_act_thrt = 0.1  # minimum throttle considered pressed
    err_angle_mul = 0.075
    des_speed_mul = 1.75

    # LiDAR
    # See HDL-32E / VLP32C datasheets.
    lidar_sensor = "rs32"  # vlp32c | hdl32e | mid360
    if lidar_sensor == "hdl32e":
        v_fov = [-30.67, 10.67]  # HDL32: [-30.67, 10.67]; VLP32: [-25, 15]
        dep_max = 70  # in meters (datasheet)
        v_res_div = 60
    elif lidar_sensor == "rs32":
        v_fov = [-16, 15]
        dep_max = 150
        v_res_div = 60
    else:  # mid360
        v_fov = [-15.5, 10.5]
        dep_max = 200  # in meters (datasheet)
        v_res_div = 50
    max_intensity = 100.0
    h_fov = 360
    v_fov_total = -v_fov[0] + v_fov[1]

    v_res = v_fov_total/v_res_div  # vertical resolution (deg)
    h_res = h_fov/(front_w*2)  # horizontal resolution (deg)
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)

    # Coverage areas / depth
    cover_area_lr = lid_cover_area_lr = 16  # left-right
    cover_area_up = lid_cover_area_bt = [-1.5, 6.5]  # bottom-top
    cover_area_f = lid_cover_area_rf = [1.25, 17.25]  # camera position to max area of interest
    cam_cover_area_lr = 8  # left-right
    cam_cover_area_rf = [0, 16]  # camera position to max area of interest
    dep_min = cover_area_f[0]  # minimum depth for front/BEV depth normalization
    # Depth-encoding multipliers
    bev_multiplier = 9
    front_multiplier = 9
    rear_multiplier = 9

    # PolarSeg
    ignore_label = 0
    grid_size = np.asarray([480, 360, 32])
    max_volume_space = np.asarray([50, np.pi, 1.5])
    min_volume_space = np.asarray([3, -np.pi, -3])
    intervals = (max_volume_space - min_volume_space) / (grid_size - 1)
    # Direct tensor variants
    grid_size_ten = torch.from_numpy(grid_size).to(gpu_device, dtype=dtype)
    max_volume_space_ten = torch.from_numpy(max_volume_space).to(gpu_device, dtype=dtype)
    min_volume_space_ten = torch.from_numpy(min_volume_space).to(gpu_device, dtype=dtype)
    intervals_ten = (max_volume_space_ten - min_volume_space_ten) / (grid_size_ten - 1)

    # SemanticKITTI / SegFormer
    # SemanticKITTI classes (see semantic-kitti.yaml)
    SEG_CLASSES = {
        'colors'        :[[0, 0, 0], [245, 150, 100], [245, 230, 100], [150, 60, 30], [180, 30, 80],
                        [255, 0, 0], [30, 30, 255], [200, 40, 255], [90, 30, 150],
                        [255, 0, 255], [255, 150, 255], [75, 0, 75], [75, 0, 175],
                        [0, 200, 255], [50, 120, 255], [0, 175, 0], [0, 60, 135],
                        [80, 240, 150], [150, 240, 255], [0, 0, 255]],
        'classes'       : ['unlabeled', 'car', 'bicycle', 'motorcycle', 'truck',
                            'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
                            'road', 'parking', 'sidewalk', 'other-ground',
                            'building', 'fence', 'vegetation', 'trunk',
                            'terrain', 'pole', 'traffic-sign']
    }
    n_class_kitti = len(SEG_CLASSES['colors'])
    n_class = n_class_kitti
    # mmsegmentation cityscapes palette: 19 classes + black for empty SDC areas
    cityscapes_palette = [[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    n_class_cityscape = len(cityscapes_palette)

    # Preprocessing batch
    bs = 1  # batch size

    # Visualization (for join_img and previews)
    fontsize = 15
    font_mul = 1
    try:
        fontx = ImageFont.truetype(font="arial.ttf", size=font_mul*fontsize)
    except OSError:
        # If arial.ttf is missing, install it under /usr/share/fonts/truetype/.
        fontx = ImageFont.load_default()
    text_gap = (fontsize+4)*font_mul
    metadata_gap = 110*font_mul

    fps = 20
    rgb_res_ori = [720, 1280]  # HxW (HD720)
    scale_w = rgb_res_ori[1]/front_w
    scaled_H_rgb = int(rgb_res_ori[0]/scale_w)
    vid_size = (4*cam_w, 4*cam_h)  # W x H

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def select_logdir(log_root=None):
    log_root = Path(log_root) if log_root else (_REPO_ROOT / "log")
    runs = [d for d in log_root.iterdir() if d.is_dir()] if log_root.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"No log runs found under {log_root}")
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)  # newest first
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
            return str(runs[int(choice) - 1])
        print("Invalid selection, please try again.")
