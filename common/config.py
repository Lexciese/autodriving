import numpy as np
import torch
import os
from pathlib import Path
from datetime import datetime
from PIL import ImageFont

try:
    import preprocessing
    _PREPROCESSING_DIR = Path(preprocessing.__file__).resolve().parent
    _REPO_ROOT = _PREPROCESSING_DIR.parent
except Exception:
    _PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "preprocessing"
    _REPO_ROOT = _PREPROCESSING_DIR.parent

_env_root = os.environ.get("AUTODRIVING_ROOT")
if _env_root:
    _REPO_ROOT = Path(_env_root).resolve()
    _PREPROCESSING_DIR = _REPO_ROOT / "preprocessing"


class GlobalConfig:
    now = datetime.now()
    string_date = now.strftime("%d_%m_%Y-%H_%M")
    use_gpu = True

    # data paths
    datadir = str(_REPO_ROOT / "datasetx") + "/"

    # image / bev dimensions
    bev_h = lidbev_h = 128
    bev_w = lidbev_w = 256
    front_h = lidfront_h = 64
    front_w = lidfront_w = 512
    cam_h = 128
    cam_w = 256
    dvs_h = cam_h
    dvs_w = cam_w

    # recording / route
    hz = 4 #1 detik ada berapa sample yang direcord
    bias_basic = 15
    bearing_bias = [-bias_basic, bias_basic, 2*bias_basic+5, bias_basic, -bias_basic+10, -bias_basic] #dalam derajat #bias untuk 0 ke 60, 60 ke 120, 120 ke 180, -180 ke -120, -120 ke -60, -60 ke 0
    rp1_close = 4 #jarak minimum untuk ganti rp1 (dalam meter)
    route_gap_distance = 6 #dalam meter
    n_buffer = 0 #dalam sekon buat MAF
    bs = 1 #batch size

    # for training
    gpu_id = '0'
    inputs = 'segdep' #segdep seg dep
    # perspectives = 'bevfro' #bevfro bev fro
    logdir = 'log/xr20_'+inputs#+'_'+perspectives
    init_stop_counter = 30
    batch_size = 4
    lr = 1e-4 # learning rate #pakai AdamW
    weight_decay = 1e-3
    #parameter untuk MGN
    MGN = True
    loss_weights = [1, 1, 1] #wp, mlp st, mlp th
    lw_alpha = 1.5
    bottleneck = 64 # #cek dengan check_arch.py
    n_fmap = [48, 96, 192, 384]

    # Data
    seq_len = 1 # jumlah input seq
    pred_len = 3 # future waypoints predicted
    n_wp = pred_len #waypoints
    wp_gap = int(hz*5) #berapa frame?
    gap_bearing = wp_gap #buat estimasi bearing berapa frame?
    root_dir = str(_REPO_ROOT / 'dataset' / 'dataset')
    logdir = root_dir+logdir+"_seq"+str(seq_len)+f"_{string_date}" #update direktori name
    train_dir = root_dir+'/train_routes'
    val_dir = root_dir+'/val_routes'
    test_dir = root_dir+'/test_routes'
    train_conditions = ['noon', 'evening', 'night'] #
    val_conditions = ['noon', 'evening', 'night'] #pokoknya kebalikannya train
    test_conditions = ['noon0', 'evening0', 'night0',
                        'noon1', 'evening1', 'night1',
                        'noon2', 'evening2', 'night2']

    # Controller
    #control weights untuk PID dan MLP dari tuningan MGN
    #urutan steering, throttle
    #baca dulu trainval_log.csv setelah training selesai, dan normalize bobotnya 0-1
    #LWS: lw_wp lw_str lw_thr saat convergence
    lws = [1, 1, 1]
    cw_pid = [lws[0]/(lws[0]+lws[1]), lws[0]/(lws[0]+lws[2])] #str, thrt
    cw_mlp = [1-cw_pid[0], 1-cw_pid[1]] #str, thrt, brk

    turn_KP = 0.5
    turn_KI = 0.25
    turn_KD = 0.15
    turn_n = 15 # buffer size

    speed_KP = 1.5
    speed_KI = 0.25
    speed_KD = 0.5
    speed_n = 15 # buffer size

    n_cmd = 3 #jumlah command yang ada: 0 lurus, 1 kiri, 2 kanan
    max_throttle = 1.0 # upper limit on throttle signal value in dataset
    wheel_radius = 0.15#radius roda robot dalam meter
    # brake_speed = 0.4 # desired speed below which brake is triggered
    # brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    # clip_delta = 0.25 # maximum change in speed input to logitudinal controller
    min_act_thrt = 0.1 #minimum nilai suatu throttle dianggap aktif diinjak
    err_angle_mul = 0.075
    des_speed_mul = 1.75

    # buat preprocessing data
    polarseg_weight_path = str(_PREPROCESSING_DIR / "polarseg" / "SemKITTI_PolarSeg.pt")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpu_device = torch.device("cuda:0")
    dtype = torch.float32
    cover_area_lr = lid_cover_area_lr = 16 #kiri - kanan
    cover_area_up = lid_cover_area_bt = [-1.5, 6.5] #bawah -> atas
    cover_area_f = lid_cover_area_rf = [1.25, 17.25] #posisi camera -> area interest max
    cam_cover_area_lr = 8 #kiri - kanan
    cam_cover_area_rf = [0, 16] #posisi camera -> area interest max
    SEG_CLASSES = { #lihat di file semantic-kitti.yaml
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
    n_class = len(SEG_CLASSES['colors'])

    #lidar setting, cek HDL-32E dan VLP32C LiDAR sensor datasheet
    lidar_sensor = "rs32" #vlp32c hdl32e mid360
    if lidar_sensor == "hdl32e":
        v_fov = [-30.67, 10.67] # HDL32 pakai [-30.67, 10.67], VLP32 pakai [-25, 15]
        dep_max = 70#/1.25 #dalam meter, baca datasheet np.sqrt(lid_cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (lid_cover_area_bt[1]-((lid_cover_area_bt[1]-lid_cover_area_bt[0])/2))**2)
        v_res_div = 60
    elif lidar_sensor == "rs32":
        v_fov = [-16, 15]
        dep_max = 150
        v_res_div = 60 # 50 # 60
    else: #"mid360" #rs32
        v_fov = [-15.5, 10.5] # HDL32 pakai [-30.67, 10.67], VLP32 pakai [-25, 15]
        dep_max = 200#/1.25 #dalam meter, baca datasheet np.sqrt(lid_cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (lid_cover_area_bt[1]-((lid_cover_area_bt[1]-lid_cover_area_bt[0])/2))**2)
        v_res_div = 50
    max_intensity = 100.0
    # v_fov_down = -1*np.radians(2)
    # v_fov_up = np.radians(24.9)
    # n_laser = 32
    # lidar_rps = 10 #rotasi per detik --> 600 rpm / 60 detik
    h_fov = 360
    # v_fov = [-25, 15] # HDL32 pakai [-30.67, 10.67], VLP32 pakai [-25, 15]
    v_fov_total = -v_fov[0] + v_fov[1]

    v_res = v_fov_total/v_res_div         #n_laser #front_h  # 1.33 #vertical resolution
    h_res = h_fov/(front_w*2)              #0.35 #horizontal resolution
    # Convert to Radians
    v_res_rad = v_res * (np.pi/180)
    h_res_rad = h_res * (np.pi/180)
    # y_fudge = 5

    #config polarseg
    ignore_label = 0
    grid_size = np.asarray([480,360,32])
    max_volume_space = np.asarray([50,np.pi,1.5])
    min_volume_space = np.asarray([3,-np.pi,-3])
    intervals = (max_volume_space - min_volume_space) / (grid_size-1)
    #untuk operasi langsung tensor
    grid_size_ten = torch.from_numpy(np.asarray([480,360,32])).to(gpu_device, dtype=dtype)
    max_volume_space_ten = torch.from_numpy(np.asarray([50,np.pi,1.5])).to(gpu_device, dtype=dtype)
    min_volume_space_ten = torch.from_numpy(np.asarray([3,-np.pi,-3])).to(gpu_device, dtype=dtype)
    intervals_ten = (max_volume_space_ten - min_volume_space_ten) / (grid_size_ten-1)

    #untuk front_dep dan bev_dep
    #100 untuk HDL32E, 200 untuk VLP32C
    # dep_max = 200#/1.25 #dalam meter, baca datasheet np.sqrt(cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (cover_area_up[1]-((cover_area_up[1]-cover_area_up[0])/2))**2)
    dep_min = cover_area_f[0]

    #settingan segformer
    segformer_weight_path = str(_PREPROCESSING_DIR / "segformer" / "segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth")
    segformer_config_path = str(_PREPROCESSING_DIR / "segformer" / "configs" / "segformer" / "segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py")
    #BACA https://mmsegmentation.readthedocs.io/en/latest/_modules/mmseg/core/evaluation/class_names.html#get_palette
    #HANYA ADA 19 CLASS?? #tambahan 0,0,0 hitam untuk area kosong pada SDC nantinya
    cityscapes_palette = [[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
            [0, 80, 100], [0, 0, 230], [119, 11, 32]] #,
    n_class_cityscape = len(cityscapes_palette)

    #other, buat join_img dll
    fontsize = 15
    font_mul = 1
    try:
        fontx = ImageFont.truetype(font="arial.ttf", size=font_mul*fontsize) #arialbold arial
    except OSError:
        fontx = ImageFont.load_default()
    #jika error font tidak ketemu, donlod dulu di https://www.freefontspro.com/14454/arial.ttf lalu copas foldernya ke /usr/share/fonts/truetype/
    text_gap = (fontsize+4)*font_mul
    metadata_gap = 110*font_mul

    speedup_factor = 15
    fps = 20
    rgb_res_ori = [720, 1280] #HxW HD720
    scale_w = rgb_res_ori[1]/front_w
    scaled_H_rgb = int(rgb_res_ori[0]/scale_w)

    vid_size = (4*cam_w, 4*cam_h) #W x H horizontal
    #vid_size = (3*cam_w, 8*cam_h) #W x H vertikal
    # scale_w_dvs = 2*dvs_res_ori[1]/lidfront_w #dikali 2 karena rgb dan dvs dari davis diconcat pada sumbu w-nya dulu nanti
    # scaled_H_dvs = int(dvs_res_ori[0]/scale_w_dvs)

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
