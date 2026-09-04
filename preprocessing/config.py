import numpy as np
import torch
import os
from PIL import ImageFont

class GlobalConfig:
    datadir = "../temp_datasetx/"
    cam_h = 128
    cam_w = 256
    lidbev_h = 256
    lidbev_w = 256
    lidfront_h = 64
    lidfront_w = 512
    dvs_h = cam_h
    dvs_w = cam_w
    # w = 256
    bs = 1 #batch size

    # route_gap_time = 10 #dalam second, sesuaikan dengan kcepatan (v=~1.25m/s, maka gap dalam meter = v x t = 1.25 x 10 = ~12.5 meter)
    route_gap_distance = 6 #dalam meter
    hz = 4 #1 detik ada berapa sample yang direcord, cek dan hitung manual di meta yml
    n_buffer = 5 #dalam sekon buat MAF
    num_wp = 5 #waypoints
    num_rp = 2
    wp_gap = int(hz*5) #berapa frame?
    gap_bearing = wp_gap #buat estimasi bearing berapa frame?
    # bias_basic = 0
    # bearing_bias = [bias_basic, bias_basic+7, bias_basic, -bias_basic+5, -bias_basic+7, bias_basic] #dalam derajat #bias untuk 0 ke 50, 50 ke 120, 120 ke 180, -180 ke -120, -120 ke -60, -60 ke 0
    # bearing_bias = [0, 0, 0, 0, 0, 0]
    rp1_close = 1 #jarak minimum untuk ganti rp1 (dalam meter)

    #settingan polarseg
    polarseg_weight_path = os.path.join(os.getcwd(), "polarseg/SemKITTI_PolarSeg.pt")
    gpu_id = "0"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id#visible_gpu #"0" "1" "0,1"
    gpu_device = torch.device("cuda:0")
    dtype = torch.float32
    lid_cover_area_lr = 24 #kiri - kanan
    lid_cover_area_bt = [-2, 7] #bawah -> atas
    lid_cover_area_rf = [-24, 24] #posisi belakang lidar -> depan lidar
    cam_cover_area_lr = [-24, 24]  # left - right coverage area
    cam_cover_area_rf = [2, 40]  # rear - front coverage area
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
    
    #lidar setting, cek HDL-32E dan VLP32C LiDAR sensor datasheet
    lidar_sensor = "rs32" #vlp32c hdl32e mid360
    if lidar_sensor == "hdl32e":
        v_fov = [-30.67, 10.67] # HDL32 pakai [-30.67, 10.67], VLP32 pakai [-25, 15]
        dep_max = 70#/1.25 #dalam meter, baca datasheet np.sqrt(lid_cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (lid_cover_area_bt[1]-((lid_cover_area_bt[1]-lid_cover_area_bt[0])/2))**2)
        v_res_div = 60
    elif lidar_sensor == "rs32":
        v_fov = [-16, 15]
        dep_max = 150
        v_res_div = 60
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

    v_res = v_fov_total/v_res_div         #n_laser #lidfront_h  # 1.33 #vertical resolution
    h_res = h_fov/(lidfront_w*2)              #0.35 #horizontal resolution
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
    # dep_max = 200#/1.25 #dalam meter, baca datasheet np.sqrt(lid_cover_area_lr**2 + (cover_area_f[1]-cover_area_f[0])**2 + (lid_cover_area_bt[1]-((lid_cover_area_bt[1]-lid_cover_area_bt[0])/2))**2)
    dep_min = 0#lid_cover_area_rf[0]



    #settingan segformer
    segformer_weight_path = os.path.join(os.getcwd(), "segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth")
    segformer_config_path = os.path.join(os.getcwd(), "segformer/configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py")
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
    fontx = ImageFont.truetype(font="arial.ttf", size=font_mul*fontsize) #arialbold arial
    #jika error font tidak ketemu, donlod dulu di https://www.freefontspro.com/14454/arial.ttf lalu copas foldernya ke /usr/share/fonts/truetype/
    # fontx = ImageFont.load_default()
    text_gap = (fontsize+4)*font_mul
    metadata_gap = 110*font_mul

    speedup_factor = 15
    fps = int(hz * speedup_factor)
    rgb_res_ori = [720, 1280] #HxW HD720
    scale_w = rgb_res_ori[1]/lidfront_w
    scaled_H_rgb = int(rgb_res_ori[0]/scale_w)

    vid_size = (4*cam_w, 4*cam_h) #W x H horizontal
    #vid_size = (3*cam_w, 8*cam_h) #W x H vertikal
    # scale_w_dvs = 2*dvs_res_ori[1]/lidfront_w #dikali 2 karena rgb dan dvs dari davis diconcat pada sumbu w-nya dulu nanti
    # scaled_H_dvs = int(dvs_res_ori[0]/scale_w_dvs)
    
