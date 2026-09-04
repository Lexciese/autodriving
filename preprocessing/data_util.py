import numpy as np
import torch
import cv2
#DARI POLARSEG dataset.py
#fokus cuma ambil grid_ind dan return_fea aja

# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = np.arctan2(input_xyz[:,1],input_xyz[:,0])
    return np.stack((rho,phi,input_xyz[:,2]),axis=1)


def preproc_spherical(raw_data, config):
    #load binfile yang berisi xyz-intensity dan split ke xyz dan sig
    xyz = raw_data[:,:3]
    # xyz = xyz[:,[1, 0, 2]] #tidak perlu karena polarnet sudah robust terhadap rotasi berapapun
    # xyz[:,0] = -1*xyz[:,0] #tidak perlu karena polarnet sudah robust terhadap flip x, y, dan xy
    sig = np.clip(np.squeeze(raw_data[:,3])/config.max_intensity, 0.0, 1.0) #intensity harus discale/clip dalam range 0 - 1.0, baca lidarsegdep_bev_front/check.py

    # convert coordinate into polar coordinates
    xyz_pol = cart2polar(xyz)
    # get grid index
    crop_range = config.max_volume_space - config.min_volume_space
    intervals = crop_range / (config.grid_size-1)

    # if (intervals==0).any(): print("Zero interval!")
    grid_ind = (np.floor((np.clip(xyz_pol,config.min_volume_space,config.max_volume_space)-config.min_volume_space)/intervals)).astype(np.int32)

    # center data on each voxel for PTnet
    voxel_centers = (grid_ind.astype(np.float32) + 0.5)*intervals + config.min_volume_space
    return_xyz = xyz_pol - voxel_centers
    return_xyz = np.concatenate((return_xyz,xyz_pol,xyz[:,:2]), axis=1)
    return_fea = np.concatenate((return_xyz,sig[...,np.newaxis]), axis=1)

    return [grid_ind], [return_fea]



#operasi torch tensor
def torch_cart2polar(input_xyz):
    rho = torch.sqrt(input_xyz[:,0]**2 + input_xyz[:,1]**2)
    phi = torch.atan2(input_xyz[:,1],input_xyz[:,0])
    return torch.stack((rho,phi,input_xyz[:,2]),dim=1)


def torch_preproc_spherical(raw_data, config):
    # convert coordinate into polar coordinates
    xyz_pol = torch_cart2polar(raw_data[:,:3])
    sig = torch.clip(raw_data[:,3]/config.max_intensity, 0.0, 1.0) #intensity harus discale/clip dalam range 0 - 1.0, baca lidarsegdep_bev_front/check.py
    # get grid index
    grid_ind = torch.floor((torch.clip(xyz_pol,config.min_volume_space_ten,config.max_volume_space_ten)-config.min_volume_space_ten)/config.intervals_ten)

    # center data on each voxel for PTnet
    voxel_centers = (grid_ind + 0.5)*config.intervals_ten + config.min_volume_space_ten
    return_xyz = xyz_pol - voxel_centers
    return_xyz = torch.cat((return_xyz,xyz_pol,raw_data[:,:2]), dim=1)
    return_fea = torch.cat((return_xyz,sig[:,None]), dim=1)

    return grid_ind.long(), return_fea



def euler_from_quaternion(w, x, y, z, rad=True): #urutannya q0, q1, q2, q3
    #https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    #https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    if rad:
        return roll_x, pitch_y, yaw_z # in radians
    else:
        return np.degrees(roll_x), np.degrees(pitch_y), np.degrees(yaw_z)




def resizecrop_matrix(image, WH_resized=[256, 128], D3=True, crop_HW=[128, 256]):

    #resize image
    resized_image = cv2.resize(image, WH_resized, interpolation=cv2.INTER_NEAREST)

    # print(image.shape)
    # upper_left_yx = [int((image.shape[0]/2) - (crop/2)), int((image.shape[1]/2) - (crop/2))]
    upper_left_yx = [int((resized_image.shape[0]/2) - (crop_HW[0]/2)), int((resized_image.shape[1]/2) - (crop_HW[1]/2))]
    if D3: #buat matrix 3d
        cropped_im = resized_image[upper_left_yx[0]:upper_left_yx[0]+crop_HW[0], upper_left_yx[1]:upper_left_yx[1]+crop_HW[1], :]
    else: #buat matrix 2d
        cropped_im = resized_image[upper_left_yx[0]:upper_left_yx[0]+crop_HW[0], upper_left_yx[1]:upper_left_yx[1]+crop_HW[1]]


    return cropped_im


def gen_bev_front_rear_seg_dep(self, ptx, pty, ptz, ptseg):
    #flatten all
    ptx = ptx.ravel()
    pty = pty.ravel()
    ptz = ptz.ravel()
    ptz_bev = ptz - self.lid_cover_area_rf[0]
    d_lidar = torch.sqrt(ptx**2 + pty**2 + ptz**2) #jarak relatif # 
    ptseg = ptseg.ravel()
    ptn = torch.ravel(torch.tensor([[n for _ in range(len(ptseg))] for n in range(self.bs)])).to(self.gpu_device, dtype=self.dtype) #dummy batch
    
    #check shape
    # print(ptn.shape)
    # print(ptseg.shape)

    #normalize ke frame untuk BEV projection
    frame_data_x = torch.round((ptx+self.lid_cover_area_lr) * (self.lidbev_w-1) / (2*self.lid_cover_area_lr))
    frame_data_z = torch.round((ptz_bev * (1-self.lidbev_h) / (self.lid_cover_area_rf[1]-self.lid_cover_area_rf[0])) + (self.lidbev_h-1))

    #BEV SEG
    #cari index interest
    boolx = torch.logical_and(frame_data_x <= self.lidbev_w-1, frame_data_x >= 0)
    bool_all = torch.logical_and(boolx, torch.logical_and(frame_data_z <= self.lidbev_h-1, frame_data_z >= 0))
    idx = bool_all.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya
    #stack n x z cls dan plot
    coorx = torch.stack([ptn, ptseg, frame_data_z, frame_data_x])
    coor_clsn = torch.unique(coorx[:, idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
    bev_seg = torch.zeros((self.bs, self.n_class_kitti, self.lidbev_h, self.lidbev_w), dtype=self.dtype, device=self.gpu_device)
    bev_seg[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #format axis dari NCHW


    # bev_dep
    idx_dlidar = torch.nonzero(bool_all) # atau torch.argwhere(bool_all)
    # coorxx = torch.stack([ptn, frame_data_z, frame_data_x])
    # coor_depn = torch.unique(coorxx[:, idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
    bev_dep = torch.zeros((self.bs, 1, self.lidbev_h, self.lidbev_w), dtype=self.dtype, device=self.gpu_device)
    linear_d = torch.clip(((4*(d_lidar[idx_dlidar]-self.dep_min)/(self.dep_max-self.dep_min))+1), min=1.0, max=10.0) #linear 1 - 10
    log_d = -1*torch.log(linear_d) + 1 #logarithmic 1 - 0
    bev_dep[ptn[idx_dlidar].long(), 0, frame_data_z[idx_dlidar].long(), frame_data_x[idx_dlidar].long()] = log_d
    # bev_dep = bev_dep / (self.dep_max - self.dep_min)
    # print(bev_dep)


    #FRONT SEG
    #baca: https://github.com/collector-m/lidar_projection/blob/master/show.py
    # Distance relative to origin when looked from top
    # d_lidar = torch.sqrt(ptx**2 + ptz**2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)


    # PROJECT INTO IMAGE COORDINATES
    front_x_img = torch.atan2(-ptz, ptx)/ self.h_res_rad
    rear_x_img = torch.atan2(ptz, ptx)/ self.h_res_rad
    y_img = torch.atan2(pty, torch.sqrt(ptx**2 + ptz**2))/ self.v_res_rad


    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / self.h_res / 2  # Theoretical min x value based on sensor specs
    front_x_img = front_x_img - x_min              # Shift
    rear_x_img = rear_x_img - x_min              # Shift
    # x_max = int(360.0 / self.h_res)       # Theoretical max x value after shifting

    y_min = self.v_fov[0] / self.v_res    # theoretical min y value based on sensor specs
    y_img = y_img - y_min             # Shift
    y_max = int(self.v_fov_total / self.v_res) # Theoretical max x value after shifting

    # y_max = int(y_max + self.y_fudge)            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.
    y_img = -1 * (y_img - y_max) #di flip

    #normalize ke frame untuk FRONT projection
    #baca https://towardsdatascience.com/spherical-projection-for-point-clouds-56a2fc258e6c
    # ptR = torch.sqrt(torch.pow(ptx,2) + torch.pow(ptx,2) + torch.pow(ptx,2))
    # pt_pitch = torch.asin(ptz/ptR)
    # pt_yaw = torch.atan2(pty,ptx)
    # frame_data_pitch = (self.lidfront_h-1) * (1-(pt_pitch-self.fov_down)/self.fov)
    # frame_data_yaw = (self.lidfront_w-1) * (0.5*((pt_yaw/np.pi)+1))
    # frame_data_y = torch.round(((pty-self.lid_cover_area_bt[0]) * (1-self.lidfront_h) / (self.lid_cover_area_bt[1]-self.lid_cover_area_bt[0])) + (self.lidfront_h-1))

    #cari index interest untuk front
    # boolxz = torch.logical_and(boolx, ptz >= self.lid_cover_area_rf[0]) #ptz >= self.lid_cover_area_rf[0] berarti point2 yang berada didepan vehicle saja
    # bool_all = torch.logical_and(boolxz, torch.logical_and(frame_data_y <= self.lidfront_h-1, frame_data_y >= 0))
    # boolx = torch.logical_and(frame_data_yaw <= self.lidfront_w-1, frame_data_yaw >= 0)
    # bool_all = torch.logical_and(boolx, torch.logical_and(frame_data_pitch <= self.lidfront_h-1, frame_data_pitch >= 0))
    front_boolx = torch.logical_and(front_x_img <= self.lidfront_w-1, front_x_img >= 0)
    front_bool_all = torch.logical_and(front_boolx, torch.logical_and(y_img <= self.lidfront_h-1, y_img >= 0))
    front_idx = front_bool_all.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya
    #stack n x z cls dan plot
    # coorx = torch.stack([ptn, ptseg, frame_data_y, frame_data_x])
    # coorx = torch.stack([ptn, ptseg, frame_data_pitch, frame_data_yaw])
    front_coorx = torch.stack([ptn, ptseg, y_img, front_x_img])
    front_coor_clsn = torch.unique(front_coorx[:, front_idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
    # front_seg = torch.zeros((self.bs, self.n_class_kitti, self.lidfront_h, self.lidfront_w), dtype=self.dtype, device=self.gpu_device)
    front_seg = torch.zeros((self.bs, self.n_class_kitti, self.lidfront_h, self.lidfront_w), dtype=self.dtype, device=self.gpu_device)
    front_seg[front_coor_clsn[0], front_coor_clsn[1], front_coor_clsn[2], front_coor_clsn[3]] = 1.0 #format axis dari NCHW


    # front_dep
    # front_dep = torch.zeros((self.bs, 1, self.lidfront_h, self.lidfront_w), dtype=self.dtype, device=self.gpu_device)
    # coorxx = torch.stack([ptn, d_lidar, y_img, x_img])
    # coor_depn = torch.unique(coorxx[:, idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
    # front_dep[coor_depn[0], 0, coor_depn[2], coor_depn[3]] = d_lidar[coor_depn[1]]
    front_idx_dlidar = torch.nonzero(front_bool_all) # atau torch.argwhere(bool_all)
    # coorxx = torch.stack([ptn, frame_data_z, frame_data_x])
    # coor_depn = torch.unique(coorxx[:, idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
    front_dep = torch.zeros((self.bs, 1, self.lidfront_h, self.lidfront_w), dtype=self.dtype, device=self.gpu_device)
    linear_d = torch.clip(((7*(d_lidar[front_idx_dlidar]-self.dep_min)/(self.dep_max-self.dep_min))+1), min=1.0, max=10.0) #linear 1 - 10
    log_d = -1*torch.log(linear_d) + 1 #logarithmic 1 - 0
    front_dep[ptn[front_idx_dlidar].long(), 0, y_img[front_idx_dlidar].long(), front_x_img[front_idx_dlidar].long()] = log_d
    # front_dep = front_dep / (self.dep_max - self.dep_min)
    # print(front_dep)

    
    
    #cari index interest untuk rear
    rear_boolx = torch.logical_and(rear_x_img <= self.lidfront_w-1, rear_x_img >= 0)
    rear_bool_all = torch.logical_and(rear_boolx, torch.logical_and(y_img <= self.lidfront_h-1, y_img >= 0))
    rear_idx = rear_bool_all.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya
    rear_coorx = torch.stack([ptn, ptseg, y_img, rear_x_img])
    rear_coor_clsn = torch.unique(rear_coorx[:, rear_idx], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
    rear_seg = torch.zeros((self.bs, self.n_class_kitti, self.lidfront_h, self.lidfront_w), dtype=self.dtype, device=self.gpu_device)
    rear_seg[rear_coor_clsn[0], rear_coor_clsn[1], rear_coor_clsn[2], rear_coor_clsn[3]] = 1.0 #format axis dari NCHW

    rear_idx_dlidar = torch.nonzero(rear_bool_all) # atau torch.argwhere(bool_all)
    rear_dep = torch.zeros((self.bs, 1, self.lidfront_h, self.lidfront_w), dtype=self.dtype, device=self.gpu_device)
    linear_d = torch.clip(((9*(d_lidar[rear_idx_dlidar]-self.dep_min)/(self.dep_max-self.dep_min))+1), min=1.0, max=10.0) #linear 1 - 10
    log_d = -1*torch.log(linear_d) + 1 #logarithmic 1 - 0
    rear_dep[ptn[rear_idx_dlidar].long(), 0, y_img[rear_idx_dlidar].long(), rear_x_img[rear_idx_dlidar].long()] = log_d

    return bev_seg, bev_dep, front_seg, front_dep, rear_seg, rear_dep




def gen_top_view_sdc(pt_cloud_x, pt_cloud_z, semseg, configx):
    #init

    #proses awal
    # depth_in = depth * 1000.0 #normalisasi ke 1 - 1000 #sudah di normalisasi
    _, label_img = torch.max(semseg, dim=1) #pada axis C
    cloud_data_n = torch.ravel(torch.tensor([[n for _ in range(semseg.shape[2]*semseg.shape[3])] for n in range(semseg.shape[0])])).to(configx.gpu_device)
    # cloud_data_x = torch.ravel(depth_in * self.x_matrix)
    # cloud_data_z = torch.ravel(depth_in)
    # cloud_data_cls = torch.ravel(label_img)
    
    #normalize ke frame  #pakai depth point_cloud
    cloud_data_x = torch.round((pt_cloud_x + configx.cam_cover_area_lr) * (semseg.shape[3]-1) / (2*configx.cam_cover_area_lr)).ravel()
    cloud_data_z = torch.round((pt_cloud_z * (1-semseg.shape[2]) / (configx.cam_cover_area_rf[1]-configx.cam_cover_area_rf[0])) + (semseg.shape[2]-1)).ravel()


    #cari index interest
    bool_xz = torch.logical_and(torch.logical_and(cloud_data_x <= semseg.shape[3]-1, cloud_data_x >= 0), torch.logical_and(cloud_data_z <= semseg.shape[2]-1, cloud_data_z >= 0))
    idx_xz = bool_xz.nonzero().squeeze() #hilangkan axis dengan size=1, sehingga tidak perlu nambahkan ".item()" nantinya

    #stack n x z cls dan plot
    # print(cloud_data_n.shape)
    # print(label_img.ravel().shape)
    # print(cloud_data_z.shape)
    # print(cloud_data_x.shape)
    coorx = torch.stack([cloud_data_n, label_img.ravel(), cloud_data_z, cloud_data_x])
    coor_clsn = torch.unique(coorx[:, idx_xz], dim=1).long() #tensor harus long supaya bisa digunakan sebagai index
    # coor_clsn = torch.stack([self.cloud_data_n[idx_xz], cloud_data_cls[idx_xz], cloud_data_z[idx_xz], cloud_data_x[idx_xz]])
    # coor_clsn = torch.unique(coor_clsn, dim=1).type(torch.long) #tensor harus long supaya bisa digunakan sebagai index
    # top_view_sc = torch.zeros((depth.shape[0], self.n_class_kitti, self.h, self.w)).float().to(configx.gpu_device)   
    top_view_sc = torch.zeros_like(semseg) #ini lebih cepat karena secara otomatis size, tipe data, dan device sama dengan yang dimiliki inputnya (semseg)
    top_view_sc[coor_clsn[0], coor_clsn[1], coor_clsn[2], coor_clsn[3]] = 1.0 #format axis dari NCHW
    # for j in range(coor_clsn.shape[1]):
    #     top_view_sc[coor_clsn[0][j]][coor_clsn[1][j]][coor_clsn[2][j]][coor_clsn[3][j]] = 1.0 #tidak perlu ".item()"

    return top_view_sc



def cls2one_hot(ss_gt, n_class):
    #inputnya adalah HWC baca cv2 secara biasanya, ambil salah satu channel saja
    ss_gt = np.transpose(ss_gt, (2,0,1)) #GANTI CHANNEL FIRST
    ss_gt = ss_gt[:1,:,:].reshape(ss_gt.shape[1], ss_gt.shape[2])
    result = (np.arange(n_class) == ss_gt[...,None]).astype(int) # jumlah class di cityscape pallete
    result = np.transpose(result, (2, 0, 1))   # (H, W, C) --> (C, H, W)
    # np.save("00009_ss.npy", result) #SUDAH BENAR!
    # print(result)
    # print(result.shape)
    return result


def colorize_seg(sem_map, colmap):
    #buat array kosong untuk menyimpan output gambar
    sem_img = np.zeros((sem_map.shape[2], sem_map.shape[3], 3))
    idx = np.argmax(sem_map[0], axis=0)
    for cmap in colmap:
        cmap_id = colmap.index(cmap)
        sem_img[np.where(idx == cmap_id)] = cmap
    # sem_img = sem_img[:, :, [2, 1, 0]]
    # print(sem_img.shape)
    return sem_img

def colorize_logdepth(depth_map):
    #inputnya sudah 0 - 1
    norm_dep = depth_map[0][0] 
    #dijadikan 1 - 0
    # norm_dep = -1*norm_dep + 1

    # logdepth = np.ones(norm_dep.shape) + (np.log(norm_dep) / 5.70378)
    # logdepth = np.clip(logdepth, 0.0, 1.0) 
    logdepth = np.repeat(norm_dep[:, :, np.newaxis], 3, axis=2) * 255 #normalisasi ke 0 - 255
    return logdepth

def colorize_depth(depth_map):
    norm_dep = depth_map / 10.0 #diubah terjauh 1, terdekat 0 karena dari ros2 message, max 9.99999, min 0.3
    norm_dep = -1*norm_dep + 1 #dibalik terjauh 0, terdekat 1
    visdep = np.repeat(norm_dep[:, :, np.newaxis], 3, axis=2) * 255 #normalisasi ke 0 - 255
    return visdep

def colorize_depthlog(depth_map):
    norm_dep = depth_map / 10.0 #diubah terjauh 1, terdekat 0 karena dari ros2 message, max 9.99999, min 0.3
    norm_dep = (norm_dep - 1) * -1 #dibalik terjauh 0, terdekat 1

    logdepth = np.ones(norm_dep.shape) + (np.log(norm_dep) / 5.70378)
    logdepth = np.clip(logdepth, 0.0, 1.0) 
    visdep = np.repeat(logdepth[:, :, np.newaxis], 3, axis=2) * 255 #normalisasi ke 0 - 255
    return visdep


def visualize_dcloud(clipped_dcloud):
    real_dep = np.sqrt(np.array(clipped_dcloud[0:1,:,:]**2 + clipped_dcloud[1:2,:,:]**2 + clipped_dcloud[2:3,:,:]**2)) #dihitung depthnya
    #dijadikan 0 - 10
    # linear_d = np.clip(((9*(real_dep[0]-0.3)/(40-0.3))+1), a_min=1.0, a_max=10.0) #linear 1 - 10
    # proc_dep = -1*np.log(linear_d) + 1 #logarithmic 1 - 0
    proc_dep = -1*np.clip(real_dep[0]/40, a_min=0.0, a_max=1.0) + 1
    visdep = np.repeat(proc_dep[:, :, np.newaxis], 3, axis=2) * 255 #normalisasi ke 0 - 255
    return visdep



def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out


def plot_sdc_rpwp(configx, local_bev_x, local_bev_y):
    #komputasi dari lokal ke bev frame buat visualisasi
    x_img = (local_bev_x+configx.cam_cover_area_lr)*(configx.cam_w-1)/(2*configx.cam_cover_area_lr)
    y_img = ((local_bev_y-configx.cam_cover_area_rf[0])*(1-configx.cam_h)/(configx.cam_cover_area_rf[1]-configx.cam_cover_area_rf[0])) + (configx.cam_h-1)

    #batasan
    x_img = np.clip(int(x_img), 0, configx.cam_w-1)#constrain
    # nextr_x_frame = np.clip(int((local_bev_x+(cover_area-cover_min)/2)*(www-1)/((cover_area-cover_min))), 0, www-1)#constrain
    y_img = np.clip(int(y_img), 0, configx.cam_h-1)#constrain

    return x_img, y_img



def plot_lidbev_rpwp(configx, local_bev_x, local_bev_y):
    #komputasi dari lokal ke bev frame buat visualisasi
    x_img = (local_bev_x+configx.lid_cover_area_lr)*(configx.lidbev_w-1)/(2*configx.lid_cover_area_lr)
    y_img = ((local_bev_y-configx.lid_cover_area_rf[0])*(1-configx.lidbev_h)/(configx.lid_cover_area_rf[1]-configx.lid_cover_area_rf[0])) + (configx.lidbev_h-1)

    #batasan
    x_img = np.clip(int(x_img), 0, configx.lidbev_w-1)#constrain
    # nextr_x_frame = np.clip(int((local_bev_x+(cover_area-cover_min)/2)*(www-1)/((cover_area-cover_min))), 0, www-1)#constrain
    y_img = np.clip(int(y_img), 0, configx.lidbev_h-1)#constrain

    return x_img, y_img

#baca: https://github.com/collector-m/lidar_projection/blob/master/show.py
def plot_lidfront_rpwp(configx, local_bev_x, local_bev_y):
    y_road = configx.lid_cover_area_bt[0] #-1.5 #ketinggian jalan dari perspektif posisi LiDAR, dalam meter
    xy_euclid = np.sqrt(local_bev_x**2 + local_bev_y**2)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-local_bev_y, local_bev_x)/ configx.h_res_rad
    y_img = np.arctan2(y_road, xy_euclid)/ configx.v_res_rad


    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / configx.h_res / 2  # Theoretical min x value based on sensor specs
    x_img = x_img - x_min              # Shift
    # x_max = int(360.0 / configx.h_res)       # Theoretical max x value after shifting

    y_min = configx.v_fov[0] / configx.v_res    # theoretical min y value based on sensor specs
    y_img = y_img - y_min              # Shift
    y_max = int(configx.v_fov_total / configx.v_res) # Theoretical max x value after shifting

    # y_max = int(y_max + configx.y_fudge)            # Fudge factor if the calculations based on
                                # spec sheet do not match the range of
                                # angles collected by in the data.
    y_img = -1 * (y_img - y_max) #di flip

    #batasan
    x_img = np.clip(int(x_img), 0, configx.lidfront_w-1)#constrain
    y_img = np.clip(int(y_img), 0, configx.lidfront_h-1)#constrain

    return x_img, y_img


def bias_slope(angle_x, bias_a, bias_b, angle_a, angle_b):
    bias_x = (((angle_x-angle_a)/(angle_b-angle_a)) * (bias_b-bias_a)) + bias_a
    return bias_x

def bearing_biasing(in_angle, bearing_bias):
    if 0 <= in_angle < 40:
        bias_x = bearing_bias[0]
    elif 40 <= in_angle < 70:
        bias_x = bias_slope(in_angle, bearing_bias[0], bearing_bias[1], 40, 70)
    elif 70 <= in_angle < 110:
        bias_x = bearing_bias[1]
    elif 110 <= in_angle < 130:
        bias_x = bias_slope(in_angle, bearing_bias[1], bearing_bias[2], 110, 130)
    elif 130 <= in_angle < 170:
        bias_x = bearing_bias[2]
    elif 170 <= in_angle <= 180:
        bias_x = bias_slope(in_angle, bearing_bias[2], (bearing_bias[2]+bearing_bias[3])/2, 170, 180)
    elif -180 <= in_angle < -170:
        bias_x = bias_slope(in_angle, (bearing_bias[2]+bearing_bias[3])/2, bearing_bias[3], -180, -170)
    elif -170 <= in_angle < -130:
        bias_x = bearing_bias[3]
    elif -130 <= in_angle < -110:
        bias_x = bias_slope(in_angle, bearing_bias[3], bearing_bias[4], -130, -110)
    elif -110 <= in_angle < -70:
        bias_x = bearing_bias[4]
    elif -70 <= in_angle < -50:
        bias_x = bias_slope(in_angle, bearing_bias[4], bearing_bias[5], -70, -50)
    elif -50 <= in_angle < 0:
        bias_x = bearing_bias[5]
    else:
        bias_x = 0
    biased_angle = in_angle+bias_x

    #normalkan
    if biased_angle > 180: #buat jadi -180 ke 0
        bearing_veh_deg = biased_angle - 360
    elif biased_angle < -180: #buat jadi 180 ke 0
        bearing_veh_deg = biased_angle + 360
    else:
        bearing_veh_deg = biased_angle

    return bearing_veh_deg






from collections import deque
class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0
    
    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        out_control = self._K_P * error + self._K_I * integral + self._K_D * derivative
        return out_control


def pid_control(waypoints, linear_velo_ms, turn_controller, speed_controller):
        #vehicular controls dari PID
        aim_point = (waypoints[1] + waypoints[0]) / 2.0 #tengah2nya wp0 dan wp1
        #90 deg ke kanan adalah 0 radian, 90 deg ke kiri adalah 1*pi radian
        angle_rad = np.clip(np.arctan2(aim_point[1], aim_point[0]), 0, np.pi) #arctan y/x
        angle_deg = np.degrees(angle_rad)
        #ke kiri adalah 0 -> +1 == 90 -> 180, ke kanan adalah 0 -> -1 == 90 -> 0
        error_angle = (angle_deg - 90.0) * 0.08
        pid_steering = turn_controller.step(error_angle)
        pid_steering = np.clip(pid_steering, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[1] - waypoints[0]) * 2.75
        pid_throttle = speed_controller.step(desired_speed - linear_velo_ms)
        pid_throttle = np.clip(pid_throttle, 0.0, 1.0)

        brake = 0
        if pid_throttle <= 0.1:
            pid_throttle = 0
            pid_steering = 0
            brake = 1

        return pid_steering, pid_throttle, brake


def latlon_to_yaw(lat, lon, lat0, lon0, offset=0.0):
    lat, lon, lat0, lon0 = map(np.radians, [lat, lon, lat0, lon0])
    dlon = lon - lon0
    x = np.sin(dlon) * np.cos(lat)
    y = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)
    yaw = np.arctan2(-x, y)
    return ((yaw + offset) + np.pi) % (2 * np.pi) - np.pi

def quaternion_to_yaw(quat: list, offset=0.0):
    yaw_list = []
    for q in quat:
        w, x, y, z = q[3], q[0], q[1], q[2]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*2 + z*2)) + 1.5708 #1.5708 offset, diputer 90 degree
        yaw_list.append(((yaw + offset) + np.pi) % (2 * np.pi) - np.pi)
    return yaw_list