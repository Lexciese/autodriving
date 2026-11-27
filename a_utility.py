import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from config import GlobalConfig

def swap_RGB2BGR(matrix):
    red = matrix[:,:,0].copy()
    blue = matrix[:,:,2].copy()
    matrix[:,:,0] = blue
    matrix[:,:,2] = red
    return matrix

# # project into NWU system
def compute_imu_yaw(q, offset=0.0):
    r = R.from_quat(q)
    # Project the sensor's X-axis (+X Forward) into world horizontal frame
    x_world = r.apply([1, 0, 0])
    # Compute Compass Yaw: arctan2(East, North)
    yaw_rad = np.arctan2(-x_world[0], x_world[1])
    return (yaw_rad + offset + np.pi) % (2.0 * np.pi) - np.pi

def magneto_to_yaw(mx, my, offset=0.0):
    yaw = np.arctan2(-my, mx)
    return ((yaw + offset) + np.pi) % (2.0 * np.pi) - np.pi

def new_yaw_from_quaternion(w, x, y, z, offset=0.0):
    """
    Convert quaternion [x, y, z, w] (ENU frame) to CCW yaw in radians [-pi, pi)
    matching latlon_to_yaw (0 = North, +pi/2 = West, -pi/2 = East).
    """
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(-cosy_cosp, siny_cosp)
    return ((yaw + offset) + np.pi) % (2 * np.pi) - np.pi


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
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y*2 + z*2))
        yaw_list.append(((yaw + offset) + np.pi) % (2 * np.pi) - np.pi)
    return yaw_list

def colorize_depth(depth_map):
    norm_dep = depth_map / 10.0 #diubah terjauh 1, terdekat 0 karena dari ros2 message, max 9.99999, min 0.3
    norm_dep = -1*norm_dep + 1 #dibalik terjauh 0, terdekat 1
    visdep = np.repeat(norm_dep[:, :, np.newaxis], 3, axis=2) * 255 #normalisasi ke 0 - 255
    return visdep

def lidar_to_histogram_features(lidar, configx):
    """Convert a LiDAR point cloud into a 2-channel BEV histogram.

    ``lidar`` must have columns ``(right, forward, up)`` in metres.  The
    returned array is ``(2, bev_h, bev_w)``: channel 0 contains points at or
    below the ground plane and channel 1 contains points above it.  Rows are
    forward and columns are lateral, with the vehicle at the bottom centre.
    """
    lat_bins = np.linspace(
        -configx.cover_area_lr,
        configx.cover_area_lr,
        configx.bev_w + 1,
    )
    fwd_bins = np.linspace(
        configx.cover_area_f[0],
        configx.cover_area_f[1],
        configx.bev_h + 1,
    )
    hist_max_per_pixel = getattr(configx, "hist_max_per_pixel", 5)
    ground_z = getattr(configx, "lidar_ground_z", 0.0)

    def splat_points(point_cloud):
        # histogramdd returns (lateral, forward); transpose to (forward, lateral)
        hist = np.histogramdd(
            point_cloud[:, [0, 1]],
            bins=(lat_bins, fwd_bins),
        )[0]
        hist = np.clip(hist, 0, hist_max_per_pixel) / float(hist_max_per_pixel)
        # row 0 = far, last row = near (vehicle at bottom)
        return np.flipud(hist.T).astype(np.float32)

    below = lidar[lidar[:, 2] <= ground_z]
    above = lidar[lidar[:, 2] > ground_z]
    below_features = splat_points(below)
    above_features = splat_points(above)
    return np.stack([below_features, above_features], axis=0)

def resize_img(image, resize_w=256, resize_h=128):
    resized_image = cv2.resize(image, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
    return resized_image

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

def crop_matrix(image, resize=1, D3=True, crop=[512, 1024]):

    # print(image.shape)
    # upper_left_yx = [int((image.shape[0]/2) - (crop/2)), int((image.shape[1]/2) - (crop/2))]
    upper_left_yx = [int((image.shape[0]/2) - (crop[0]/2)), int((image.shape[1]/2) - (crop[1]/2))]
    if D3: #buat matrix 3d
        cropped_im = image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1], :]
    else: #buat matrix 2d
        cropped_im = image[upper_left_yx[0]:upper_left_yx[0]+crop[0], upper_left_yx[1]:upper_left_yx[1]+crop[1]]

    #resize image
    WH_resized = (int(cropped_im.shape[1]/resize), int(cropped_im.shape[0]/resize))
    resized_image = cv2.resize(cropped_im, WH_resized, interpolation=cv2.INTER_NEAREST)

    return resized_image

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

#buat ngecek GT SEG aja
def check_gt_seg(config: GlobalConfig, gt_seg):
    gt_seg = gt_seg.cpu().detach().numpy()

    #buat array untuk nyimpan out gambar
    imgx = np.zeros((gt_seg.shape[2], gt_seg.shape[3], 3))
    #ambil tensor segmentationnya
    inx = np.argmax(gt_seg[0], axis=0)
    for cmap in config.SEG_CLASSES['colors']:
        cmap_id = config.SEG_CLASSES['colors'].index(cmap)
        imgx[np.where(inx == cmap_id)] = cmap

    #GANTI ORDER BGR KE RGB, SWAP!
    imgx = swap_RGB2BGR(imgx)
    cv2.imwrite(str(config.logdir / "check_gt_seg.png"), imgx) #cetak gt segmentation


#Class untuk penyimpanan dan perhitungan update loss
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    #update kalkulasi
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#Class NN Module untuk Perhitungan BCE Dice Loss
def BCEDice(Yp, Yt, smooth=1e-7):
    #.view(-1) artinya matrix tensornya di flatten kan dulu
    Yp = Yp.view(-1)
    Yt = Yt.view(-1)
    #hitung BCE
    bce = F.binary_cross_entropy(Yp, Yt, reduction='mean')
    #hitung dice loss
    intersection = (Yp * Yt).sum() #irisan
    #rumus DICE
    dice_loss = 1 - ((2. * intersection + smooth) / (Yp.sum() + Yt.sum() + smooth))
    #kalkulasi lossnya
    bce_dice_loss = bce + dice_loss
    return bce_dice_loss


#fungsi renormalize loss weights seperti di paper gradnorm
def renormalize_params_lw(current_lw, config: GlobalConfig):
    #detach dulu paramsnya dari torch, pindah ke CPU
    lw = np.array([tens.cpu().detach().numpy() for tens in current_lw])
    lws = np.array([lw[i][0] for i in range(len(lw))])
    #fungsi renormalize untuk algoritma 1 di papaer gradnorm
    coef = np.array(config.loss_weights).sum()/lws.sum()
    new_lws = [coef*lwx for lwx in lws]
    #buat torch float tensor lagi dan masukkan ke cuda memory
    normalized_lws = [torch.cuda.FloatTensor([lw]).clone().detach().requires_grad_(True) for lw in new_lws]
    return normalized_lws

def hampel_filter(lat_pt, lon_pt, latlon_buffer, n_sigmas=3.0):
    k = 1.4826
    latlon_buffer['lat_buf'].append(lat_pt)
    latlon_buffer['lon_buf'].append(lon_pt)

    if len(latlon_buffer['lat_buf']) < 3:
        return lat_pt, lon_pt, False

    win_lat = np.array(latlon_buffer['lat_buf'])
    med_lat = np.median(win_lat)
    mad_lat = np.median(np.abs(win_lat - med_lat))
    thresh_lat = k * mad_lat * n_sigmas

    win_lon = np.array(latlon_buffer['lon_buf'])
    med_lon = np.median(win_lon)
    mad_lon = np.median(np.abs(win_lon - med_lon))
    thresh_lon = k * mad_lon * n_sigmas

    curr_lat = latlon_buffer['lat_buf'][-1]
    curr_lon = latlon_buffer['lon_buf'][-1]

    is_lat_outlier = (thresh_lat > 0) and (np.abs(curr_lat - med_lat) > thresh_lat)
    is_lon_outlier = (thresh_lon > 0) and (np.abs(curr_lon - med_lon) > thresh_lon)

    if is_lat_outlier or is_lon_outlier:
        last_correct_lat = latlon_buffer['lat_buf'][-2]
        last_correct_lon = latlon_buffer['lon_buf'][-2]
        
        latlon_buffer['lat_buf'][-1] = last_correct_lat
        latlon_buffer['lon_buf'][-1] = last_correct_lon
        
        return last_correct_lat, last_correct_lon, True

    return curr_lat, curr_lon, False

def bearing_filter(raw_bearing_rad, buffer):
    buffer['sin'].append(np.sin(raw_bearing_rad))
    buffer['cos'].append(np.cos(raw_bearing_rad))

    avg_sin = np.mean(buffer['sin'])
    avg_cos = np.mean(buffer['cos'])

    return np.arctan2(avg_sin, avg_cos)


def bias_slope(in_angle, bias_start, bias_end, angle_start, angle_end):
    """Linearly interpolate bias across sector transition boundaries."""
    if angle_end == angle_start:
        return bias_start
    ratio = (in_angle - angle_start) / (angle_end - angle_start)
    return bias_start + ratio * (bias_end - bias_start)


def bearing_biasing(in_angle, bearing_bias):
    """Apply 6-sector piecewise-linear bias correction to raw IMU yaw (deg)."""
    if isinstance(bearing_bias, dict):
        bias_map = bearing_bias.get("bearing_bias_deg", bearing_bias)
        b0 = bias_map["0 to 60"]
        b1 = bias_map["60 to 120"]
        b2 = bias_map["120 to 180"]
        b3 = bias_map["-180 to -120"]
        b4 = bias_map["-120 to -60"]
        b5 = bias_map["-60 to 0"]
    else:
        b0, b1, b2, b3, b4, b5 = bearing_bias

    if 0 <= in_angle < 50:
        bias_x = b0
    elif 50 <= in_angle < 70:
        bias_x = bias_slope(in_angle, b0, b1, 50, 70)
    elif 70 <= in_angle < 110:
        bias_x = b1
    elif 110 <= in_angle < 130:
        bias_x = bias_slope(in_angle, b1, b2, 110, 130)
    elif 130 <= in_angle < 170:
        bias_x = b2
    elif 170 <= in_angle <= 180:
        bias_x = bias_slope(in_angle, b2, (b2 + b3) / 2.0, 170, 180)
    elif -180 <= in_angle < -170:
        bias_x = bias_slope(in_angle, (b2 + b3) / 2.0, b3, -180, -170)
    elif -170 <= in_angle < -130:
        bias_x = b3
    elif -130 <= in_angle < -110:
        bias_x = bias_slope(in_angle, b3, b4, -130, -110)
    elif -110 <= in_angle < -70:
        bias_x = b4
    elif -70 <= in_angle < -50:
        bias_x = bias_slope(in_angle, b4, b5, -70, -50)
    elif -50 <= in_angle < 0:
        bias_x = b5
    else:
        bias_x = 0.0

    biased_angle = in_angle + bias_x

    # Normalize to [-180, 180]
    if biased_angle > 180:
        bearing_veh_deg = biased_angle - 360
    elif biased_angle < -180:
        bearing_veh_deg = biased_angle + 360
    else:
        bearing_veh_deg = biased_angle

    return bearing_veh_deg
