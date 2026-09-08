import numpy as np
import yaml
import cv2
import torch
import os
torch.backends.cudnn.benchmark = True


#polarseg
from pypcd import pypcd #https://github.com/dimatura/pypcd/issues/7 #pip3 install --upgrade git+https://github.com/klintan/pypcd.git 
from preprocessing.polarseg.network.BEV_Unet import BEV_Unet
from preprocessing.polarseg.network.ptBEV import ptBEVnet
from preprocessing.data_util import preproc_spherical, gen_bev_front_rear_seg_dep, colorize_seg, colorize_logdepth
from preprocessing.config import GlobalConfig
configx = GlobalConfig()
# print(configx.polarseg_weight_path)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=configx.gpu_id#visible_gpu #"0" "1" "0,1"


#init model
BEV_model=BEV_Unet(n_class=configx.n_class_kitti-1, n_height=configx.grid_size[2], input_batch_norm=True, dropout=0.5, circular_padding=True)
polarseg = ptBEVnet(BEV_model, pt_model='pointnet', grid_size=configx.grid_size, fea_dim=9, max_pt_per_encode=256, out_pt_fea_dim=512, kernal_size=1, pt_selection='random', fea_compre=configx.grid_size[2])
polarseg.load_state_dict(torch.load(configx.polarseg_weight_path))
polarseg.to(configx.gpu_device)
polarseg.eval()


#loop pada semua route
route_list = os.listdir(configx.datadir)
route_list.sort()
for route in route_list:
    # if route in route_listx: #kalau termasuk route yang tidak diproses, skip
    #     continue
    if os.path.isfile(configx.datadir+route):  #kalau dia file, maka skip
        continue
    print(route)
    ddir_meta = configx.datadir+route+"/meta/"
    ddir_lidar = configx.datadir+route+"/lidar/cld/"
    #buat dir prediksi dan visualisasi
    ddir_lidseg = configx.datadir+route+"/lidar/seg/"
    ddir_lidseg_bev = configx.datadir+route+"/lidar/img/bev_seg/"
    ddir_lidseg_fro = configx.datadir+route+"/lidar/img/front_seg/"
    ddir_lidseg_rea = configx.datadir+route+"/lidar/img/rear_seg/"
    ddir_liddep_bev = configx.datadir+route+"/lidar/img/bev_dep/"
    ddir_liddep_fro = configx.datadir+route+"/lidar/img/front_dep/"
    ddir_liddep_rea = configx.datadir+route+"/lidar/img/rear_dep/"
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
            lid_pc = pypcd.PointCloud.from_path(ddir_lidar+filenum+".pcd")
            lid_x = lid_pc.pc_data['x']
            lid_y = lid_pc.pc_data['y']
            lid_z = lid_pc.pc_data['z']
            if 'intensity' in lid_pc.pc_data.dtype.names:
                lid_intensity = lid_pc.pc_data['intensity']
            else:
                lid_intensity = np.ones(lid_x.shape[0], dtype=np.float32)
            in_velodyne = np.zeros(lid_x.shape[0] + lid_y.shape[0] + lid_z.shape[0] + lid_intensity.shape[0], dtype=np.float32)
            in_velodyne[0::4] = lid_x
            in_velodyne[1::4] = lid_y
            in_velodyne[2::4] = lid_z
            in_velodyne[3::4] = lid_intensity
            in_velodyne = in_velodyne.astype('float32').reshape((-1, 4))
            # in_velodyne = np.fromfile(ddir_lidar+filenum+".bin", dtype=np.float32).reshape((-1, 4))

            # print("3-----", len(in_velodyne[:,3]))
            # print("max: ", in_velodyne[:,3].max())
            # print("min: ", in_velodyne[:,3].min())
            # print(len(np.where(in_velodyne[:,3] > 100)[0]))

            """"""
            #OPERASI PADA CPU
            #preprocess spherical dan pindah ke torch GPU
            test_grid, test_pt_fea = preproc_spherical(in_velodyne, configx)
            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(configx.gpu_device) for i in test_pt_fea]
            test_grid_ten = [torch.from_numpy(i[:,:2]).to(configx.gpu_device) for i in test_grid]

            #PREDIKSI
            predict_labels = polarseg(test_pt_fea_ten,test_grid_ten)
            predict_labels = torch.argmax(predict_labels,1).type(torch.uint8)
            predict_labels = predict_labels.cpu().detach().numpy()

            #postprocessing
            test_pred_label = predict_labels[0, test_grid[0][:,0], test_grid[0][:,1], test_grid[0][:,2]] +1 #tambahkan label 0 karena sebelumnya dikurangi 1 saat init model
            # print(test_pred_label.shape)
            test_pred_label = np.expand_dims(test_pred_label, axis=1) 
            # print(test_pred_label.shape)
            np.save(ddir_lidseg+filenum+".npy", test_pred_label)
            

            #plot bev dan front lidar segmentation
            pred_lidseg = np.load(ddir_lidseg+filenum+".npy") #class id 1 - 19

            # print("shape")
            # print(in_velodyne.shape) #[:, xyzr]
            # print(pred_lidseg.shape)
            # print(in_velodyne[:,3])

            #VELODYNE VLP32C / KITTI / mid360 / rs32 --> xyz = 120 dan x dikali dengan -1
            #VELODYNE HDL32E --> xyz = 021
            if configx.lidar_sensor == "rs32":
                ptx_ten = torch.tensor(in_velodyne[:,1]).to(configx.gpu_device, dtype=configx.dtype) * -1
                pty_ten = torch.tensor(in_velodyne[:,2]).to(configx.gpu_device, dtype=configx.dtype)
                ptz_ten = torch.tensor(in_velodyne[:,0]).to(configx.gpu_device, dtype=configx.dtype)
            else: #hdl32e
                ptx_ten = torch.tensor(in_velodyne[:,0]).to(configx.gpu_device, dtype=configx.dtype)
                pty_ten = torch.tensor(in_velodyne[:,2]).to(configx.gpu_device, dtype=configx.dtype)
                ptz_ten = torch.tensor(in_velodyne[:,1]).to(configx.gpu_device, dtype=configx.dtype)
            ptseg_ten = torch.tensor(pred_lidseg[:,0]).to(configx.gpu_device, dtype=configx.dtype) #[:,0] karena di expand dim, shapenya jadi (N_pt clod, 1), jadi harus dibuat [:,0], untuk inference ini bisa dihilangkan
            """
            #OPERASI TENSOR PADA GPU
            in_velodyne_ten = torch.from_numpy(in_velodyne).to(configx.gpu_device, dtype=configx.dtype)
            test_grid_ten, test_pt_fea_ten = torch_preproc_spherical(in_velodyne_ten, configx)

            #PREDIKSI
            predict_labels = torch.argmax(polarseg([test_pt_fea_ten],[test_grid_ten[:,:2]]),1)

            #postprocessing
            ptseg_ten = predict_labels[0, test_grid_ten[:,0], test_grid_ten[:,1], test_grid_ten[:,2]] +1 #tambahkan label 0 karena sebelumnya dikurangi 1 saat init model

            #SAVE, PINDAH KE CPU NUMPY Dulu
            test_pred_label = np.expand_dims(ptseg_ten.cpu().detach().numpy(), axis=1) 
            np.save(ddir_lidseg+filenum+".npy", test_pred_label)

            #VELODYNE VLP32C / SEMANTIC KITTI --> xyz = 120 dan x dikali dengan -1
            #VELODYNE HDL32E --> xyz = 021
            if configx.lidar_sensor == "vlp32c":
                ptx_ten = in_velodyne_ten[:,1] * -1
                pty_ten = in_velodyne_ten[:,2]
                ptz_ten = in_velodyne_ten[:,0]
            else: #hdl32e
                ptx_ten = in_velodyne_ten[:,0]
                pty_ten = in_velodyne_ten[:,2]
                ptz_ten = in_velodyne_ten[:,1]
            """

            bev_seg, bev_dep, front_seg, front_dep, rear_seg, rear_dep = gen_bev_front_rear_seg_dep(configx, ptx_ten, pty_ten, ptz_ten, ptseg_ten) #pti_ten
            # print(bev_map.shape)
            # print(front_map.shape)
            bev_segcol = colorize_seg(bev_seg.cpu().detach().numpy(), configx.SEG_CLASSES['colors'])
            bev_depcol = colorize_logdepth(bev_dep.cpu().detach().numpy())
            front_segcol = colorize_seg(front_seg.cpu().detach().numpy(), configx.SEG_CLASSES['colors'])  
            front_depcol = colorize_logdepth(front_dep.cpu().detach().numpy()) 
            rear_segcol = colorize_seg(rear_seg.cpu().detach().numpy(), configx.SEG_CLASSES['colors'])  
            rear_depcol = colorize_logdepth(rear_dep.cpu().detach().numpy()) 

            cv2.imwrite(ddir_lidseg_bev+filenum+".png", bev_segcol)
            cv2.imwrite(ddir_liddep_bev+filenum+".png", bev_depcol)
            cv2.imwrite(ddir_lidseg_fro+filenum+".png", front_segcol)
            cv2.imwrite(ddir_liddep_fro+filenum+".png", front_depcol)
            cv2.imwrite(ddir_lidseg_rea+filenum+".png", rear_segcol)
            cv2.imwrite(ddir_liddep_rea+filenum+".png", rear_depcol)



"""


#load ID and color map
with open("semantic-kitti.yaml", "r") as stream:
    kitti_config = yaml.safe_load(stream)

idx = kitti_config['learning_map_inv']
color = kitti_config['color_map']
color_map = idx
for id, cls in idx.items():
    color_map[id] = color[cls]



filename = "000000"
in_velodyne = np.fromfile(seq_folder+"velodyne/"+filename+".bin", dtype=np.float32).reshape((-1, 4))
out_pred_np = np.load(seq_folder+"predictions_np/"+filename+".npy") #classnya 1 -> 19
out_gt = np.fromfile(seq_folder+"labels/"+filename+".label", dtype=np.float32)
out_pred = np.fromfile(seq_folder+"predictions/"+filename+".label", dtype=np.float32)
out_pred_remapped = np.fromfile(seq_folder+"predictions_remapped/"+filename+".label", dtype=np.float32)


print("shape")
print(in_velodyne.shape)
print(out_gt.shape)
print(out_pred.shape)
print(out_pred_remapped.shape)
print()


print("classes")
uniq_cls_gt = np.unique(out_gt)
uniq_cls_pred = np.unique(out_pred)
uniq_cls_pred_remapped = np.unique(out_pred_remapped)
print(uniq_cls_gt)
print(uniq_cls_pred)
print(uniq_cls_pred_remapped)
print()



print(out_pred_np.shape)
# print(color_map)

print("------------------------------------------------------------------------")

#buat array kosong untuk menyimpan output gambar
bev_sem = np.zeros((out_pred_np.shape[1], out_pred_np.shape[2], 3))
idx = out_pred_np[0,:,:,0]#.reshape(out_pred_np.shape[1], out_pred_np.shape[2]) 

print(idx)

seg_cmap = list(color_map.values())
print(seg_cmap)
for cmap in seg_cmap:
    cmap_id = seg_cmap.index(cmap)
    bev_sem[np.where(idx == cmap_id)] = cmap
# bev_sem = bev_sem[:, :, [2, 1, 0]]
# print(bev_sem)

cv2.imwrite(filename+".png", bev_sem)
"""