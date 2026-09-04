# PENGEMBANGAN DARI: https://colab.research.google.com/drive/1ssW3JFR1_JakjDcP2vlWdeZiW15MDGDF?authuser=1#scrollTo=WfFQU0I_zG_-

from pypcd import pypcd #https://github.com/dimatura/pypcd/issues/7 #pip3 install --upgrade git+https://github.com/klintan/pypcd.git 
from torch import torch
torch.backends.cudnn.benchmark = True
from PIL import Image
import cv2
import numpy as np
from scipy import interpolate
import os
from data_util import colorize_seg, cls2one_hot, colorize_depth, colorize_depthlog
from config import GlobalConfig
configx = GlobalConfig()

""""""
#INIT MODEL
#pip install -q git+https://github.com/huggingface/transformers.git
#cek pretrained model di sini https://huggingface.co/models?search=mask2former
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")


#loop pada semua route
route_list = os.listdir(configx.datadir)
route_list.sort()
for route in route_list:
    # if route in route_listx: #kalau termasuk route yang tidak diproses, skip
    #     continue
    if os.path.isfile(configx.datadir+route):  #kalau dia file, maka skip
        continue
    ddir_meta = configx.datadir+route+"/meta/"
    ddir_rgb = configx.datadir+route+"/camera/rgb/"
    ddir_depmap = configx.datadir+route+"/camera/depth/map/"
    ddir_depcld = configx.datadir+route+"/camera/depth/cld/"
    ddir_depcld2 = configx.datadir+route+"/camera/depth/cld2/"

    #buat direktori buat nyimpan depth, segmentation GT, dan lainnya
    ddir_depimg = configx.datadir+route+"/camera/depth/img/"
    ddir_seg = configx.datadir+route+"/camera/seg/map/"
    ddir_segimg = configx.datadir+route+"/camera/seg/img/"
    
    os.makedirs(ddir_depimg, exist_ok=True)
    os.makedirs(ddir_seg, exist_ok=True)
    os.makedirs(ddir_segimg, exist_ok=True)


    file_list = os.listdir(ddir_meta)
    file_list.sort()

    with torch.no_grad():
        for filex in file_list:
            file_name = filex[:-3]+"png"
            if os.path.isfile(ddir_depimg+file_name) and os.path.isfile(ddir_segimg+file_name):
                print("file exist")
            else:
                #VISUALIZE DEPTH
                #FRONT
                """
                # load depth cloud
                dep_pc = pypcd.PointCloud.from_path(ddir_depcld+file_name[:-3]+"pcd")
                lid_x = dep_pc.pc_data['x']
                lid_y = dep_pc.pc_data['y']
                lid_z = dep_pc.pc_data['z']
                pt_cloud = np.zeros(lid_x.shape[0] + lid_y.shape[0] + lid_z.shape[0], dtype=np.float32)
                pt_cloud[0::3] = lid_x
                pt_cloud[1::3] = lid_y
                pt_cloud[2::3] = lid_z
                pt_cloud = pt_cloud.astype('float32').reshape((-1, 3))
                print(pt_cloud.shape)
                print(pt_cloud.max())
                print(pt_cloud.min())
                
                dep_pc2 = np.load(ddir_depcld2+file_name[:-3]+"npy")
                print(dep_pc2['x'].shape)
                print(dep_pc2['x'].max())
                print(dep_pc2['x'].min())
                
                print("-------------------------------------------------------")
                """

                """"""
                depmap = np.load(ddir_depmap+file_name[:-3]+"npy")
                depmap = np.nan_to_num(depmap, nan=0.3, posinf=10.0, neginf=0.3)
                # print(depmap.shape)
                # print(depmap.min())
                # print(depmap.max())
                # print(depmap)
                depimg = colorize_depth(depmap)
                cv2.imwrite(ddir_depimg+file_name, depimg)
                
                """
                depmap = np.load(ddir_depmap+file_name[:-3]+"npy")
                depmap = np.nan_to_num(depmap, nan=np.nan, posinf=10, neginf=0.3)
                mask = np.where(~np.isnan(depmap))
                interp = interpolate.NearestNDInterpolator(np.transpose(mask), depmap[mask])
                depmap_clean = interp(*np.indices(depmap.shape))
                depimg = colorize_depthlog(depmap_clean)
                cv2.imwrite(ddir_depimg+file_name, depimg)
                """
                

                """"""
                #VISUALIZE SEGMENTATION
                #FRONT CAMERA
                image = Image.open(ddir_rgb+file_name)
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                seg_pred = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
                #save hasil prediksi sebagai GT, nama filenya sama dengan rgb
                cv2.imwrite(ddir_seg+file_name, seg_pred.cpu().detach().numpy())
                
                #CETAK GAMBAR BERWARNA SS dan SDC dalam format RGB JUGA
                seg_gt = cls2one_hot(cv2.imread(ddir_seg+file_name)+1, n_class=len(configx.cityscapes_palette)) #jadinya 19xHxW one hot
                seg_gt = torch.from_numpy(np.array(np.expand_dims(seg_gt, axis=0))).to(configx.gpu_device) #jadiin tensor, tambah 1 axis buat batch
                fro_segcol = colorize_seg(seg_gt.cpu().detach().numpy(), configx.cityscapes_palette)

                #simpan gambar             #switch RGB BGR
                cv2.imwrite(ddir_segimg+file_name, fro_segcol[:,:,[2,1,0]])
                print(ddir_seg+file_name)
                
           
