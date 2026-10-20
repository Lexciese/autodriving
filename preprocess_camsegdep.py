import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from config import GlobalConfig
from preprocess_util import cls2one_hot, colorize_depth, colorize_seg

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configx = GlobalConfig()

# 1. INIT MODEL & MOVE TO GPU
processor = AutoImageProcessor.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-semantic"
)
model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-semantic"
).to(device)
model.eval()

# Loop across routes
route_list = sorted(p.name for p in configx.datadir.iterdir() if p.is_dir())
if configx.select_route != "all":
    route_list = [configx.select_route]
    print(f"only route: {configx.select_route} is selected")

for route in route_list:
    route_path = configx.datadir / route
    ddir_meta = route_path / "meta"
    ddir_rgb = route_path / "camera" / "rgb"
    ddir_depmap = route_path / "camera" / "depth" / "map"

    ddir_depimg = route_path / "camera" / "depth" / "img"
    ddir_seg = route_path / "camera" / "seg" / "map"
    ddir_segimg = route_path / "camera" / "seg" / "img"

    ddir_depimg.mkdir(parents=True, exist_ok=True)
    ddir_seg.mkdir(parents=True, exist_ok=True)
    ddir_segimg.mkdir(parents=True, exist_ok=True)

    file_list = sorted(p.name for p in ddir_meta.iterdir())

    with torch.inference_mode():
        for filex in tqdm(file_list, desc="Processing camsegdep Files", unit="file"):
            file_name = filex[:-3] + "png"

            if (ddir_depimg / file_name).is_file() and (
                ddir_segimg / file_name
            ).is_file():
                continue

            depmap = np.load(ddir_depmap / f"{file_name[:-3]}npy")
            depmap = np.nan_to_num(depmap, nan=0.3, posinf=10.0, neginf=0.3)
            depimg = colorize_depth(depmap)
            cv2.imwrite(str(ddir_depimg / file_name), depimg)

            image = Image.open(ddir_rgb / file_name)

            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            seg_pred = processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]

            seg_pred_np = seg_pred.cpu().numpy()

            cv2.imwrite(str(ddir_seg / file_name), seg_pred_np)

            seg_pred_3ch = np.repeat(seg_pred_np[:, :, np.newaxis], 3, axis=2)

            seg_gt = cls2one_hot(
                seg_pred_3ch + 1, n_class=len(configx.cityscapes_palette)
            )
            seg_gt = (
                torch.from_numpy(np.expand_dims(seg_gt, axis=0))
                .to(configx.gpu_device)
            )

            fro_segcol = colorize_seg(
                seg_gt.cpu().numpy(), configx.cityscapes_palette
            )

            cv2.imwrite(
                str(ddir_segimg / file_name), fro_segcol[:, :, [2, 1, 0]]
            )
            print(f"Processed: {ddir_seg / file_name}")