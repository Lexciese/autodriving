import torch
from torch import cat, nn
import torchvision.models as models
import numpy as np

from config import GlobalConfig

def kaiming_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class ai33(nn.Module):
    def __init__(self, config: GlobalConfig, device):
        super().__init__()
        self.config: GlobalConfig = config
        self.gpu_device = device

        feature_map = [48, 96, 192, 384]
        if config.inputs == 'segdep':
            in_ch = config.n_class + 1
        elif config.inputs == 'seg':
            in_ch = config.n_class
        else: # dep
            in_ch = 1

        # BEV EfficientNet Encoder
        self.bev_encoder = models.efficientnet_b0(weights=None)
        self.bev_encoder.features[0][0] = nn.Conv2d(
            in_channels=in_ch, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bev_encoder.classifier = nn.Sequential()
        self.bev_encoder.avgpool = nn.Sequential()
        self.bev_encoder.apply(kaiming_init)

        # Front View EfficientNet Encoder
        self.fro_encoder = models.efficientnet_b0(weights=None)
        self.fro_encoder.features[0][0] = nn.Conv2d(
            in_channels=in_ch, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.fro_encoder.classifier = nn.Sequential()
        self.fro_encoder.avgpool = nn.Sequential()
        self.fro_encoder.apply(kaiming_init)

        eff_out_channels = 1280

        self.align_pool = nn.AdaptiveAvgPool2d((4, 8))

        # NecksNet feature fusion sequential block
        self.necks_net = nn.Sequential(
            nn.Conv2d(in_channels=2 * eff_out_channels, out_channels=eff_out_channels, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=eff_out_channels, out_features=feature_map[2])
        )

        self.gru = nn.GRUCell(input_size=7, hidden_size=feature_map[2])
        self.pred_dwp = nn.Sequential( 
            nn.Linear(feature_map[2], feature_map[1]),
            nn.Linear(feature_map[1], 2)
        )
        
    def forward(self, bev_segs, bev_deps, front_segs, front_deps, rp1, rp2, velo_in):
        BEV_features_sum = 0
        FRO_features_sum = 0

        for i in range(self.config.seq_len):
            if self.config.inputs == 'segdep':
                bev_in = cat([bev_segs[i], bev_deps[i]], dim=1)
                fro_in = cat([front_segs[i], front_deps[i]], dim=1)
            elif self.config.inputs == 'seg':
                bev_in = bev_segs[i]
                fro_in = front_segs[i]
            else: # dep
                bev_in = bev_deps[i]
                fro_in = front_deps[i]

            bev_feat = self.bev_encoder.features(bev_in)
            fro_feat = self.fro_encoder.features(fro_in)

            # Align spatial shapes to [4, 8] for tensor concatenation
            bev_feat = self.align_pool(bev_feat)
            fro_feat = self.align_pool(fro_feat)

            BEV_features_sum += bev_feat
            FRO_features_sum += fro_feat

        hx = self.necks_net(cat([BEV_features_sum, FRO_features_sum], dim=1))

        xy = torch.zeros(size=(hx.shape[0], 2)).to(self.config.gpu_device, dtype=self.config.dtype)
        out_wp = list()

        for _ in range(self.config.pred_len):
            ins = torch.cat([xy, rp1, rp2, velo_in], dim=1)
            hx = self.gru(ins, hx)
            d_xy = self.pred_dwp(hx)
            xy = xy + d_xy
            out_wp.append(xy)

        pred_wp = torch.stack(out_wp, dim=1)
        return pred_wp