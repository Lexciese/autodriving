import torch
from torch import cat, nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

def kaiming_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding, stride):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, padding_mode='zeros'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, padding_mode='zeros'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.block1.apply(kaiming_init)
        self.block2.apply(kaiming_init)

    def forward(self, x):
        x = self.block1(x)
        y = self.block2(x)
        return y

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, feature_map, is_bev):
        super().__init__()
        self.layer1 = DoubleConv(in_channels=in_channels, out_channels=feature_map[0], kernel_size=5, dilation=2, padding=3, stride=1)
        self.layer2 = DoubleConv(in_channels=feature_map[0], out_channels=feature_map[1], kernel_size=3, dilation=2, padding=2, stride=1)
        self.layer3 = DoubleConv(in_channels=feature_map[1], out_channels=feature_map[2], kernel_size=3, dilation=1, padding=1, stride=1)
        self.layer4 = DoubleConv(in_channels=feature_map[2], out_channels=feature_map[3], kernel_size=3, dilation=1, padding=1, stride=1)
        if is_bev == True:
            self.pool1 = nn.AvgPool2d(kernel_size=[2, 2])
            self.pool2 = nn.AvgPool2d(kernel_size=[4, 4])
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=[2, 4])
            self.pool2 = nn.AvgPool2d(kernel_size=[2, 4])
        self.pool3 = nn.MaxPool2d(kernel_size=[2, 2])
        self.pool4 = nn.MaxPool2d(kernel_size=[2, 2])

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        y = self.pool4(self.layer4(x))
        return y

class NecksNet(nn.Module):
    def __init__(self, feature_map):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels=2*feature_map[3], out_channels=feature_map[3], kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.dense_layer = nn.Linear(in_features=feature_map[3], out_features=feature_map[2])

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.pool(x)
        x = self.flatten(x)
        y = self.dense_layer(x)
        return y


class xr20(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.gpu_device = device

        feature_map = [48, 96, 192, 384]
        if config.inputs == 'segdep':
            in_ch = config.n_class + 1
        elif config.inputs == 'seg':
            in_ch = config.n_class
        else: # dep
            in_ch = 1

        self.bev_encoder_block = EncoderBlock(in_channels=in_ch, feature_map=feature_map, is_bev=True)
        self.fro_encoder_block = EncoderBlock(in_channels=in_ch, feature_map=feature_map, is_bev=False)
        self.necks_net = NecksNet(feature_map=feature_map)
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
            else: #dep
                bev_in = bev_deps[i]
                fro_in = front_deps[i]
            bev_encoder = self.bev_encoder_block(bev_in)
            fro_encoder = self.fro_encoder_block(fro_in)
            BEV_features_sum += bev_encoder
            FRO_features_sum += fro_encoder

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