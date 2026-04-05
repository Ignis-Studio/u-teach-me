import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.shortcut(x))


class SimpleEncoder(nn.Module):
    """轻量 CNN 编码器，输出 512 维特征"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            ConvBlock(64,  128, stride=2),
            ConvBlock(128, 256, stride=1),
            ConvBlock(256, 512, stride=1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)  # (B, 512)


class IDMModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = SimpleEncoder()

        # 分类头：全局平均池化后分类（分类不需要空间信息）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # 坐标回归头：在特征图上直接回归，保留空间信息
        # 两帧特征图拼接后 1024 通道，用 1x1 卷积压缩，再预测坐标
        self.coord_conv = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(84 * 84, 2),
            nn.Sigmoid(),
        )

    def forward(self, frame_t, frame_t1):
        feat_t  = self.encoder(frame_t)   # (B, 512, 7, 7)
        feat_t1 = self.encoder(frame_t1)  # (B, 512, 7, 7)

        # 分类用 frame_t 的特征就够了
        action_logits = self.classifier(feat_t)

        # 坐标：拼接两帧特征图，在空间维度上找变化位置
        feat_cat = torch.cat([feat_t, feat_t1], dim=1)  # (B, 1024, 7, 7)
        heatmap  = self.coord_conv(feat_cat)             # (B, 1, 7, 7)
        coords   = self.regressor(heatmap.view(heatmap.size(0), -1))  # (B, 2)

        return action_logits, coords