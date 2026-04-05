"""
IDM 量化验证脚本
用法：python eval_idm.py --data dataset/click_005
不传 --data 则在整个 dataset 上评估
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# ─── 配置（需和 train_idm.py 保持一致）────────────────────────────────────────

IMG_SIZE     = 672
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ACTION_TYPES = ['click', 'dblclick', 'move', 'key', 'scroll']
TYPE_TO_IDX  = {t: i for i, t in enumerate(ACTION_TYPES)}
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# 屏幕分辨率，用于把归一化坐标换算回像素
SCREEN_W = 1920
SCREEN_H = 1080

# ─── 复制自 train_idm.py ──────────────────────────────────────────────────────

def load_image(path):
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(MEAN)) / np.array(STD)
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).float()


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

from utils import *

# ─── 评估 ────────────────────────────────────────────────────────────────────

def evaluate(data_dir):
    samples = list(Path(data_dir).rglob('action.json'))
    if not samples:
        print(f'在 {data_dir} 下找不到任何样本')
        return

    # 加载模型
    model = IDMModel(len(ACTION_TYPES)).to(DEVICE)
    model.load_state_dict(torch.load('idm_best.pth', map_location=DEVICE))
    model.eval()

    # 按类型统计
    type_correct  = defaultdict(int)
    type_total    = defaultdict(int)
    type_coord_err = defaultdict(list)  # 像素误差列表

    with torch.no_grad():
        for action_json in samples:
            sample_dir = action_json.parent

            ft  = load_image(sample_dir / 'frame_t.png').unsqueeze(0).to(DEVICE)
            ft1 = load_image(sample_dir / 'frame_t1.png').unsqueeze(0).to(DEVICE)

            with open(action_json) as f:
                action = json.load(f)

            gt_type  = action['type']
            gt_x_px  = action.get('x', 0)
            gt_y_px  = action.get('y', 0)

            logits, coords = model(ft, ft1)
            pred_type  = ACTION_TYPES[logits.argmax(1).item()]
            pred_x_px  = coords[0][0].item() * SCREEN_W
            pred_y_px  = coords[0][1].item() * SCREEN_H

            # 分类
            type_total[gt_type] += 1
            if pred_type == gt_type:
                type_correct[gt_type] += 1

            # 坐标误差（像素，只对有坐标意义的类型统计）
            if gt_type in ('click', 'dblclick', 'move'):
                err = ((pred_x_px - gt_x_px)**2 + (pred_y_px - gt_y_px)**2) ** 0.5
                type_coord_err[gt_type].append(err)

    # ── 打印结果 ──
    print(f'\n{"="*50}')
    print(f'评估路径：{data_dir}')
    print(f'总样本数：{sum(type_total.values())}')
    print(f'{"="*50}')

    print(f'\n【分类准确率】')
    total_correct = sum(type_correct.values())
    total_samples = sum(type_total.values())
    print(f'  总体：{total_correct/total_samples:.3f} ({total_correct}/{total_samples})')
    for t in ACTION_TYPES:
        n = type_total[t]
        if n == 0:
            continue
        acc = type_correct[t] / n
        bar = '█' * int(acc * 20) + '░' * (20 - int(acc * 20))
        print(f'  {t:10s}: {bar} {acc:.3f} ({type_correct[t]}/{n})')

    print(f'\n【坐标误差（像素）】')
    for t in ('click', 'dblclick', 'move'):
        errs = type_coord_err[t]
        if not errs:
            continue
        mean_err = sum(errs) / len(errs)
        median_err = sorted(errs)[len(errs)//2]
        max_err = max(errs)
        print(f'  {t:10s}: 平均={mean_err:.1f}px  中位数={median_err:.1f}px  最大={max_err:.1f}px')

    print(f'\n{"="*50}')
    print('参考标准：分类准确率 > 0.8，坐标误差 < 20px 视为通过')
    print(f'{"="*50}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset',
                        help='评估的数据目录，默认为整个 dataset/')
    args = parser.parse_args()
    evaluate(args.data)
