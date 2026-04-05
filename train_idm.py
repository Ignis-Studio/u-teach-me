"""
IDM 训练脚本（无 torchvision 依赖）
逆动力学模型：给定前后两帧，预测中间发生的动作类型和坐标
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# ─── 配置 ────────────────────────────────────────────────────────────────────

DATASET_DIR = Path('dataset')
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-4
VAL_SPLIT   = 0.2
PATIENCE    = 10
IMG_SIZE    = 672
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ACTION_TYPES = ['click', 'dblclick', 'move', 'key', 'scroll']
TYPE_TO_IDX  = {t: i for i, t in enumerate(ACTION_TYPES)}
NUM_CLASSES  = len(ACTION_TYPES)

# ImageNet 均值和标准差，手动归一化用
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

print(f'使用设备: {DEVICE}')

# ─── 图像处理（不依赖 torchvision）────────────────────────────────────────────

def load_image(path):
    """加载图片，resize，归一化，返回 (3, H, W) tensor"""
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
    # 归一化
    arr = (arr - np.array(MEAN)) / np.array(STD)
    # HWC → CHW
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).float()

# ─── 数据集 ──────────────────────────────────────────────────────────────────

class IDMDataset(Dataset):
    def __init__(self, samples):
        self.data = []
        print(f'  加载 {len(samples)} 个样本到内存...')
        for sample_dir in samples:
            img_t  = load_image(sample_dir / 'frame_t.png')
            img_t1 = load_image(sample_dir / 'frame_t1.png')
            with open(sample_dir / 'action.json') as f:
                action = json.load(f)
            action_type = TYPE_TO_IDX[action['type']]
            x = action.get('x', 0) / 1920
            y = action.get('y', 0) / 1080
            self.data.append((
                img_t,
                img_t1,
                torch.tensor(action_type, dtype=torch.long),
                torch.tensor([x, y], dtype=torch.float32),
            ))
        print(f'  加载完成')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_samples():
    return [s.parent for s in DATASET_DIR.rglob('action.json')]

# ─── 模型 ────────────────────────────────────────────────────────────────────

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




# ─── 训练 ────────────────────────────────────────────────────────────────────

def train():
    all_samples = load_samples()
    random.shuffle(all_samples)

    n_val         = int(len(all_samples) * VAL_SPLIT)
    val_samples   = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    print(f'训练集: {len(train_samples)} 样本，验证集: {len(val_samples)} 样本')

    train_loader = DataLoader(IDMDataset(train_samples), batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(IDMDataset(val_samples),   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model       = IDMModel(NUM_CLASSES).to(DEVICE)
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.MSELoss()
    optimizer   = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss    = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # ── 训练 ──
        model.train()
        train_cls, train_reg, correct, total = 0, 0, 0, 0

        for ft, ft1, action_type, coords in train_loader:

            ft, ft1 = ft.to(DEVICE), ft1.to(DEVICE)
            action_type = action_type.to(DEVICE)
            coords = coords.to(DEVICE)

            optimizer.zero_grad()
            logits, pred_coords = model(ft, ft1)

            loss = cls_loss_fn(logits, action_type) + reg_loss_fn(pred_coords, coords)
            loss.backward()
            optimizer.step()

            train_cls += cls_loss_fn(logits, action_type).item()
            train_reg += reg_loss_fn(pred_coords, coords).item()
            correct   += (logits.argmax(1) == action_type).sum().item()
            total     += len(action_type)

        # ── 验证 ──
        model.eval()
        val_cls, val_reg, val_correct, val_total = 0, 0, 0, 0
        per_type_correct = {t: 0 for t in ACTION_TYPES}
        per_type_total   = {t: 0 for t in ACTION_TYPES}

        with torch.no_grad():
            for ft, ft1, action_type, coords in val_loader:
                ft, ft1 = ft.to(DEVICE), ft1.to(DEVICE)
                action_type = action_type.to(DEVICE)
                coords = coords.to(DEVICE)

                logits, pred_coords = model(ft, ft1)
                val_cls += cls_loss_fn(logits, action_type).item()
                val_reg += reg_loss_fn(pred_coords, coords).item()

                preds = logits.argmax(1)
                val_correct += (preds == action_type).sum().item()
                val_total   += len(action_type)

                for pred, gt in zip(preds.cpu(), action_type.cpu()):
                    gt_name = ACTION_TYPES[gt.item()]
                    per_type_total[gt_name] += 1
                    if pred.item() == gt.item():
                        per_type_correct[gt_name] += 1

        val_loss = val_cls + val_reg
        scheduler.step()

        print(f'Epoch {epoch:3d}/{EPOCHS} | '
              f'train_acc={correct/total:.3f} '
              f'cls={train_cls/len(train_loader):.4f} '
              f'reg={train_reg/len(train_loader):.4f} | '
              f'val_acc={val_correct/val_total:.3f} '
              f'val_loss={val_loss/len(val_loader):.4f}')

        if epoch % 10 == 0:
            print('  各类型准确率：')
            for t in ACTION_TYPES:
                n = per_type_total[t]
                if n > 0:
                    print(f'    {t:10s}: {per_type_correct[t]/n:.3f} ({per_type_correct[t]}/{n})')

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'idm_best.pth')
            print(f'  ✓ 保存最优模型 (val_loss={val_loss/len(val_loader):.4f})')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'早停：验证集 loss 连续 {PATIENCE} 轮未改善')
                break

    print('训练完成，最优模型保存在 idm_best.pth')


if __name__ == '__main__':
    train()
