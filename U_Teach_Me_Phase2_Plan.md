# U Teach Me — IDM 第二阶段执行方案
> SigLIP + Temporal Conv + Attention 架构

---

## 背景与目标

第一阶段验证了数据管道的正确性，同时证明轻量 CNN（SimpleEncoder）无法胜任 IDM 任务。根本原因有二：

- 没有预训练视觉编码器，从零学特征，1552 个样本远不够
- 两帧输入（t-N, t+N）信息量不足——GUI 操作后屏幕变化有延迟，相邻帧几乎一致

第二阶段目标：

- 用 **SigLIP** 替换 CNN 编码器，获得强大的预训练视觉特征
- 将输入从两帧改为**连续帧序列**（3 秒 × 20fps = 60 帧），IDM 一次性预测整段序列的动作
- 架构忠实于 VPT：`SigLIP → Temporal Conv → Unmasked Attention → 动作预测头`

---

## 架构设计

### 整体结构（对标 VPT）

| 模块 | 说明 |
|------|------|
| 图像编码器 | SigLIP so400m-patch14-384（冻结），替换 VPT 的 ResNet |
| Temporal Conv | `nn.Conv1d`，跨相邻帧捕捉局部时序变化 |
| Unmasked Attention | `nn.MultiheadAttention`，non-causal，全序列建立长程依赖 |
| 动作预测头 | 分类头（动作类型）+ 回归头（归一化坐标） |

### 关键设计参数

| 参数 | 值 |
|------|----|
| 输入序列长度 | 60 帧（3 秒 × 20fps） |
| 视频统一帧率 | 20fps（对齐 VPT） |
| 有效标签帧 | 全部 60 帧 |
| SigLIP 输入分辨率 | 384×384 |
| SigLIP 输出维度 | 1152 维特征向量（每帧） |
| SigLIP 权重 | 完全冻结，不参与训练 |
| Temporal Conv 核大小 | 3（看前后各 1 帧） |
| Attention heads | 8 |

### 数据流

```
视频片段 (60, H, W, 3)
  → 降采样到 20fps
  → resize 到 384×384
  → SigLIP 编码每帧       →  (60, 1152)
  → Temporal Conv         →  (60, 1152)
  → Unmasked Attention    →  (60, 1152)
  → 动作分类头            →  (60, N_classes)
  → 坐标回归头            →  (60, 2)   # 归一化 x, y
  → 全部 60 帧的预测都计算 loss
```

---

## 步骤 0：环境准备

在台式机（RTX 5090，`D:\Programming\u-teach-me`）执行。

### 0.1 安装依赖

```bash
uv add transformers einops
```

> `transformers` 用于加载 SigLIP，`einops` 用于张量维度操作。PyTorch nightly cu128 已有，不需要重装。

### 0.2 下载 SigLIP 权重

第一次运行训练脚本时会自动从 HuggingFace 下载，无需手动操作。默认缓存在：

```
C:\Users\<你的用户名>\.cache\huggingface\
```

> ⚠️ 如果网络访问 HuggingFace 受限，需要挂 VPN 或提前手动下载权重文件。

### 0.3 验证

```python
from transformers import SiglipVisionModel
model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
print(model)  # 能打印出模型结构即成功
```

---

## 步骤 1：重写 processor.py

将现有的「抽单张前后帧」逻辑，改为「切连续时间窗口」逻辑。

### 1.1 核心逻辑变化

| 旧逻辑 | 新逻辑 |
|--------|--------|
| 每个事件抽 t-N 帧和 t+N 帧 | 以事件为锚点，切出前后共 60 帧的连续片段 |
| 每个样本 = 2 张图 | 每个样本 = 60 帧序列 + 60 个动作标签 |
| 固定帧偏移 | 统一重采样到 20fps 后再切窗口 |
| 输出：`frame_before.png`, `frame_after.png` | 输出：`frames/` 目录（60 张）+ `labels.json` |

### 1.2 新数据集格式

每个样本对应一个子目录：

```
dataset/
  sample_0001/
    frames/
      000.png   # 第 0 帧
      001.png
      ...
      059.png   # 第 59 帧
    labels.json
  sample_0002/
    ...
```

`labels.json` 格式：

```json
[
  {"frame": 0,  "action": "none",  "x": null,  "y": null},
  {"frame": 1,  "action": "none",  "x": null,  "y": null},
  ...
  {"frame": 30, "action": "click", "x": 0.512, "y": 0.347},
  ...
  {"frame": 59, "action": "none",  "x": null,  "y": null}
]
```

> `x`, `y` 是归一化坐标（0~1 范围，相对于屏幕宽高）。`action` 为字符串，`none` 表示该帧没有输入事件。

### 1.3 窗口切割规则

- 以每个 pynput 事件的时间戳为中心，向前取 1.5 秒，向后取 1.5 秒
- 如果录屏开头或结尾不足 1.5 秒，该事件跳过（不生成样本）
- 相邻两个事件如果窗口重叠超过 50%，只保留前一个（避免重复数据）
- 所有帧在切窗口之前统一降采样到 20fps

### 1.4 验证方法

运行新 `processor.py` 后，逐项检查：

1. `dataset/` 目录下出现若干 `sample_XXXX/` 子目录
2. 每个子目录里有 60 张 `frames/000.png` ~ `059.png`
3. `labels.json` 里有 60 条记录，中间附近有非 `none` 的动作
4. 随机打开几个样本，用肉眼确认帧序列连续、动作标签时间合理

---

## 步骤 2：重写 train_idm.py

### 2.1 Dataset 类

```python
class IDMDataset(Dataset):
    def __getitem__(self, idx):
        # 读取 60 张帧 → tensor (60, 3, 384, 384)
        # 读取 labels.json → action_labels (60,), coord_labels (60, 2)
        # 返回 frames, action_labels, coord_labels
```

> 帧在这里 resize 到 384×384 并做 SigLIP 标准归一化（mean/std 从 `SiglipImageProcessor` 获取）。

### 2.2 模型结构

```python
class IDMModel(nn.Module):
    def __init__(self):
        # 1. SigLIP 编码器（冻结）
        self.encoder = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
        for p in self.encoder.parameters():
            p.requires_grad = False

        # 2. Temporal Conv（在时间维捕捉局部变化）
        self.temporal_conv = nn.Conv1d(1152, 1152, kernel_size=3, padding=1)

        # 3. Unmasked Attention（全序列长程依赖）
        self.attention = nn.MultiheadAttention(1152, num_heads=8, batch_first=True)

        # 4. 预测头
        self.action_head = nn.Linear(1152, N_CLASSES)
        self.coord_head  = nn.Linear(1152, 2)
```

### 2.3 前向传播

```python
def forward(self, frames):  # frames: (B, 60, 3, 384, 384)
    B, T, C, H, W = frames.shape

    # 展平 batch 和时间维，一次过 SigLIP
    x = frames.view(B * T, C, H, W)
    x = self.encoder(x).pooler_output   # (B*T, 1152)
    x = x.view(B, T, 1152)              # (B, 60, 1152)

    # Temporal Conv（在时间维滑动）
    x = x.transpose(1, 2)               # (B, 1152, 60)
    x = self.temporal_conv(x)
    x = x.transpose(1, 2)               # (B, 60, 1152)

    # Attention（non-causal，全序列可见）
    x, _ = self.attention(x, x, x)      # (B, 60, 1152)

    # 全部 60 帧都预测
    action_logits = self.action_head(x)          # (B, 60, N_CLASSES)
    coords = torch.sigmoid(self.coord_head(x))   # (B, 60, 2)

    return action_logits, coords
```

### 2.4 Loss 计算

只在全部 60 帧上计算 loss：

- **动作分类**：`CrossEntropyLoss`，对 `none` 类降权（权重约为其他类的 0.1）
- **坐标回归**：只在动作类型不是 `none` 的帧上计算 `MSELoss`
- **总 loss** = 分类 loss + `lambda` × 坐标 loss（`lambda` 初始设为 1.0）

### 2.5 验证方法

1. 训练 10 个 epoch，确认 loss 在下降（不要求收敛）
2. `eval_idm.py` 跑验证集，看各动作类型的分类准确率是否超过 50%
3. 坐标预测的平均误差应在 0.1 以内（10% 屏幕宽度）

> ⚠️ 如果所有帧都预测为 `none`，说明 `none` 类权重降得不够，继续降低它的权重。

---

## 步骤 3：扩充训练数据

当前 1552 个样本转换格式后，实际可用的序列样本数量会大幅减少（窗口重叠过滤）。需要补充数据。

### 3.1 自己录更多数据（优先）

用现有 `recorder.py` 多录几段操作视频，覆盖所有动作类型：

| 动作类型 | 录制内容示例 |
|----------|-------------|
| `click` | 点击按钮、链接、菜单 |
| `dblclick` | 双击文件、图标 |
| `scroll` | 在网页、文档里滚动 |
| `drag` | 拖拽文件、调整窗口大小 |
| `key` | 输入文字、Ctrl+C/V 等快捷键 |

**目标：每种动作类型至少 200 个有效窗口样本。**

### 3.2 转换 ScreenAgent 数据集（并行进行）

ScreenAgent 是开源的 GUI 操作数据集，需要写 `convert_screenagent.py` 将其转换为新格式。

转换脚本目标：
- 将 ScreenAgent 的截图序列 + 动作标注 → 新格式的 `sample_XXXX/` 目录
- 动作类型映射到统一标签体系（`click / scroll / key / drag / dblclick / none`）

> 这个脚本留到步骤 1 和 2 验证通过后再写，不影响主流程推进。

---

## 里程碑与验收标准

| 里程碑 | 验收标准 |
|--------|----------|
| 步骤 0 完成 | `uv add` 成功，`import transformers` 不报错 |
| 步骤 1 完成 | processor.py 生成格式正确的数据集，肉眼抽查 10 个样本无误 |
| 步骤 2 基础完成 | 模型可以前向传播不报错，loss 能反向传播 |
| 步骤 2 训练验证 | 各动作类型分类准确率 > 50%，坐标误差 < 0.1 |
| 数据扩充完成 | 每类动作 > 200 个有效窗口样本 |
| 第二阶段完成 | `eval_idm.py` 整体分类准确率 > 70% |

---

## 常见问题

### 显存不够

SigLIP 编码 60 帧 × batch size 是显存大户。如果 RTX 5090 32GB 仍然不够：

- 先降 batch size 到 2 或 1
- 用 `torch.utils.checkpoint` 对 SigLIP 编码做梯度检查点（节省显存但略慢）
- SigLIP 已冻结，可以预先把所有帧的特征提取出来存到磁盘，训练时直接读特征而不过编码器（推荐，大幅提速）

### HuggingFace 下载慢或失败

```python
# 在能访问 HuggingFace 的网络环境下运行一次
from transformers import SiglipVisionModel
SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")
# 下载完成后缓存在本地，后续离线可用
```

### none 类不平衡

GUI 录屏里大部分帧没有动作，`none` 类样本远多于其他类。已在 loss 里降权处理，如果效果不好：

- 每个窗口至少有 1 个非 `none` 动作才保留（processor.py 应已有此逻辑）
- 适当提高事件周围帧的采样密度

### PyTorch nightly 兼容性

RTX 5090（sm_120）必须使用 PyTorch nightly cu128，stable 版不支持。确认方法：

```bash
python -c "import torch; print(torch.__version__)"
# 版本号应包含 dev 字样，如 2.x.0.dev20250xxx+cu128
```

---

## 文件清单

| 文件 | 状态 |
|------|------|
| `recorder.py` | ✅ 已完成，无需修改 |
| `processor.py` | 🔧 需重写（步骤 1） |
| `train_idm.py` | 🔧 需重写（步骤 2） |
| `eval_idm.py` | 🔧 需小幅修改以适配新输出格式 |
| `convert_screenagent.py` | 📋 待创建（步骤 3，暂缓） |
| `dataset/sample_XXXX/` | 📋 新数据集格式（步骤 1 生成） |
