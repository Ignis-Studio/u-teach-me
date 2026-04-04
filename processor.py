import os, json, subprocess, statistics, shutil
from pathlib import Path
from PIL import Image, ImageChops

FPS = 30
IMG_SIZE = 224
DIFF_THRESHOLD = 0.15  # 像素差均值低于此值则跳过（画面没变化）


def extract_frame(video_path, timestamp, output_path):
    """用 ffmpeg 精确抽取某时间点的帧，resize 到 224x224"""
    cmd = [
        'ffmpeg', '-y',
        '-ss', f'{timestamp:.4f}',
        '-i', str(video_path),
        '-vframes', '1',
        '-vf', f'scale={IMG_SIZE}:{IMG_SIZE}',
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(result.stderr.decode('utf-8', errors='ignore')[-300:])
    return result.returncode == 0


def frames_are_similar(p1, p2):
    """判断两帧是否几乎相同（没有有效操作发生）"""
    img1 = Image.open(p1).convert('RGB')
    img2 = Image.open(p2).convert('RGB')
    diff = ImageChops.difference(img1, img2)
    avg = statistics.mean(sum(px) / 3 for px in diff.getdata())
    print(f'    像素差: {avg:.4f}')
    return avg < DIFF_THRESHOLD


def process_one(video_path, json_path, output_dir):
    output_dir = Path(output_dir)

    # 已处理过则跳过
    if output_dir.exists() and any(output_dir.iterdir()):
        existing = len(list(output_dir.iterdir()))
        print(f'  已处理过（{existing} 个样本），跳过。如需重新处理请手动删除 {output_dir}')
        return existing

    with open(json_path) as f:
        data = json.load(f)

    events = data['events']
    fps = data.get('fps', FPS)
    os.makedirs(output_dir, exist_ok=True)
    saved, skipped = 0, 0

    for i, event in enumerate(events):
        t = event['t']
        t_next = t + 1.0 / fps

        sample_dir = output_dir / f'sample_{i:04d}'
        sample_dir.mkdir(exist_ok=True)

        ft  = sample_dir / 'frame_t.png'
        ft1 = sample_dir / 'frame_t1.png'

        ok1 = extract_frame(video_path, t, ft)
        ok2 = extract_frame(video_path, t_next, ft1)

        if not ok1 or not ok2:
            print(f'  跳过 sample_{i:04d}：ffmpeg 抽帧失败')
            shutil.rmtree(sample_dir, ignore_errors=True)
            skipped += 1
            continue

        if not ft.exists() or not ft1.exists():
            shutil.rmtree(sample_dir, ignore_errors=True)
            skipped += 1
            continue

        if frames_are_similar(ft, ft1):
            shutil.rmtree(sample_dir)
            skipped += 1
            continue

        with open(sample_dir / 'action.json', 'w') as f:
            json.dump(event, f, indent=2)

        saved += 1

    print(f'  生成 {saved} 个样本，跳过 {skipped} 个')
    return saved


def check_dataset(dataset_dir):
    """统计数据集质量"""
    samples = list(Path(dataset_dir).rglob('action.json'))
    if not samples:
        print('没有找到任何样本，请检查 dataset/ 目录')
        return

    from collections import Counter
    types = Counter(json.load(open(s))['type'] for s in samples)
    print(f'\n数据集统计：')
    print(f'  总样本数：{len(samples)}')
    print(f'  各类型分布：{dict(types)}')
    if len(samples) < 100:
        print('  ⚠️  样本数量偏少（建议 300+），考虑补录几段')
    missing = [t for t in ['click', 'move', 'key'] if types.get(t, 0) == 0]
    if missing:
        print(f'  ⚠️  缺少操作类型：{missing}，请补录对应段')
    else:
        print('  ✓  各操作类型均有样本')


# 主流程
recordings = Path('recordings')
dataset    = Path('dataset')
total = 0

for jf in sorted(recordings.glob('*.json')):
    vf = jf.with_suffix('.mp4')
    if not vf.exists():
        print(f'跳过 {jf.stem}：找不到对应 .mp4')
        continue
    print(f'处理 {jf.stem}...')
    total += process_one(vf, jf, dataset / jf.stem)

print(f'\n全部完成，共生成 {total} 个样本')
check_dataset(dataset)
