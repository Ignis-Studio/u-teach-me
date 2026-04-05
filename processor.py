import os, json, subprocess, shutil, concurrent.futures
from pathlib import Path

FPS = 30
IMG_SIZE = 672

# 文件名前缀 → 只保留的事件类型
# 注意：长前缀必须排在短前缀前面，避免 click 误匹配 dblclick
PREFIX_TO_TYPE = [
    ('dblclick', 'dblclick'),
    ('click',    'click'),
    ('drag',     'move'),
    ('key',      'key'),
    ('scroll',   'scroll'),
]

def convert_to_420p(recordings_dir):
    recordings_dir = Path(recordings_dir)
    for vf in sorted(recordings_dir.glob('*.mp4')):
        tmp = vf.with_suffix('.tmp.mp4')
        print(f'转换 {vf.name} -> yuv420p...')
        result = subprocess.run([
            'ffmpeg', '-y',
            '-i', str(vf),
            '-vf', 'format=yuv420p',
            '-c:v', 'h264_nvenc',
            str(tmp)
        ], capture_output=True)
        if result.returncode == 0:
            tmp.replace(vf)  # 覆盖原文件
            print(f'  ✓ {vf.name}')
        else:
            tmp.unlink(missing_ok=True)
            print(f'  ✗ {vf.name} 转换失败：')
            print(result.stderr.decode('utf-8', errors='ignore')[-300:])

def get_expected_type(stem):
    """根据文件名前缀判断应该保留的事件类型，长前缀优先匹配"""
    for prefix, event_type in PREFIX_TO_TYPE:
        if stem.startswith(prefix + '_'):
            return prefix, event_type
    return None, None


def extract_frame(video_path, timestamp, output_path):
    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-hwaccel_output_format', 'cuda',
        '-ss', f'{timestamp:.4f}',
        '-i', str(video_path),
        '-vframes', '1',
        '-vf', f'scale_cuda={IMG_SIZE}:{IMG_SIZE},hwdownload,format=nv12',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        # 降级到CPU
        cmd_cpu = [
            'ffmpeg', '-y',
            '-ss', f'{timestamp:.4f}',
            '-i', str(video_path),
            '-vframes', '1',
            '-vf', f'scale={IMG_SIZE}:{IMG_SIZE}',
            str(output_path)
        ]
        result = subprocess.run(cmd_cpu, capture_output=True)
        if result.returncode != 0:
            print(result.stderr.decode('utf-8', errors='ignore')[-300:])

    return result.returncode == 0

def find_dblclicks(events, threshold=0.3):
    """
    把时间间隔小于 threshold 秒的相邻两次 click 合并为一个 dblclick 事件。
    返回合并后的事件列表，跳过被合并掉的第二次 click。
    """
    result = []
    i = 0
    while i < len(events):
        e = events[i]
        if (e.get('type') == 'click'
                and i + 1 < len(events)
                and events[i+1].get('type') == 'click'
                and events[i+1]['t'] - e['t'] < threshold):
            # 合并为 dblclick，取第一次点击的坐标和时间
            result.append({
                't': e['t'],
                'type': 'dblclick',
                'x': e['x'],
                'y': e['y'],
                'button': e.get('button', ''),
            })
            i += 2  # 跳过第二次 click
        else:
            i += 1
    return result


def process_one(video_path, json_path, output_dir):
    output_dir = Path(output_dir)
    stem = json_path.stem

    category, expected_type = get_expected_type(stem)
    if expected_type is None:
        print(f'  无法识别文件名前缀：{stem}，跳过')
        return 0

    if output_dir.exists() and any(output_dir.iterdir()):
        existing = len(list(output_dir.iterdir()))
        print(f'  已处理过（{existing} 个样本），跳过')
        return existing

    with open(json_path) as f:
        data = json.load(f)

    raw_events = data['events']
    fps = data.get('fps', FPS)

    if category == 'dblclick':
        events = find_dblclicks(raw_events)
    else:
        events = [e for e in raw_events if e.get('type') == expected_type]

    os.makedirs(output_dir, exist_ok=True)
    saved, skipped = 0, 0

    def process_event(args):
        i, event = args
        t = event['t']
        t_before = max(0, t - 3.0 / fps)
        t_next = t + 3.0 / fps

        sample_dir = output_dir / f'sample_{i:04d}'
        sample_dir.mkdir(exist_ok=True)

        ft = sample_dir / 'frame_t.png'
        ft1 = sample_dir / 'frame_t1.png'

        ok1 = extract_frame(video_path, t_before, ft)
        ok2 = extract_frame(video_path, t_next, ft1)

        if not ok1 or not ok2 or not ft.exists() or not ft1.exists():
            shutil.rmtree(sample_dir, ignore_errors=True)
            return i, False, event

        with open(sample_dir / 'action.json', 'w') as f:
            json.dump(event, f, indent=2)
        return i, True, event

    # 并发数别开太高，GPU解码有并发上限，8~16一般够
    MAX_WORKERS = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_event, enumerate(events)))

    for i, ok, event in results:
        if ok:
            saved += 1
        else:
            print(f'  跳过 sample_{i:04d}：ffmpeg 抽帧失败')
            skipped += 1

    print(f'  类型={expected_type}，生成 {saved} 个样本，跳过 {skipped} 个')
    return saved


def check_dataset(dataset_dir):
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
    missing = [t for t in ['click', 'dblclick', 'key', 'scroll'] if types.get(t, 0) == 0]
    if missing:
        print(f'  ⚠️  缺少操作类型：{missing}，请补录对应段')
    else:
        print('  ✓  各操作类型均有样本')


# 主流程
recordings = Path('recordings')
dataset    = Path('dataset')
total = 0

convert_to_420p(recordings)

for jf in sorted(recordings.glob('*.json')):
    vf = jf.with_suffix('.mp4')
    if not vf.exists():
        print(f'跳过 {jf.stem}：找不到对应 .mp4')
        continue
    print(f'处理 {jf.stem}...')
    total += process_one(vf, jf, dataset / jf.stem)

print(f'\n全部完成，共生成 {total} 个样本')
check_dataset(dataset)
