import subprocess
import threading
import time
import json
import re
import platform
import os
from pathlib import Path
from pynput import mouse, keyboard as kb
from pynput.keyboard import HotKey

if platform.system() == 'Windows':
    import winsound

RECORDINGS_DIR = Path('recordings')
RECORDINGS_DIR.mkdir(exist_ok=True)
FPS = 30

def beep_start():
    if platform.system() == 'Windows':
        winsound.Beep(880, 200)
    else:
        print('\a', end='', flush=True)

def beep_stop():
    if platform.system() == 'Windows':
        winsound.Beep(660, 150)
        time.sleep(0.1)
        winsound.Beep(440, 200)
    else:
        print('\a\a', end='', flush=True)


class Recorder:
    def __init__(self):
        self.events = []
        self.start_time = None
        self.ffmpeg_proc = None
        self.mouse_listener = None
        self.running = False

    def on_click(self, x, y, button, pressed):
        if not pressed or not self.running:
            return
        self.events.append({
            't': time.time() - self.start_time,
            'type': 'click',
            'x': x, 'y': y,
            'button': str(button),
        })

    def on_move(self, x, y):
        if not self.running:
            return
        now = time.time() - self.start_time
        if self.events and self.events[-1]['type'] == 'move':
            if now - self.events[-1]['t'] < 0.1:
                return
        self.events.append({'t': now, 'type': 'move', 'x': x, 'y': y})

    def on_scroll(self, x, y, dx, dy):
        if not self.running:
            return
        self.events.append({
            't': time.time() - self.start_time,
            'type': 'scroll',
            'x': x, 'y': y,
            'dx': dx, 'dy': dy,
        })

    def on_key(self, key):
        if not self.running:
            return
        try:
            k = key.char
        except AttributeError:
            k = str(key)
        self.events.append({
            't': time.time() - self.start_time,
            'type': 'key', 'key': k,
        })

    def start(self, name):
        self.events = []
        self.running = True

        video_path = RECORDINGS_DIR / f'{name}.mp4'

        system = platform.system()
        input_args = []
        if system == 'Windows':
            input_args = ['-f', 'gdigrab', '-i', 'desktop']
        elif system == 'Linux':
            display = os.environ.get('DISPLAY', ':0.0')
            input_args = ['-f', 'x11grab', '-i', display]
        elif system == 'Darwin':
            input_args = ['-f', 'avfoundation', '-i', '1']
        else:
            print(f'不支持的操作系统: {system}')
            return

        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(FPS),
        ] + input_args + [
            '-vf', 'scale=1920:1088',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            str(video_path)
        ]

        self.ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0 / FPS)
        self.start_time = time.time()

        self.mouse_listener = mouse.Listener(
            on_click=self.on_click,
            on_move=self.on_move,
            on_scroll=self.on_scroll,
        )
        self.mouse_listener.start()

    def stop(self, name):
        self.running = False

        if self.mouse_listener:
            self.mouse_listener.stop()

        if self.ffmpeg_proc:
            try:
                self.ffmpeg_proc.stdin.write(b'q')
                self.ffmpeg_proc.stdin.flush()
                self.ffmpeg_proc.wait(timeout=5)
            except Exception:
                self.ffmpeg_proc.kill()

        json_path = RECORDINGS_DIR / f'{name}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'start_time': self.start_time,
                'fps': FPS,
                'events': self.events,
            }, f, indent=2)

        return len(self.events)


class App:
    def __init__(self):
        self.recorder = Recorder()
        self.is_recording = False
        self.current_name = self._ask_name()

    def _ask_name(self):
        print('=' * 40)
        print('IDM 录制工具 (跨平台版)')
        print('Ctrl+Shift+R  开始录制')
        print('Ctrl+Shift+S  停止录制')
        print('=' * 40)
        name = input('录制名称（直接回车使用 click_001）：').strip()
        return name if name else 'click_001'

    def _on_start(self):
        if self.is_recording:
            return
        self.is_recording = True
        threading.Thread(target=self._do_start, daemon=True).start()

    def _do_start(self):
        print(f'[开始] {self.current_name}')
        self.recorder.start(self.current_name)
        beep_start()

    def _on_stop(self):
        if not self.is_recording:
            return
        self.is_recording = False
        threading.Thread(target=self._do_stop, daemon=True).start()

    def _do_stop(self):
        count = self.recorder.stop(self.current_name)
        beep_stop()
        print(f'[保存] {self.current_name}.mp4 / .json（{count} 个事件）')
        self.current_name = self._increment(self.current_name)
        print(f'[就绪] 下一段：{self.current_name}')

    def _increment(self, name):
        m = re.match(r'^(.*?)(\d+)$', name)
        if m:
            prefix, num = m.group(1), m.group(2)
            return f'{prefix}{str(int(num)+1).zfill(len(num))}'
        return name + '_2'

    def run(self):
        hotkey_start = HotKey(HotKey.parse('<ctrl>+<shift>+r'), self._on_start)
        hotkey_stop  = HotKey(HotKey.parse('<ctrl>+<shift>+s'), self._on_stop)

        def on_press(key):
            self.recorder.on_key(key)
            hotkey_start.press(listener.canonical(key))
            hotkey_stop.press(listener.canonical(key))

        def on_release(key):
            hotkey_start.release(listener.canonical(key))
            hotkey_stop.release(listener.canonical(key))

        print('等待快捷键，Ctrl+C 退出...')
        with kb.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


if __name__ == '__main__':
    app = App()
    app.run()
