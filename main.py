import tkinter as tk
from tkinter import ttk, messagebox
import subprocess, threading, time, json, os, re
from pathlib import Path
from pynput import mouse, keyboard
import platform 
import signal

RECORDINGS_DIR = Path('recordings')
RECORDINGS_DIR.mkdir(exist_ok=True)
FPS = 30


class Recorder:
    def __init__(self):
        self.events = []
        self.start_time = None
        self.ffmpeg_proc = None
        self.running = False

        # 【修复1】在初始化时启动一次监听器，永远不调用 .stop()
        # 依靠 self.running 标志位来控制是否真正记录数据，完美避开 Linux 下的死锁问题
        self.mouse_listener = mouse.Listener(
            on_click=self.on_click,
            on_move=self.on_move,
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_press,
        )
        self.mouse_listener.start()
        self.keyboard_listener.start()

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

    def on_press(self, key):
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

    def start(self, name, status_callback):
        self.events = []
        self.running = True

        video_path = RECORDINGS_DIR / f'{name}.mp4'

        # 【修复2】跨平台 FFmpeg 参数
        sys_name = platform.system()
        if sys_name == 'Windows':
            input_opts = ['-f', 'gdigrab', '-i', 'desktop']
        elif sys_name == 'Linux':
            # Linux 通常使用 x11grab 获取屏幕
            display = os.environ.get('DISPLAY', ':0.0')
            input_opts = ['-f', 'x11grab', '-i', display]
        elif sys_name == 'Darwin':  # macOS 支持
            input_opts = ['-f', 'avfoundation', '-i', '1']
        else:
            input_opts = ['-f', 'x11grab', '-i', ':0.0']

        cmd = [
            'ffmpeg', '-y',
            *input_opts,
            '-framerate', str(FPS),
            '-c:v', 'libx264',
            '-vf', 'scale=1920:1080',
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

        def update_status():
            while self.running:
                status_callback(f'录制中... 已记录 {len(self.events)} 个事件')
                time.sleep(0.5)

        threading.Thread(target=update_status, daemon=True).start()

    def stop(self, name):
        self.running = False  # 停止记录事件

        # 【修复3】跨平台优雅结束 FFmpeg
        if self.ffmpeg_proc:
            try:
                if platform.system() == 'Windows':
                    self.ffmpeg_proc.stdin.write(b'q')
                    self.ffmpeg_proc.stdin.flush()
                else:
                    # Linux/macOS 下发送 SIGINT (Ctrl+C) 是最稳妥的停止方式
                    self.ffmpeg_proc.send_signal(signal.SIGINT)

                self.ffmpeg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_proc.kill()
            except Exception:
                self.ffmpeg_proc.kill()

        json_path = RECORDINGS_DIR / f'{name}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'start_time': self.start_time,
                'fps': FPS,
                'events': self.events,
            }, f, indent=2)

        return len(self.events)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title('IDM 数据录制工具')
        self.root.resizable(False, False)
        self.root.attributes('-topmost', True)  # 窗口始终置顶
        self.recorder = Recorder()
        self.is_recording = False
        self._build_ui()

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.grid()

        # 录制名称
        ttk.Label(frame, text='录制名称：').grid(row=0, column=0, sticky='w', pady=5)
        self.name_var = tk.StringVar(value='click_001')
        self.name_entry = ttk.Entry(frame, textvariable=self.name_var, width=20)
        self.name_entry.grid(row=0, column=1, columnspan=2, sticky='ew', pady=5)

        # 按钮
        self.start_btn = ttk.Button(
            frame, text='⏺  开始录制', command=self.start_recording, width=18)
        self.start_btn.grid(row=1, column=0, columnspan=2, pady=8, padx=(0, 5))

        self.stop_btn = ttk.Button(
            frame, text='⏹  停止录制', command=self.stop_recording,
            width=18, state='disabled')
        self.stop_btn.grid(row=1, column=2, pady=8)

        # 状态栏
        self.status_var = tk.StringVar(value='就绪')
        ttk.Label(
            frame, textvariable=self.status_var,
            foreground='gray', font=('Arial', 9)
        ).grid(row=2, column=0, columnspan=3, pady=(5, 0))

    def start_recording(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning('提示', '请先输入录制名称')
            return

        if (Path('recordings') / f'{name}.mp4').exists():
            if not messagebox.askyesno('文件已存在', f'{name}.mp4 已存在，覆盖吗？'):
                return

        self.is_recording = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.name_entry.config(state='disabled')

        def status_cb(msg):
            self.root.after(0, lambda: self.status_var.set(msg))

        threading.Thread(
            target=self.recorder.start,
            args=(name, status_cb),
            daemon=True
        ).start()

    def stop_recording(self):
        if not self.is_recording:
            return

        self.stop_btn.config(state='disabled')
        self.status_var.set('正在保存...')
        name = self.name_var.get().strip()

        def do_stop():
            count = self.recorder.stop(name)
            def finish():
                self.is_recording = False
                self.start_btn.config(state='normal')
                self.name_entry.config(state='normal')
                self.status_var.set(f'已保存 {name}.mp4 / {name}.json（{count} 个事件）')
                self._auto_increment_name()
            self.root.after(0, finish)

        threading.Thread(target=do_stop, daemon=True).start()

    def _auto_increment_name(self):
        """自动将名称末尾数字 +1，比如 click_001 → click_002"""
        name = self.name_var.get().strip()
        m = re.match(r'^(.*?)(\d+)$', name)
        if m:
            prefix, num = m.group(1), m.group(2)
            self.name_var.set(f'{prefix}{str(int(num)+1).zfill(len(num))}')


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
