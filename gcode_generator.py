"""
GCode生成器 & GRBL控制器
功能：图片转GCode、串口连接设备、手动控制、实时发送GCode
类似LaserGRBL的完整控制器
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageFilter, ImageOps
import numpy as np
import cv2
import os
import sys
import subprocess
import tempfile
import threading
import queue
import time
import re
import xml.etree.ElementTree as ET

# 尝试导入串口库
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class PaperSizes:
    """纸张规格定义 (单位: mm)"""
    SIZES = {
        'A4': (210, 297),
        'A3': (297, 420),
        'A5': (148, 210),
        'A2': (420, 594),
        'A1': (594, 841),
        'Letter': (216, 279),
        'Legal': (216, 356),
        '自定义': (200, 200)
    }


class GRBLController:
    """GRBL串口通信控制器"""
    
    def __init__(self):
        self.serial_port = None
        self.connected = False
        self.response_queue = queue.Queue()
        self.status = "Idle"
        self.position = {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        self.read_thread = None
        self.running = False
        self.streaming = False
        self.stream_progress = 0
        self.stream_total = 0
        self.callbacks = {
            'status': None,
            'response': None,
            'progress': None,
            'position': None
        }
    
    @staticmethod
    def list_ports():
        """列出可用串口"""
        if not SERIAL_AVAILABLE:
            return []
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def connect(self, port, baudrate=115200):
        """连接到GRBL设备"""
        if not SERIAL_AVAILABLE:
            raise Exception("未安装pyserial库，请运行: pip install pyserial")
        
        try:
            self.serial_port = serial.Serial(port, baudrate, timeout=0.1)
            time.sleep(2)  # 等待GRBL初始化
            self.serial_port.flushInput()
            self.connected = True
            self.running = True
            
            # 启动读取线程
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            
            # 发送状态查询
            self.send_command("?")
            
            return True
        except Exception as e:
            self.connected = False
            raise e
    
    def disconnect(self):
        """断开连接"""
        self.running = False
        self.streaming = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.connected = False
        self.status = "Disconnected"
    
    def _read_loop(self):
        """后台读取串口数据"""
        while self.running and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting:
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self._process_response(line)
            except Exception as e:
                if self.running:
                    self._callback('response', f"读取错误: {e}")
            time.sleep(0.01)
    
    def _process_response(self, line):
        """处理GRBL响应"""
        # 状态报告 <Idle|MPos:0.000,0.000,0.000|...>
        if line.startswith('<') and line.endswith('>'):
            self._parse_status(line)
        elif line == 'ok':
            self.response_queue.put('ok')
            # 不显示 ok 响应，避免刷屏
        elif line.startswith('error'):
            self.response_queue.put(line)
            self._callback('response', f"[错误] {line}")
        elif line.startswith('ALARM'):
            self.status = "Alarm"
            self._callback('status', self.status)
            self._callback('response', f"[警报] {line}")
        elif line.startswith('Grbl'):
            self._callback('response', f"[GRBL] {line}")
        elif line.startswith('$') and '=' in line:
            # 解析设置值 $0=2
            self._parse_setting(line)
        else:
            self._callback('response', line)
    
    def _parse_setting(self, line):
        """解析设置行 $0=2"""
        try:
            match = re.match(r'\$(\d+)=(.+)', line)
            if match:
                param = int(match.group(1))
                value = match.group(2).strip()
                self._callback('setting', (param, value))
        except:
            pass
    
    def _parse_status(self, line):
        """解析状态报告"""
        # <Idle|MPos:0.000,0.000,0.000|FS:0,0>
        match = re.match(r'<(\w+)\|MPos:([^|>]+)', line)
        if match:
            self.status = match.group(1)
            pos_str = match.group(2)
            coords = pos_str.split(',')
            if len(coords) >= 3:
                self.position = {
                    'X': float(coords[0]),
                    'Y': float(coords[1]),
                    'Z': float(coords[2])
                }
            self._callback('status', self.status)
            self._callback('position', self.position)
    
    def _callback(self, event, data):
        """触发回调"""
        if self.callbacks.get(event):
            try:
                self.callbacks[event](data)
            except:
                pass
    
    def send_command(self, cmd, silent=False):
        """发送单条命令"""
        if not self.connected or not self.serial_port:
            return False
        try:
            # 清理命令，只保留有效ASCII字符
            cmd = cmd.strip()
            cmd = ''.join(c for c in cmd if ord(c) < 128)
            cmd = cmd.replace('\r', '').replace('\n', '')
            
            self.serial_port.write((cmd + '\n').encode('ascii', errors='ignore'))
            if not silent:
                self._callback('response', f"> {cmd}")
            return True
        except Exception as e:
            if not silent:
                self._callback('response', f"发送失败: {e}")
            return False
    
    def send_command_wait(self, cmd, timeout=10):
        """发送命令并等待响应"""
        # 清空队列
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except:
                break
        
        if not self.send_command(cmd):
            return False, "发送失败"
        
        try:
            response = self.response_queue.get(timeout=timeout)
            return response == 'ok', response
        except queue.Empty:
            return False, "超时"
    
    def stream_gcode(self, gcode_lines, progress_callback=None, on_complete=None):
        """流式发送GCode - 使用字符计数协议，保持缓冲区一半空闲"""
        if not self.connected:
            return False
        
        self.streaming = True
        self.stream_total = len(gcode_lines)
        
        # GRBL缓冲区128字节，保持一半空闲
        RX_BUFFER_SIZE = 128
        BUFFER_THRESHOLD = RX_BUFFER_SIZE // 2  # 64字节
        
        def stream_thread():
            buffer_used = 0  # 当前缓冲区已用字节
            pending_cmds = []  # 待确认的命令及其字节数
            
            for i, line in enumerate(gcode_lines):
                if not self.streaming:
                    break
                
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                
                # 命令字节数 (含\n)
                cmd_len = len(line) + 1
                
                # 等待缓冲区有足够空间
                while buffer_used + cmd_len > BUFFER_THRESHOLD and self.streaming:
                    try:
                        response = self.response_queue.get(timeout=0.1)
                        if pending_cmds:
                            freed = pending_cmds.pop(0)
                            buffer_used -= freed
                    except queue.Empty:
                        continue
                
                if not self.streaming:
                    break
                
                # 发送命令
                if self.send_command(line):
                    buffer_used += cmd_len
                    pending_cmds.append(cmd_len)
                else:
                    self._callback('response', f"发送失败: {line}")
                
                self.stream_progress += 1
                self._callback('progress', (self.stream_progress, self.stream_total))
            
            # 等待所有待确认命令完成
            while pending_cmds and self.streaming:
                try:
                    self.response_queue.get(timeout=1.0)
                    pending_cmds.pop(0)
                except queue.Empty:
                    break
            
            completed = self.streaming
            self.streaming = False
            
            if completed:
                self._callback('response', "=== GCode发送完成 ===")
                if on_complete:
                    on_complete()
        
        thread = threading.Thread(target=stream_thread, daemon=True)
        thread.start()
        return True
    
    def stop_stream(self):
        """停止流式发送"""
        self.streaming = False
    
    def emergency_stop(self):
        """急停 (软复位)"""
        if self.serial_port and self.serial_port.is_open:
            self.streaming = False
            self.serial_port.write(b'\x18')  # Ctrl+X
            self._callback('response', "[急停] 发送软复位")
    
    def unlock(self):
        """解锁"""
        self.send_command("$X")
    
    def home(self):
        """归位"""
        self.send_command("$H")
    
    def jog(self, axis, distance, feed_rate=1000):
        """点动控制"""
        # 先取消之前的jog命令
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.write(b'\x85')  # Jog Cancel
        cmd = f"$J=G91 {axis}{distance} F{feed_rate}"
        self.send_command(cmd)
    
    def set_zero(self, axes='XYZ'):
        """设置当前位置为零点"""
        cmd = f"G92 {' '.join([f'{a}0' for a in axes])}"
        self.send_command(cmd)
    
    def query_status(self):
        """查询状态（静默，不显示在控制台）"""
        if self.connected:
            self.send_command("?", silent=True)
    
    def get_settings(self):
        """获取GRBL设置"""
        if self.connected:
            self.send_command("$$")
    
    def set_setting(self, param, value):
        """设置单个参数"""
        if self.connected:
            self.send_command(f"${param}={value}")


# GRBL参数定义
GRBL_SETTINGS = {
    0: ('步进脉冲时间', '微秒', 'Step pulse time'),
    1: ('步进空闲延迟', '毫秒', 'Step idle delay'),
    2: ('步进脉冲反转', '掩码', 'Step pulse invert'),
    3: ('步进方向反转', '掩码', 'Step direction invert'),
    4: ('使能引脚反转', '布尔', 'Invert step enable pin'),
    5: ('限位开关反转', '布尔', 'Invert limit pins'),
    6: ('探针引脚反转', '布尔', 'Invert probe pin'),
    10: ('状态报告选项', '掩码', 'Status report options'),
    11: ('弧线偏差', '毫米', 'Junction deviation'),
    12: ('弧线容差', '毫米', 'Arc tolerance'),
    13: ('英寸报告', '布尔', 'Report in inches'),
    20: ('软限位启用', '布尔', 'Soft limits enable'),
    21: ('硬限位启用', '布尔', 'Hard limits enable'),
    22: ('归位循环启用', '布尔', 'Homing cycle enable'),
    23: ('归位方向反转', '掩码', 'Homing direction invert'),
    24: ('归位定位速率', 'mm/min', 'Homing locate feed rate'),
    25: ('归位搜索速率', 'mm/min', 'Homing search seek rate'),
    26: ('归位开关消抖延迟', '毫秒', 'Homing switch debounce'),
    27: ('归位开关拉开距离', '毫米', 'Homing switch pull-off'),
    30: ('最大主轴转速', 'RPM', 'Maximum spindle speed'),
    31: ('最小主轴转速', 'RPM', 'Minimum spindle speed'),
    32: ('激光模式启用', '布尔', 'Laser-mode enable'),
    100: ('X轴分辨率', 'step/mm', 'X-axis resolution'),
    101: ('Y轴分辨率', 'step/mm', 'Y-axis resolution'),
    102: ('Z轴分辨率', 'step/mm', 'Z-axis resolution'),
    110: ('X轴最大速率', 'mm/min', 'X-axis max rate'),
    111: ('Y轴最大速率', 'mm/min', 'Y-axis max rate'),
    112: ('Z轴最大速率', 'mm/min', 'Z-axis max rate'),
    120: ('X轴加速度', 'mm/sec²', 'X-axis acceleration'),
    121: ('Y轴加速度', 'mm/sec²', 'Y-axis acceleration'),
    122: ('Z轴加速度', 'mm/sec²', 'Z-axis acceleration'),
    130: ('X轴最大行程', '毫米', 'X-axis max travel'),
    131: ('Y轴最大行程', '毫米', 'Y-axis max travel'),
    132: ('Z轴最大行程', '毫米', 'Z-axis max travel'),
}


class GCodeGenerator:
    """
    GCode生成器核心类
    支持玻璃板高度补偿：通过五点插值计算任意位置的Z高度
    """
    
    def __init__(self, feed_rate=1000, z_up=5, z_down=0, z_compensation=None, paper_width=210, paper_height=297):
        self.feed_rate = feed_rate
        self.z_up = z_up
        self.z_down = z_down
        self.paper_width = paper_width
        self.paper_height = paper_height
        
        # 高度补偿参数: {'enabled': bool, 'tl': float, 'tr': float, 'bl': float, 'br': float, 'center': float}
        self.z_compensation = z_compensation or {'enabled': False}
        
    def calculate_z_at_position(self, x, y):
        """
        计算指定位置的Z高度（考虑玻璃板形变）
        支持曲率控制：0=线性, 0.5=平缓, 1=陡峭
        """
        if not self.z_compensation.get('enabled', False):
            return self.z_down
        
        # 获取三点高度
        z_left = self.z_compensation.get('left', self.z_down)    # 左侧 (x=0)
        z_center = self.z_compensation.get('center', self.z_down) # 中间 (x=0.5)
        z_right = self.z_compensation.get('right', self.z_down)  # 右侧 (x=1)
        curve = self.z_compensation.get('curve', 0.5)            # 曲率 0-1
        
        # 归一化X坐标 (0-1)
        nx = max(0, min(1, x / self.paper_width)) if self.paper_width > 0 else 0.5
        
        # 线性插值
        if nx <= 0.5:
            t = nx * 2
            z_linear = z_left * (1 - t) + z_center * t
        else:
            t = (nx - 0.5) * 2
            z_linear = z_center * (1 - t) + z_right * t
        
        # 招物线插值（拉格朗日二次插值）
        L0 = 2 * (nx - 0.5) * (nx - 1)
        L1 = -4 * nx * (nx - 1)
        L2 = 2 * nx * (nx - 0.5)
        z_parabola = z_left * L0 + z_center * L1 + z_right * L2
        
        # 根据曲率混合：0=线性, 1=招物线
        z_final = z_linear * (1 - curve) + z_parabola * curve
        
        return z_final
        
    def generate_header(self):
        return [
            "; GCode generated by GRBL Controller",
            "G21",           # 毫米模式
            "G90",           # 绝对坐标
            "G94",           # 每分钟进给模式
            "G17",           # XY平面
            f"G0 Z{int(self.z_up)}",
            "G0 X0 Y0",
        ]
    
    def generate_footer(self):
        return [
            f"G0 Z{int(self.z_up)}",
            "G0 X0 Y0",
            "M2"
        ]
    
    def pen_up(self):
        return f"G0 Z{int(self.z_up)}"
    
    def pen_down(self, x=None, y=None):
        """落笔，如果启用高度补偿则根据位置计算Z高度"""
        if x is not None and y is not None:
            z = self.calculate_z_at_position(x, y)
            z_str = f"{z:.2f}".rstrip('0').rstrip('.')
        else:
            z_str = str(int(self.z_down))
        return f"G1 Z{z_str} F{int(self.feed_rate)}"
    
    def move_to(self, x, y, rapid=True):
        # 简化数值格式，最多1位小数
        x_str = f"{x:.1f}".rstrip('0').rstrip('.')
        y_str = f"{y:.1f}".rstrip('0').rstrip('.')
        if rapid:
            return f"G0 X{x_str} Y{y_str}"
        else:
            return f"G1 X{x_str} Y{y_str} F{int(self.feed_rate)}"
    
    def move_to_with_z(self, x, y, rapid=False):
        """移动到指定位置，如果启用高度补偿则同时调整Z"""
        x_str = f"{x:.1f}".rstrip('0').rstrip('.')
        y_str = f"{y:.1f}".rstrip('0').rstrip('.')
        
        if self.z_compensation.get('enabled', False):
            z = self.calculate_z_at_position(x, y)
            z_str = f"{z:.2f}".rstrip('0').rstrip('.')
            return f"G1 X{x_str} Y{y_str} Z{z_str} F{int(self.feed_rate)}"
        else:
            if rapid:
                return f"G0 X{x_str} Y{y_str}"
            else:
                return f"G1 X{x_str} Y{y_str} F{int(self.feed_rate)}"
    
    def generate_from_contours(self, contours, offset_x=0, offset_y=0, scale=1.0, paper_height=297):
        gcode = self.generate_header()
        
        # 更新纸张尺寸用于高度补偿计算
        self.paper_height = paper_height
        
        z_comp_enabled = self.z_compensation.get('enabled', False)
        
        for contour in contours:
            if len(contour) < 2:
                continue
            
            first_point = contour[0]
            x = first_point[0] * scale + offset_x
            y = paper_height - (first_point[1] * scale + offset_y)
            
            gcode.append(self.pen_up())
            gcode.append(self.move_to(x, y, rapid=True))
            gcode.append(self.pen_down(x, y))
            
            for point in contour[1:]:
                x = point[0] * scale + offset_x
                y = paper_height - (point[1] * scale + offset_y)
                
                if z_comp_enabled:
                    # 启用高度补偿时，每步都调整Z高度
                    gcode.append(self.move_to_with_z(x, y))
                else:
                    gcode.append(self.move_to(x, y, rapid=False))
            
            if len(contour) > 2:
                x = first_point[0] * scale + offset_x
                y = paper_height - (first_point[1] * scale + offset_y)
                if z_comp_enabled:
                    gcode.append(self.move_to_with_z(x, y))
                else:
                    gcode.append(self.move_to(x, y, rapid=False))
        
        gcode.extend(self.generate_footer())
        return gcode


class ImageProcessor:
    """图像处理类"""
    
    @staticmethod
    def load_image(path):
        """加载图片，支持中文路径"""
        # cv2.imread不支持中文路径，使用numpy和imdecode替代
        try:
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except:
            return None
    
    @staticmethod
    def to_grayscale(image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def detect_edges(image, low_threshold=50, high_threshold=150):
        gray = ImageProcessor.to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges
    
    @staticmethod
    def find_contours(edge_image, simplify=True, epsilon_factor=0.001):
        contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        result = []
        for contour in contours:
            if simplify:
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape(-1, 2).tolist()
            else:
                points = contour.reshape(-1, 2).tolist()
            
            if len(points) >= 2:
                result.append(points)
        
        return result
    
    @staticmethod
    def crop_image(image, x1, y1, x2, y2):
        return image[y1:y2, x1:x2].copy()


class GCodeApp:
    """GCode生成器 & GRBL控制器主应用"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("GCode生成器 & GRBL控制器")
        self.root.geometry("1500x950")
        
        # 状态变量
        self.original_image = None
        self.processed_image = None
        self.contours = []
        self.image_path = None
        self.gcode_lines = []
        self.debug_enabled_var = tk.BooleanVar(value=False)
        
        # 选择区域变量
        self.selection_start = None
        self.selection_rect = None
        self.crop_region = None
        
        # 原图旋转角度
        self.source_rotation = 0
        
        # 画布上的图像位置和缩放
        self.image_offset_x = 10
        self.image_offset_y = 10
        self.image_scale = 1.0
        
        # 拖动相关
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # 纸张设置
        self.paper_size = 'A4'
        self.paper_width, self.paper_height = PaperSizes.SIZES['A4']
        
        # 缩放比例 (像素/毫米)
        self.pixels_per_mm = 2.5
        
        # Ground参照功能
        self.ground_image = None        # 参照图片
        self.ground_photo = None        # Tk图片对象
        self.ground_points = []         # 定标点 [(x,y), (x,y)] 图片像素坐标
        self.ground_real_size = 100     # 实际尺寸(mm)
        self.ground_scale = 0.1         # mm/像素 缩放比例
        self.ground_offset_x = 0        # 偏移(mm)
        self.ground_offset_y = 0
        self.ground_rotation = 0        # 旋转角度(度)
        self.ground_locked = False      # 是否锁定
        self.ground_mode = None         # 'point1', 'point2', 'move', None
        self.ground_visible = True      # 是否显示
        
        # 图像/轮廓旋转
        self.image_rotation = 0         # 图像旋转角度(度)
        
        # === PS风格直接操作变量 ===
        self.selected_object = None     # 'contour' 或 'ground' 或 None
        self.transform_mode = None      # 'move', 'scale', 'rotate', None
        self.transform_handle = None    # 当前拖动的控制点: 'nw','n','ne','e','se','s','sw','w','rotate'
        self.transform_start_x = 0      # 拖动开始位置
        self.transform_start_y = 0
        self.transform_center_x = 0     # 变换中心(mm)
        self.transform_center_y = 0
        self.transform_start_angle = 0  # 旋转起始角度
        self.transform_start_scale = 1.0 # 缩放起始比例
        self.transform_start_offset_x = 0 # 移动起始偏移
        self.transform_start_offset_y = 0
        
        # 对象边界框缓存(mm坐标)
        self.contour_bounds = None      # (min_x, min_y, max_x, max_y) in mm
        self.ground_bounds = None       # (min_x, min_y, max_x, max_y) in mm
        
        # GRBL控制器
        self.grbl = GRBLController()
        self.grbl.callbacks['status'] = self.on_grbl_status
        self.grbl.callbacks['response'] = self.on_grbl_response
        self.grbl.callbacks['progress'] = self.on_grbl_progress
        self.grbl.callbacks['position'] = self.on_grbl_position
        
        # 状态查询定时器
        self.status_timer = None
        
        self._create_ui()
        self._start_status_polling()
    
    def _create_ui(self):
        """创建UI界面"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建左右分栏
        left_panel = ttk.Frame(main_frame, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧使用Notebook分类
        left_notebook = ttk.Notebook(left_panel)
        left_notebook.pack(fill=tk.BOTH, expand=True)
        
        # === 连接控制标签页 ===
        conn_tab = ttk.Frame(left_notebook)
        left_notebook.add(conn_tab, text="连接控制")
        
        # 串口连接区
        conn_frame = ttk.LabelFrame(conn_tab, text="设备连接")
        conn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        port_frame = ttk.Frame(conn_frame)
        port_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(port_frame, text="串口:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(port_frame, textvariable=self.port_var, width=12)
        self.port_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(port_frame, text="刷新", command=self.refresh_ports, width=6).pack(side=tk.LEFT)
        
        baud_frame = ttk.Frame(conn_frame)
        baud_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(baud_frame, text="波特率:").pack(side=tk.LEFT)
        self.baud_var = tk.StringVar(value='115200')
        baud_combo = ttk.Combobox(baud_frame, textvariable=self.baud_var, width=10,
                                   values=['9600', '19200', '38400', '57600', '115200', '250000'])
        baud_combo.pack(side=tk.LEFT, padx=5)
        
        conn_btn_frame = ttk.Frame(conn_frame)
        conn_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.connect_btn = ttk.Button(conn_btn_frame, text="连接", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # 连接状态
        self.conn_status_label = ttk.Label(conn_frame, text="未连接", foreground='red')
        self.conn_status_label.pack(pady=5)
        
        # 手动控制区
        jog_frame = ttk.LabelFrame(conn_tab, text="手动控制 (Jog)")
        jog_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 步进距离
        step_frame = ttk.Frame(jog_frame)
        step_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(step_frame, text="步进(mm):").pack(side=tk.LEFT)
        self.jog_step_var = tk.StringVar(value='10')
        step_combo = ttk.Combobox(step_frame, textvariable=self.jog_step_var, width=8,
                                   values=['0.1', '1', '5', '10', '50', '100'])
        step_combo.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(step_frame, text="速度:").pack(side=tk.LEFT)
        self.jog_speed_var = tk.StringVar(value='1000')
        ttk.Entry(step_frame, textvariable=self.jog_speed_var, width=6).pack(side=tk.LEFT, padx=2)
        
        # XY控制按钮
        xy_frame = ttk.Frame(jog_frame)
        xy_frame.pack(pady=5)
        
        ttk.Button(xy_frame, text="Y+", width=5, command=lambda: self.jog('Y', 1)).grid(row=0, column=1, pady=2)
        ttk.Button(xy_frame, text="X-", width=5, command=lambda: self.jog('X', -1)).grid(row=1, column=0, padx=2)
        ttk.Button(xy_frame, text="●", width=5, command=self.go_home).grid(row=1, column=1)
        ttk.Button(xy_frame, text="X+", width=5, command=lambda: self.jog('X', 1)).grid(row=1, column=2, padx=2)
        ttk.Button(xy_frame, text="Y-", width=5, command=lambda: self.jog('Y', -1)).grid(row=2, column=1, pady=2)
        
        # Z控制
        z_frame = ttk.Frame(jog_frame)
        z_frame.pack(pady=5)
        ttk.Button(z_frame, text="Z+", width=6, command=lambda: self.jog('Z', 1)).pack(side=tk.LEFT, padx=5)
        ttk.Button(z_frame, text="Z-", width=6, command=lambda: self.jog('Z', -1)).pack(side=tk.LEFT, padx=5)
        
        # 抬笔/落笔快捷按钮
        pen_frame = ttk.Frame(jog_frame)
        pen_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(pen_frame, text="抬笔", command=self.pen_up_action).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(pen_frame, text="落笔", command=self.pen_down_action).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        # 位置显示
        pos_frame = ttk.LabelFrame(conn_tab, text="当前位置")
        pos_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.pos_x_label = ttk.Label(pos_frame, text="X: 0.000", font=('Consolas', 11))
        self.pos_x_label.pack(side=tk.LEFT, padx=10)
        self.pos_y_label = ttk.Label(pos_frame, text="Y: 0.000", font=('Consolas', 11))
        self.pos_y_label.pack(side=tk.LEFT, padx=10)
        self.pos_z_label = ttk.Label(pos_frame, text="Z: 0.000", font=('Consolas', 11))
        self.pos_z_label.pack(side=tk.LEFT, padx=10)
        
        # 快捷操作
        quick_frame = ttk.LabelFrame(conn_tab, text="快捷操作")
        quick_frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_grid = ttk.Frame(quick_frame)
        btn_grid.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(btn_grid, text="归位 $H", command=self.grbl.home).grid(row=0, column=0, padx=2, pady=2, sticky='ew')
        ttk.Button(btn_grid, text="解锁 $X", command=self.grbl.unlock).grid(row=0, column=1, padx=2, pady=2, sticky='ew')
        ttk.Button(btn_grid, text="设为零点", command=lambda: self.grbl.set_zero()).grid(row=1, column=0, padx=2, pady=2, sticky='ew')
        ttk.Button(btn_grid, text="回零点", command=lambda: self.grbl.send_command("G0 X0 Y0 Z0")).grid(row=1, column=1, padx=2, pady=2, sticky='ew')
        ttk.Button(btn_grid, text="GRBL设置", command=self.open_grbl_settings).grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky='ew')
        
        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)
        
        # 发送指令窗口
        cmd_frame = ttk.LabelFrame(conn_tab, text="发送指令")
        cmd_frame.pack(fill=tk.X, padx=5, pady=5)
        
        cmd_input_frame = ttk.Frame(cmd_frame)
        cmd_input_frame.pack(fill=tk.X, padx=5, pady=5)
        self.manual_cmd_var = tk.StringVar()
        self.cmd_entry = ttk.Entry(cmd_input_frame, textvariable=self.manual_cmd_var, font=('Consolas', 10))
        self.cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cmd_entry.bind('<Return>', lambda e: self.send_manual_command())
        ttk.Button(cmd_input_frame, text="发送", command=self.send_manual_command, width=6).pack(side=tk.LEFT, padx=5)
        
        # 急停按钮
        emergency_frame = ttk.Frame(conn_tab)
        emergency_frame.pack(fill=tk.X, padx=5, pady=10)
        self.emergency_btn = tk.Button(emergency_frame, text="急 停", bg='red', fg='white',
                                        font=('Arial', 14, 'bold'), height=2, command=self.emergency_stop)
        self.emergency_btn.pack(fill=tk.X)
        
        # === 图像处理标签页 ===
        image_tab = ttk.Frame(left_notebook)
        left_notebook.add(image_tab, text="图像处理")
        
        # 创建可滚动容器
        image_canvas = tk.Canvas(image_tab, highlightthickness=0)
        image_scrollbar = ttk.Scrollbar(image_tab, orient="vertical", command=image_canvas.yview)
        image_scroll_frame = ttk.Frame(image_canvas)
        
        image_scroll_frame.bind("<Configure>", lambda e: image_canvas.configure(scrollregion=image_canvas.bbox("all")))
        image_canvas.create_window((0, 0), window=image_scroll_frame, anchor="nw", width=300)
        image_canvas.configure(yscrollcommand=image_scrollbar.set)
        
        # 鼠标滚轮支持 - 只在鼠标悬停在该区域时生效
        def _on_image_mousewheel(event):
            image_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        image_canvas.bind("<Enter>", lambda e: image_canvas.bind_all("<MouseWheel>", _on_image_mousewheel))
        image_canvas.bind("<Leave>", lambda e: image_canvas.unbind_all("<MouseWheel>"))
        
        image_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 以下所有内容放在 image_scroll_frame 中
        
        # 文件操作
        file_frame = ttk.LabelFrame(image_scroll_frame, text="文件操作")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="选择图片", command=self.load_image).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="加载GCode文件", command=self.load_gcode_file).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(file_frame, text="导出GCode", command=self.export_gcode).pack(fill=tk.X, padx=5, pady=2)
        
        # === 文字输入功能 ===
        text_frame = ttk.LabelFrame(image_scroll_frame, text="文字输入")
        text_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 文字输入框
        self.text_input_var = tk.StringVar(value='')
        ttk.Entry(text_frame, textvariable=self.text_input_var, font=('Arial', 12)).pack(fill=tk.X, padx=5, pady=2)
        
        # 字体设置
        font_frame = ttk.Frame(text_frame)
        font_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(font_frame, text="字体:").pack(side=tk.LEFT)
        self.font_var = tk.StringVar(value='simhei')
        font_combo = ttk.Combobox(font_frame, textvariable=self.font_var, width=14,
                                   values=['simhei', 'simsun', 'msyh', 'arial', 'times', 'consola'])
        font_combo.pack(side=tk.LEFT, padx=2)
        
        # 字体大小滑块
        size_frame = ttk.Frame(text_frame)
        size_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(size_frame, text="大小:").pack(side=tk.LEFT)
        self.font_size_var = tk.IntVar(value=80)
        self.font_size_label = ttk.Label(size_frame, text="80")
        self.font_size_label.pack(side=tk.RIGHT, padx=5)
        font_size_scale = ttk.Scale(size_frame, from_=20, to=200, variable=self.font_size_var, 
                                     orient=tk.HORIZONTAL, command=lambda v: self.font_size_label.config(text=str(int(float(v)))))
        font_size_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 笔画粗细滑块
        stroke_frame = ttk.Frame(text_frame)
        stroke_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(stroke_frame, text="笔画:").pack(side=tk.LEFT)
        self.stroke_width_var = tk.IntVar(value=3)
        self.stroke_width_label = ttk.Label(stroke_frame, text="3")
        self.stroke_width_label.pack(side=tk.RIGHT, padx=5)
        stroke_scale = ttk.Scale(stroke_frame, from_=1, to=10, variable=self.stroke_width_var, 
                                  orient=tk.HORIZONTAL, command=lambda v: self.stroke_width_label.config(text=str(int(float(v)))))
        stroke_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 生成按钮
        ttk.Button(text_frame, text="生成文字骨架", command=self.generate_text_contours).pack(fill=tk.X, padx=5, pady=5)
        
        # 纸张设置
        paper_frame = ttk.LabelFrame(image_scroll_frame, text="纸张/工作区域")
        paper_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(paper_frame, text="规格:").pack(anchor=tk.W, padx=5, pady=2)
        self.paper_var = tk.StringVar(value='A4')
        paper_combo = ttk.Combobox(paper_frame, textvariable=self.paper_var, 
                                   values=list(PaperSizes.SIZES.keys()), state='readonly')
        paper_combo.pack(fill=tk.X, padx=5, pady=2)
        paper_combo.bind('<<ComboboxSelected>>', self.on_paper_change)
        
        custom_frame = ttk.Frame(paper_frame)
        custom_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(custom_frame, text="宽:").pack(side=tk.LEFT)
        self.custom_width_var = tk.StringVar(value='200')
        ttk.Entry(custom_frame, textvariable=self.custom_width_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(custom_frame, text="高:").pack(side=tk.LEFT)
        self.custom_height_var = tk.StringVar(value='200')
        ttk.Entry(custom_frame, textvariable=self.custom_height_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(custom_frame, text="mm").pack(side=tk.LEFT)
        ttk.Button(paper_frame, text="应用自定义", command=self.apply_custom_size).pack(fill=tk.X, padx=5, pady=2)
        
        orient_frame = ttk.Frame(paper_frame)
        orient_frame.pack(fill=tk.X, padx=5, pady=2)
        self.orientation_var = tk.StringVar(value='portrait')
        ttk.Radiobutton(orient_frame, text="纵向", variable=self.orientation_var, 
                        value='portrait', command=self.on_orientation_change).pack(side=tk.LEFT)
        ttk.Radiobutton(orient_frame, text="横向", variable=self.orientation_var, 
                        value='landscape', command=self.on_orientation_change).pack(side=tk.LEFT)
        
        # 边缘检测参数
        process_frame = ttk.LabelFrame(image_scroll_frame, text="边缘检测")
        process_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(process_frame, text="阈值(低):").pack(anchor=tk.W, padx=5)
        self.low_threshold_var = tk.IntVar(value=50)
        ttk.Scale(process_frame, from_=0, to=255, variable=self.low_threshold_var, 
                  orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5)
        
        ttk.Label(process_frame, text="阈值(高):").pack(anchor=tk.W, padx=5)
        self.high_threshold_var = tk.IntVar(value=150)
        ttk.Scale(process_frame, from_=0, to=255, variable=self.high_threshold_var, 
                  orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5)
        
        ttk.Label(process_frame, text="轮廓简化:").pack(anchor=tk.W, padx=5)
        self.simplify_var = tk.DoubleVar(value=0.002)
        ttk.Scale(process_frame, from_=0.0001, to=0.01, variable=self.simplify_var, 
                  orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5)
        
        # OCR模式选项
        ttk.Separator(process_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)
        self.ocr_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(process_frame, text="OCR文字识别模式", variable=self.ocr_mode_var).pack(anchor=tk.W, padx=5)
        
        ocr_font_frame = ttk.Frame(process_frame)
        ocr_font_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(ocr_font_frame, text="输出字体:").pack(side=tk.LEFT)
        self.ocr_font_var = tk.StringVar(value='simhei')
        ocr_font_combo = ttk.Combobox(ocr_font_frame, textvariable=self.ocr_font_var, width=10,
                                       values=['simhei', 'simsun', 'msyh', 'arial', 'times'])
        ocr_font_combo.pack(side=tk.LEFT, padx=5)

        ocr_dbg_frame = ttk.Frame(process_frame)
        ocr_dbg_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(ocr_dbg_frame, text="运行调试输出", variable=self.debug_enabled_var).pack(side=tk.LEFT)
        ttk.Button(ocr_dbg_frame, text="清空调试", command=self.clear_debug_log, width=8).pack(side=tk.RIGHT)

        ocr_cmp_frame = ttk.Frame(process_frame)
        ocr_cmp_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(ocr_cmp_frame, text="原图/识别后对比", command=self.show_ocr_compare).pack(fill=tk.X)
        
        ttk.Button(process_frame, text="处理图像", command=self.process_image).pack(fill=tk.X, padx=5, pady=5)
        
        # === 工程化单线化处理 ===
        thinning_frame = ttk.LabelFrame(image_scroll_frame, text="工程化单线化")
        thinning_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(thinning_frame, text="将图像转换为单像素细线:", foreground='blue').pack(anchor=tk.W, padx=5, pady=2)
        ttk.Label(thinning_frame, text="• 文字区域: 提取中轴骨架", foreground='gray').pack(anchor=tk.W, padx=10)
        ttk.Label(thinning_frame, text="• 非文字区域: 细线化处理", foreground='gray').pack(anchor=tk.W, padx=10)
        
        # 文字检测灵敏度
        text_sens_frame = ttk.Frame(thinning_frame)
        text_sens_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(text_sens_frame, text="文字检测灵敏度:").pack(side=tk.LEFT)
        self.text_sensitivity_var = tk.IntVar(value=50)
        self.text_sens_label = ttk.Label(text_sens_frame, text="50", width=3)
        self.text_sens_label.pack(side=tk.RIGHT, padx=5)
        text_sens_scale = ttk.Scale(text_sens_frame, from_=0, to=100, variable=self.text_sensitivity_var,
                                     orient=tk.HORIZONTAL, command=lambda v: self.text_sens_label.config(text=str(int(float(v)))))
        text_sens_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 去噪强度
        denoise_frame = ttk.Frame(thinning_frame)
        denoise_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(denoise_frame, text="去噪强度:").pack(side=tk.LEFT)
        self.denoise_var = tk.IntVar(value=3)
        self.denoise_label = ttk.Label(denoise_frame, text="3", width=3)
        self.denoise_label.pack(side=tk.RIGHT, padx=5)
        denoise_scale = ttk.Scale(denoise_frame, from_=0, to=10, variable=self.denoise_var,
                                   orient=tk.HORIZONTAL, command=lambda v: self.denoise_label.config(text=str(int(float(v)))))
        denoise_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 最小轮廓长度
        min_len_frame = ttk.Frame(thinning_frame)
        min_len_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(min_len_frame, text="最小线条长度:").pack(side=tk.LEFT)
        self.min_contour_len_var = tk.IntVar(value=10)
        self.min_len_label = ttk.Label(min_len_frame, text="10px", width=5)
        self.min_len_label.pack(side=tk.RIGHT, padx=5)
        min_len_scale = ttk.Scale(min_len_frame, from_=1, to=50, variable=self.min_contour_len_var,
                                   orient=tk.HORIZONTAL, command=lambda v: self.min_len_label.config(text=f"{int(float(v))}px"))
        min_len_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 处理按钮
        ttk.Button(thinning_frame, text="工程化单线化处理", command=self.process_engineering_thinning).pack(fill=tk.X, padx=5, pady=5)

        algo1_frame = ttk.LabelFrame(image_scroll_frame, text="算法一：Medial Axis（平滑+路径优化）")
        algo1_frame.pack(fill=tk.X, padx=5, pady=5)

        algo1_denoise_frame = ttk.Frame(algo1_frame)
        algo1_denoise_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo1_denoise_frame, text="去噪强度:").pack(side=tk.LEFT)
        self.ma_denoise_var = tk.IntVar(value=3)
        self.ma_denoise_label = ttk.Label(algo1_denoise_frame, text="3", width=3)
        self.ma_denoise_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo1_denoise_frame, from_=0, to=10, variable=self.ma_denoise_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.ma_denoise_label.config(text=str(int(float(v))))).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo1_minlen_frame = ttk.Frame(algo1_frame)
        algo1_minlen_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo1_minlen_frame, text="最小线条长度:").pack(side=tk.LEFT)
        self.ma_min_contour_len_var = tk.IntVar(value=10)
        self.ma_min_contour_len_label = ttk.Label(algo1_minlen_frame, text="10px", width=6)
        self.ma_min_contour_len_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo1_minlen_frame, from_=1, to=200, variable=self.ma_min_contour_len_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.ma_min_contour_len_label.config(text=f"{int(float(v))}px")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo1_simplify_frame = ttk.Frame(algo1_frame)
        algo1_simplify_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo1_simplify_frame, text="轮廓简化:").pack(side=tk.LEFT)
        self.ma_simplify_var = tk.DoubleVar(value=0.002)
        ttk.Scale(algo1_simplify_frame, from_=0.0001, to=0.02, variable=self.ma_simplify_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo1_smooth_frame = ttk.Frame(algo1_frame)
        algo1_smooth_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo1_smooth_frame, text="B-spline平滑:").pack(side=tk.LEFT)
        self.ma_bspline_iter_var = tk.IntVar(value=2)
        self.ma_bspline_iter_label = ttk.Label(algo1_smooth_frame, text="2", width=3)
        self.ma_bspline_iter_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo1_smooth_frame, from_=0, to=5, variable=self.ma_bspline_iter_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.ma_bspline_iter_label.config(text=str(int(float(v))))).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.ma_optimize_paths_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(algo1_frame, text="路径优化(减少空行程)", variable=self.ma_optimize_paths_var).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Button(algo1_frame, text="算法一：Medial Axis 处理", command=self.process_algo_medial_axis).pack(fill=tk.X, padx=5, pady=5)

        algo2_frame = ttk.LabelFrame(image_scroll_frame, text="算法二：二值化骨架 + Hough直线替换 + 曲线平滑")
        algo2_frame.pack(fill=tk.X, padx=5, pady=5)

        algo2_denoise_frame = ttk.Frame(algo2_frame)
        algo2_denoise_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo2_denoise_frame, text="去噪强度:").pack(side=tk.LEFT)
        self.hough2_denoise_var = tk.IntVar(value=3)
        self.hough2_denoise_label = ttk.Label(algo2_denoise_frame, text="3", width=3)
        self.hough2_denoise_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo2_denoise_frame, from_=0, to=10, variable=self.hough2_denoise_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.hough2_denoise_label.config(text=str(int(float(v))))).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo2_bin_frame = ttk.Frame(algo2_frame)
        algo2_bin_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo2_bin_frame, text="二值化:").pack(side=tk.LEFT)
        self.hough2_binarize_method_var = tk.StringVar(value='adaptive')
        ttk.Combobox(algo2_bin_frame, textvariable=self.hough2_binarize_method_var, width=12, state='readonly',
                     values=['adaptive', 'otsu', 'fixed']).pack(side=tk.RIGHT, padx=2)

        algo2_bin_th_frame = ttk.Frame(algo2_frame)
        algo2_bin_th_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo2_bin_th_frame, text="阈值(固定):").pack(side=tk.LEFT)
        self.hough2_fixed_threshold_var = tk.IntVar(value=128)
        self.hough2_fixed_threshold_label = ttk.Label(algo2_bin_th_frame, text="128", width=4)
        self.hough2_fixed_threshold_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo2_bin_th_frame, from_=0, to=255, variable=self.hough2_fixed_threshold_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.hough2_fixed_threshold_label.config(text=str(int(float(v))))).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo2_minlen_frame = ttk.Frame(algo2_frame)
        algo2_minlen_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo2_minlen_frame, text="最小线条长度:").pack(side=tk.LEFT)
        self.hough2_min_contour_len_var = tk.IntVar(value=10)
        self.hough2_min_contour_len_label = ttk.Label(algo2_minlen_frame, text="10px", width=6)
        self.hough2_min_contour_len_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo2_minlen_frame, from_=1, to=200, variable=self.hough2_min_contour_len_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.hough2_min_contour_len_label.config(text=f"{int(float(v))}px")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo2_hough_len_frame = ttk.Frame(algo2_frame)
        algo2_hough_len_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo2_hough_len_frame, text="直线最小长度:").pack(side=tk.LEFT)
        self.hough2_min_len_var = tk.IntVar(value=60)
        self.hough2_min_len_label = ttk.Label(algo2_hough_len_frame, text="60px", width=6)
        self.hough2_min_len_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo2_hough_len_frame, from_=10, to=500, variable=self.hough2_min_len_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.hough2_min_len_label.config(text=f"{int(float(v))}px")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo2_hough_gap_frame = ttk.Frame(algo2_frame)
        algo2_hough_gap_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo2_hough_gap_frame, text="直线最大间隙:").pack(side=tk.LEFT)
        self.hough2_max_gap_var = tk.IntVar(value=10)
        self.hough2_max_gap_label = ttk.Label(algo2_hough_gap_frame, text="10px", width=6)
        self.hough2_max_gap_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo2_hough_gap_frame, from_=0, to=100, variable=self.hough2_max_gap_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.hough2_max_gap_label.config(text=f"{int(float(v))}px")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo2_simplify_frame = ttk.Frame(algo2_frame)
        algo2_simplify_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo2_simplify_frame, text="轮廓简化:").pack(side=tk.LEFT)
        self.hough2_simplify_var = tk.DoubleVar(value=0.002)
        ttk.Scale(algo2_simplify_frame, from_=0.0001, to=0.02, variable=self.hough2_simplify_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo2_smooth_frame = ttk.Frame(algo2_frame)
        algo2_smooth_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo2_smooth_frame, text="曲线B-spline平滑:").pack(side=tk.LEFT)
        self.hough2_bspline_iter_var = tk.IntVar(value=2)
        self.hough2_bspline_iter_label = ttk.Label(algo2_smooth_frame, text="2", width=3)
        self.hough2_bspline_iter_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo2_smooth_frame, from_=0, to=5, variable=self.hough2_bspline_iter_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.hough2_bspline_iter_label.config(text=str(int(float(v))))).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.hough2_optimize_paths_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(algo2_frame, text="路径优化(减少空行程)", variable=self.hough2_optimize_paths_var).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Button(algo2_frame, text="算法二：Hough直线替换 处理", command=self.process_algo_hough_replace).pack(fill=tk.X, padx=5, pady=5)

        algo3_frame = ttk.LabelFrame(image_scroll_frame, text="算法三：Potrace + 直线约束")
        algo3_frame.pack(fill=tk.X, padx=5, pady=5)

        algo3_denoise_frame = ttk.Frame(algo3_frame)
        algo3_denoise_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo3_denoise_frame, text="去噪强度:").pack(side=tk.LEFT)
        self.potrace_denoise_var = tk.IntVar(value=3)
        self.potrace_denoise_label = ttk.Label(algo3_denoise_frame, text="3", width=3)
        self.potrace_denoise_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo3_denoise_frame, from_=0, to=10, variable=self.potrace_denoise_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.potrace_denoise_label.config(text=str(int(float(v))))).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo3_bin_frame = ttk.Frame(algo3_frame)
        algo3_bin_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo3_bin_frame, text="二值化:").pack(side=tk.LEFT)
        self.potrace_binarize_method_var = tk.StringVar(value='otsu')
        ttk.Combobox(algo3_bin_frame, textvariable=self.potrace_binarize_method_var, width=12, state='readonly',
                     values=['adaptive', 'otsu', 'fixed']).pack(side=tk.RIGHT, padx=2)

        algo3_bin_th_frame = ttk.Frame(algo3_frame)
        algo3_bin_th_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo3_bin_th_frame, text="阈值(固定):").pack(side=tk.LEFT)
        self.potrace_fixed_threshold_var = tk.IntVar(value=128)
        self.potrace_fixed_threshold_label = ttk.Label(algo3_bin_th_frame, text="128", width=4)
        self.potrace_fixed_threshold_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo3_bin_th_frame, from_=0, to=255, variable=self.potrace_fixed_threshold_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.potrace_fixed_threshold_label.config(text=str(int(float(v))))).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo3_simplify_frame = ttk.Frame(algo3_frame)
        algo3_simplify_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo3_simplify_frame, text="轮廓简化:").pack(side=tk.LEFT)
        self.potrace_simplify_var = tk.DoubleVar(value=0.002)
        ttk.Scale(algo3_simplify_frame, from_=0.0001, to=0.05, variable=self.potrace_simplify_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo3_minlen_frame = ttk.Frame(algo3_frame)
        algo3_minlen_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo3_minlen_frame, text="最小轮廓长度:").pack(side=tk.LEFT)
        self.potrace_min_contour_len_var = tk.IntVar(value=20)
        self.potrace_min_contour_len_label = ttk.Label(algo3_minlen_frame, text="20px", width=6)
        self.potrace_min_contour_len_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo3_minlen_frame, from_=2, to=500, variable=self.potrace_min_contour_len_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.potrace_min_contour_len_label.config(text=f"{int(float(v))}px")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo3_lineang_frame = ttk.Frame(algo3_frame)
        algo3_lineang_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo3_lineang_frame, text="直线角度容差:").pack(side=tk.LEFT)
        self.potrace_line_angle_tol_var = tk.IntVar(value=8)
        self.potrace_line_angle_tol_label = ttk.Label(algo3_lineang_frame, text="8°", width=5)
        self.potrace_line_angle_tol_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo3_lineang_frame, from_=1, to=30, variable=self.potrace_line_angle_tol_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.potrace_line_angle_tol_label.config(text=f"{int(float(v))}°")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo3_linedist_frame = ttk.Frame(algo3_frame)
        algo3_linedist_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo3_linedist_frame, text="直线距离容差:").pack(side=tk.LEFT)
        self.potrace_line_dist_tol_var = tk.IntVar(value=2)
        self.potrace_line_dist_tol_label = ttk.Label(algo3_linedist_frame, text="2px", width=6)
        self.potrace_line_dist_tol_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo3_linedist_frame, from_=0, to=10, variable=self.potrace_line_dist_tol_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.potrace_line_dist_tol_label.config(text=f"{int(float(v))}px")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Button(algo3_frame, text="算法三：Potrace 处理", command=self.process_algo_potrace).pack(fill=tk.X, padx=5, pady=5)

        algo4_frame = ttk.LabelFrame(image_scroll_frame, text="算法四：DeepSVG / Pix2Vector")
        algo4_frame.pack(fill=tk.X, padx=5, pady=5)

        algo4_backend_frame = ttk.Frame(algo4_frame)
        algo4_backend_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo4_backend_frame, text="后端:").pack(side=tk.LEFT)
        self.dsvg_backend_var = tk.StringVar(value='DeepSVG')
        ttk.Combobox(algo4_backend_frame, textvariable=self.dsvg_backend_var, width=12, state='readonly',
                     values=['DeepSVG', 'Pix2Vector']).pack(side=tk.RIGHT, padx=2)

        algo4_cmd_frame = ttk.Frame(algo4_frame)
        algo4_cmd_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo4_cmd_frame, text="推理命令(可空):").pack(side=tk.LEFT)
        self.dsvg_command_var = tk.StringVar(value='')
        ttk.Entry(algo4_cmd_frame, textvariable=self.dsvg_command_var, width=22).pack(side=tk.RIGHT, padx=2)

        algo4_sample_frame = ttk.Frame(algo4_frame)
        algo4_sample_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo4_sample_frame, text="曲线采样步长:").pack(side=tk.LEFT)
        self.dsvg_sample_step_var = tk.IntVar(value=4)
        self.dsvg_sample_step_label = ttk.Label(algo4_sample_frame, text="4px", width=6)
        self.dsvg_sample_step_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo4_sample_frame, from_=1, to=20, variable=self.dsvg_sample_step_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.dsvg_sample_step_label.config(text=f"{int(float(v))}px")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo4_simplify_frame = ttk.Frame(algo4_frame)
        algo4_simplify_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo4_simplify_frame, text="轮廓简化:").pack(side=tk.LEFT)
        self.dsvg_simplify_var = tk.DoubleVar(value=0.002)
        ttk.Scale(algo4_simplify_frame, from_=0.0001, to=0.05, variable=self.dsvg_simplify_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        algo4_minlen_frame = ttk.Frame(algo4_frame)
        algo4_minlen_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(algo4_minlen_frame, text="最小线条长度:").pack(side=tk.LEFT)
        self.dsvg_min_contour_len_var = tk.IntVar(value=20)
        self.dsvg_min_contour_len_label = ttk.Label(algo4_minlen_frame, text="20px", width=6)
        self.dsvg_min_contour_len_label.pack(side=tk.RIGHT, padx=5)
        ttk.Scale(algo4_minlen_frame, from_=2, to=500, variable=self.dsvg_min_contour_len_var,
                  orient=tk.HORIZONTAL, command=lambda v: self.dsvg_min_contour_len_label.config(text=f"{int(float(v))}px")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.dsvg_optimize_paths_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(algo4_frame, text="路径优化(减少空行程)", variable=self.dsvg_optimize_paths_var).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Button(algo4_frame, text="算法四：DeepSVG / Pix2Vector 处理", command=self.process_algo_deepsvg_pix2vector).pack(fill=tk.X, padx=5, pady=5)
        
        # 区域选择
        region_frame = ttk.LabelFrame(image_scroll_frame, text="区域选择")
        region_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 原图旋转调整
        source_rotate_frame = ttk.Frame(region_frame)
        source_rotate_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(source_rotate_frame, text="旋转:").pack(side=tk.LEFT)
        self.source_rotation_var = tk.IntVar(value=0)
        self.source_rotation_label = ttk.Label(source_rotate_frame, text="0°", width=5)
        self.source_rotation_label.pack(side=tk.RIGHT)
        source_rotate_scale = ttk.Scale(source_rotate_frame, from_=-180, to=180, variable=self.source_rotation_var,
                                          orient=tk.HORIZONTAL, command=self._on_source_rotation_change)
        source_rotate_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 旋转快捷按钮
        source_rotate_btn_frame = ttk.Frame(region_frame)
        source_rotate_btn_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(source_rotate_btn_frame, text="-90°", width=5, command=lambda: self._rotate_source(-90)).pack(side=tk.LEFT, padx=1)
        ttk.Button(source_rotate_btn_frame, text="-45°", width=5, command=lambda: self._rotate_source(-45)).pack(side=tk.LEFT, padx=1)
        ttk.Button(source_rotate_btn_frame, text="0°", width=5, command=lambda: self._rotate_source(0, absolute=True)).pack(side=tk.LEFT, padx=1)
        ttk.Button(source_rotate_btn_frame, text="+45°", width=5, command=lambda: self._rotate_source(45)).pack(side=tk.LEFT, padx=1)
        ttk.Button(source_rotate_btn_frame, text="+90°", width=5, command=lambda: self._rotate_source(90)).pack(side=tk.LEFT, padx=1)
        
        ttk.Separator(region_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(region_frame, text="框选区域", command=self.start_selection_mode).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(region_frame, text="确认区域", command=self.confirm_selection).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(region_frame, text="重置", command=self.reset_selection).pack(fill=tk.X, padx=5, pady=2)
        self.selection_label = ttk.Label(region_frame, text="未选择")
        self.selection_label.pack(padx=5, pady=2)
        
        # === 图像旋转功能 ===
        rotate_frame = ttk.LabelFrame(image_scroll_frame, text="图像旋转")
        rotate_frame.pack(fill=tk.X, padx=5, pady=5)
        
        rotate_slider_frame = ttk.Frame(rotate_frame)
        rotate_slider_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(rotate_slider_frame, text="角度:").pack(side=tk.LEFT)
        self.image_rotation_var = tk.IntVar(value=0)
        self.image_rotation_label = ttk.Label(rotate_slider_frame, text="0°", width=5)
        self.image_rotation_label.pack(side=tk.RIGHT)
        image_rotate_scale = ttk.Scale(rotate_slider_frame, from_=-180, to=180, variable=self.image_rotation_var,
                                        orient=tk.HORIZONTAL, command=self._on_image_rotation_change)
        image_rotate_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        rotate_btn_frame = ttk.Frame(rotate_frame)
        rotate_btn_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(rotate_btn_frame, text="-90°", width=6, command=lambda: self._rotate_image(-90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rotate_btn_frame, text="-45°", width=6, command=lambda: self._rotate_image(-45)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rotate_btn_frame, text="0°", width=6, command=lambda: self._rotate_image(0, absolute=True)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rotate_btn_frame, text="+45°", width=6, command=lambda: self._rotate_image(45)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rotate_btn_frame, text="+90°", width=6, command=lambda: self._rotate_image(90)).pack(side=tk.LEFT, padx=2)
        
        # === Ground参照功能 ===
        ground_frame = ttk.LabelFrame(image_scroll_frame, text="Ground参照")
        ground_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 加载参照图
        ground_load_frame = ttk.Frame(ground_frame)
        ground_load_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(ground_load_frame, text="加载参照图片", command=self.load_ground_image).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        self.ground_lock_btn = ttk.Button(ground_load_frame, text="锁定", command=self.toggle_ground_lock)
        self.ground_lock_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        
        # 显示控制
        ground_show_frame = ttk.Frame(ground_frame)
        ground_show_frame.pack(fill=tk.X, padx=5, pady=2)
        self.ground_visible_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ground_show_frame, text="显示参照图", variable=self.ground_visible_var, 
                       command=self.toggle_ground_visible).pack(side=tk.LEFT)
        ttk.Button(ground_show_frame, text="清除", command=self.clear_ground).pack(side=tk.RIGHT, padx=5)
        
        # 操作提示
        ttk.Label(ground_frame, text="点击选中后可拖动/缩放/旋转", foreground='blue').pack(padx=5, pady=2)
        
        # 状态显示
        self.ground_status_label = ttk.Label(ground_frame, text="未加载参照图", foreground='gray')
        self.ground_status_label.pack(padx=5, pady=2)
        
        # 保留必要的变量初始化
        self.ground_rotation_var = tk.IntVar(value=0)
        self.ground_rotation_label = ttk.Label(ground_frame)  # 隐藏的label
        self.ground_size_var = tk.StringVar(value='100')
        
        # === GCode设置标签页 ===
        gcode_tab = ttk.Frame(left_notebook)
        left_notebook.add(gcode_tab, text="GCode设置")
        
        # 位置和缩放
        position_frame = ttk.LabelFrame(gcode_tab, text="位置和缩放")
        position_frame.pack(fill=tk.X, padx=5, pady=5)
        
        pos_grid = ttk.Frame(position_frame)
        pos_grid.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(pos_grid, text="X偏移(mm):").grid(row=0, column=0, sticky=tk.W)
        self.offset_x_var = tk.StringVar(value='10')
        ttk.Entry(pos_grid, textvariable=self.offset_x_var, width=8).grid(row=0, column=1, padx=2)
        
        ttk.Label(pos_grid, text="Y偏移(mm):").grid(row=1, column=0, sticky=tk.W)
        self.offset_y_var = tk.StringVar(value='10')
        ttk.Entry(pos_grid, textvariable=self.offset_y_var, width=8).grid(row=1, column=1, padx=2)
        
        ttk.Label(pos_grid, text="缩放比例:").grid(row=2, column=0, sticky=tk.W)
        self.scale_var = tk.StringVar(value='1.0')
        ttk.Entry(pos_grid, textvariable=self.scale_var, width=8).grid(row=2, column=1, padx=2)
        
        ttk.Button(position_frame, text="应用", command=self.apply_position).pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(position_frame, text="拖动预览区调整位置", foreground='blue').pack(padx=5)
        
        # GCode参数
        gcode_param_frame = ttk.LabelFrame(gcode_tab, text="GCode参数")
        gcode_param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        param_grid = ttk.Frame(gcode_param_frame)
        param_grid.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(param_grid, text="进给速度:").grid(row=0, column=0, sticky=tk.W)
        self.feed_rate_var = tk.StringVar(value='1000')
        ttk.Entry(param_grid, textvariable=self.feed_rate_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(param_grid, text="mm/min").grid(row=0, column=2, sticky=tk.W)
        
        ttk.Label(param_grid, text="抬笔高度:").grid(row=1, column=0, sticky=tk.W)
        self.z_up_var = tk.StringVar(value='5')
        ttk.Entry(param_grid, textvariable=self.z_up_var, width=8).grid(row=1, column=1, padx=2)
        ttk.Label(param_grid, text="mm").grid(row=1, column=2, sticky=tk.W)
        
        ttk.Label(param_grid, text="落笔高度:").grid(row=2, column=0, sticky=tk.W)
        self.z_down_var = tk.StringVar(value='0')
        ttk.Entry(param_grid, textvariable=self.z_down_var, width=8).grid(row=2, column=1, padx=2)
        ttk.Label(param_grid, text="mm").grid(row=2, column=2, sticky=tk.W)
        
        # === 玻璃板高度补偿 ===
        z_comp_frame = ttk.LabelFrame(gcode_tab, text="玻璃板高度补偿")
        z_comp_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 启用开关
        self.z_comp_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(z_comp_frame, text="启用高度补偿", variable=self.z_comp_enabled_var).pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Label(z_comp_frame, text="左右钢架固定，仅补偿X方向形变:", foreground='blue').pack(anchor=tk.W, padx=5)
        
        # 三点高度输入：左侧、中间、右侧
        z_grid = ttk.Frame(z_comp_frame)
        z_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # 左侧
        ttk.Label(z_grid, text="左侧:").grid(row=0, column=0, sticky=tk.E, padx=2)
        self.z_left_var = tk.StringVar(value='0')
        ttk.Entry(z_grid, textvariable=self.z_left_var, width=6).grid(row=0, column=1, padx=2)
        
        # 中间
        ttk.Label(z_grid, text="中间:").grid(row=0, column=2, sticky=tk.E, padx=2)
        self.z_center_var = tk.StringVar(value='0')
        ttk.Entry(z_grid, textvariable=self.z_center_var, width=6).grid(row=0, column=3, padx=2)
        
        # 右侧
        ttk.Label(z_grid, text="右侧:").grid(row=0, column=4, sticky=tk.E, padx=2)
        self.z_right_var = tk.StringVar(value='0')
        ttk.Entry(z_grid, textvariable=self.z_right_var, width=6).grid(row=0, column=5, padx=2)
        ttk.Label(z_grid, text="mm").grid(row=0, column=6, sticky=tk.W)
        
        # 曲率控制
        curve_frame = ttk.Frame(z_comp_frame)
        curve_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(curve_frame, text="曲率:").pack(side=tk.LEFT)
        self.z_curve_var = tk.DoubleVar(value=0.5)
        self.z_curve_label = ttk.Label(curve_frame, text="0.5", width=4)
        self.z_curve_label.pack(side=tk.RIGHT)
        z_curve_scale = ttk.Scale(curve_frame, from_=0, to=1, variable=self.z_curve_var,
                                   orient=tk.HORIZONTAL, 
                                   command=lambda v: self.z_curve_label.config(text=f"{float(v):.1f}"))
        z_curve_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(z_comp_frame, text="0=线性  0.5=平缓  1=陡峭", font=('Arial', 8), foreground='gray').pack()
        
        # 提示图
        ttk.Label(z_comp_frame, text="钢架│      中间下凹      │钢架", font=('Consolas', 9), foreground='gray').pack()
        ttk.Label(z_comp_frame, text="  左 │════╬════│ 右", font=('Consolas', 9), foreground='gray').pack()
        
        # 导出按钮
        ttk.Button(gcode_param_frame, text="导出GCode文件", command=self.save_gcode).pack(fill=tk.X, padx=5, pady=5)
        
        # 发送控制
        send_frame = ttk.LabelFrame(gcode_tab, text="发送GCode")
        send_frame.pack(fill=tk.X, padx=5, pady=5)
        
        send_btn_frame = ttk.Frame(send_frame)
        send_btn_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.start_btn = ttk.Button(send_btn_frame, text="开始发送", command=self.start_streaming)
        self.start_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.pause_btn = ttk.Button(send_btn_frame, text="暂停", command=self.pause_streaming, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.resume_btn = ttk.Button(send_btn_frame, text="继续", command=self.resume_streaming, state=tk.DISABLED)
        self.resume_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.end_btn = ttk.Button(send_btn_frame, text="结束", command=self.end_streaming, state=tk.DISABLED)
        self.end_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        # 发送状态: 'idle', 'streaming', 'paused'
        self.stream_state = 'idle'
        
        # 进度条
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(send_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        self.progress_label = ttk.Label(send_frame, text="0 / 0")
        self.progress_label.pack(pady=2)
        
        # 状态标签
        self.status_label = ttk.Label(gcode_tab, text="就绪", foreground='green')
        self.status_label.pack(pady=10)
        
        # 右侧区域
        # 创建上下分栏
        top_right = ttk.Frame(right_panel)
        top_right.pack(fill=tk.BOTH, expand=True)
        
        bottom_right = ttk.Frame(right_panel, height=200)
        bottom_right.pack(fill=tk.X, side=tk.BOTTOM)
        bottom_right.pack_propagate(False)
        
        # 右上：画布区域
        self.notebook = ttk.Notebook(top_right)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 原图标签页
        original_tab = ttk.Frame(self.notebook)
        self.notebook.add(original_tab, text="原图/选择区域")
        
        self.original_canvas = tk.Canvas(original_tab, bg='gray85')
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        self.original_canvas.bind('<Button-1>', self.on_original_canvas_click)
        self.original_canvas.bind('<B1-Motion>', self.on_original_canvas_drag)
        self.original_canvas.bind('<ButtonRelease-1>', self.on_original_canvas_release)
        
        # 预览标签页
        preview_tab = ttk.Frame(self.notebook)
        self.notebook.add(preview_tab, text="纸张预览")
        
        self.preview_canvas = tk.Canvas(preview_tab, bg='white')
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind('<Button-1>', self.on_preview_click)
        self.preview_canvas.bind('<B1-Motion>', self.on_preview_drag)
        self.preview_canvas.bind('<ButtonRelease-1>', self.on_preview_release)
        self.preview_canvas.bind('<MouseWheel>', self.on_mouse_wheel)
        
        # 绑定Delete键删除选中对象
        self.root.bind('<Delete>', self.delete_selected_object)
        
        bottom_notebook = ttk.Notebook(bottom_right)
        bottom_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        console_tab = ttk.Frame(bottom_notebook)
        bottom_notebook.add(console_tab, text="控制台")
        self.console = scrolledtext.ScrolledText(console_tab, height=10, font=('Consolas', 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.config(state=tk.DISABLED)

        debug_tab = ttk.Frame(bottom_notebook)
        bottom_notebook.add(debug_tab, text="调试")
        self.debug_console = scrolledtext.ScrolledText(debug_tab, height=10, font=('Consolas', 9))
        self.debug_console.pack(fill=tk.BOTH, expand=True)
        self.debug_console.config(state=tk.DISABLED)
        
        # 初始化
        self.selection_mode = False
        self.refresh_ports()
        self.root.after(100, self.draw_paper)
    
    def _start_status_polling(self):
        """启动状态轮询"""
        if self.grbl.connected:
            self.grbl.query_status()
        self.status_timer = self.root.after(500, self._start_status_polling)
    
    def log(self, message):
        """写入控制台"""
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, message + '\n')
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)

    def debug_log(self, message):
        if not getattr(self, 'debug_enabled_var', None):
            return
        if not self.debug_enabled_var.get():
            return
        if not getattr(self, 'debug_console', None):
            return
        self.debug_console.config(state=tk.NORMAL)
        self.debug_console.insert(tk.END, message + '\n')
        self.debug_console.see(tk.END)
        self.debug_console.config(state=tk.DISABLED)

    def clear_debug_log(self):
        if not getattr(self, 'debug_console', None):
            return
        self.debug_console.config(state=tk.NORMAL)
        self.debug_console.delete('1.0', tk.END)
        self.debug_console.config(state=tk.DISABLED)
    
    # === GRBL回调函数 ===
    def on_grbl_status(self, status):
        self.root.after(0, lambda: self.conn_status_label.config(
            text=f"已连接 - {status}", 
            foreground='green' if status == 'Idle' else 'orange'
        ))
    
    def on_grbl_response(self, response):
        self.root.after(0, lambda: self.log(response))
    
    def on_grbl_progress(self, progress):
        current, total = progress
        percent = (current / total * 100) if total > 0 else 0
        self.root.after(0, lambda: self._update_progress(percent, current, total))
    
    def _update_progress(self, percent, current, total):
        self.progress_var.set(percent)
        self.progress_label.config(text=f"{current} / {total}")
    
    def on_grbl_position(self, position):
        self.root.after(0, lambda: self._update_position(position))
    
    def _update_position(self, pos):
        self.pos_x_label.config(text=f"X: {pos['X']:.3f}")
        self.pos_y_label.config(text=f"Y: {pos['Y']:.3f}")
        self.pos_z_label.config(text=f"Z: {pos['Z']:.3f}")
    
    # === 连接控制 ===
    def refresh_ports(self):
        ports = GRBLController.list_ports()
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.set(ports[0])
    
    def toggle_connection(self):
        if self.grbl.connected:
            self.grbl.disconnect()
            self.connect_btn.config(text="连接")
            self.conn_status_label.config(text="未连接", foreground='red')
            self.log("已断开连接")
        else:
            port = self.port_var.get()
            baud = int(self.baud_var.get())
            if not port:
                messagebox.showwarning("警告", "请选择串口")
                return
            try:
                self.grbl.connect(port, baud)
                self.connect_btn.config(text="断开")
                self.conn_status_label.config(text="已连接", foreground='green')
                self.log(f"已连接到 {port} @ {baud}")
            except Exception as e:
                messagebox.showerror("连接失败", str(e))
    
    # === 手动控制 ===
    def jog(self, axis, direction):
        if not self.grbl.connected:
            messagebox.showwarning("警告", "请先连接设备")
            return
        try:
            step = float(self.jog_step_var.get()) * direction
            speed = int(self.jog_speed_var.get())
            self.grbl.jog(axis, step, speed)
        except ValueError:
            messagebox.showerror("错误", "请输入有效的步进值")
    
    def go_home(self):
        if self.grbl.connected:
            self.grbl.send_command("G0 X0 Y0")
    
    def pen_up_action(self):
        if self.grbl.connected:
            try:
                z_up = float(self.z_up_var.get())
                self.grbl.send_command(f"G0 Z{z_up}")
            except ValueError:
                pass
    
    def pen_down_action(self):
        if self.grbl.connected:
            try:
                z_down = float(self.z_down_var.get())
                self.grbl.send_command(f"G1 Z{z_down} F500")
            except ValueError:
                pass
    
    def emergency_stop(self):
        self.grbl.emergency_stop()
        self.log("[急停] 已发送急停命令")
    
    def send_manual_command(self):
        cmd = self.manual_cmd_var.get().strip()
        if cmd and self.grbl.connected:
            self.grbl.send_command(cmd)
            self.manual_cmd_var.set("")
    
    def open_grbl_settings(self):
        """打开GRBL设置窗口"""
        if not self.grbl.connected:
            messagebox.showwarning("警告", "请先连接设备")
            return
        
        # 创建设置窗口
        settings_win = tk.Toplevel(self.root)
        settings_win.title("GRBL设置")
        settings_win.geometry("500x600")
        settings_win.transient(self.root)
        
        # 创建滚动框架
        canvas = tk.Canvas(settings_win)
        scrollbar = ttk.Scrollbar(settings_win, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 存储输入框变量
        setting_vars = {}
        
        # 常用设置分组
        groups = {
            "运动参数": [100, 101, 102, 110, 111, 112, 120, 121, 122],
            "行程限制": [130, 131, 132, 20, 21],
            "归位设置": [22, 23, 24, 25, 26, 27],
            "主轴/激光": [30, 31, 32],
            "基础设置": [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13],
        }
        
        for group_name, params in groups.items():
            group_frame = ttk.LabelFrame(scrollable_frame, text=group_name)
            group_frame.pack(fill=tk.X, padx=10, pady=5)
            
            for param in params:
                if param in GRBL_SETTINGS:
                    name, unit, desc = GRBL_SETTINGS[param]
                    
                    row_frame = ttk.Frame(group_frame)
                    row_frame.pack(fill=tk.X, padx=5, pady=2)
                    
                    ttk.Label(row_frame, text=f"${param}", width=5).pack(side=tk.LEFT)
                    ttk.Label(row_frame, text=name, width=15).pack(side=tk.LEFT)
                    
                    var = tk.StringVar()
                    entry = ttk.Entry(row_frame, textvariable=var, width=12)
                    entry.pack(side=tk.LEFT, padx=5)
                    setting_vars[param] = var
                    
                    ttk.Label(row_frame, text=unit, width=10).pack(side=tk.LEFT)
        
        # 按钮区
        btn_frame = ttk.Frame(settings_win)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 设置回调来更新输入框
        def on_setting_received(data):
            param, value = data
            if param in setting_vars:
                self.root.after(0, lambda p=param, v=value: setting_vars[p].set(v))
        
        # 保存旧的回调
        old_setting_callback = self.grbl.callbacks.get('setting')
        self.grbl.callbacks['setting'] = on_setting_received
        
        def read_settings():
            """读取当前设置"""
            # 清空现有值
            for var in setting_vars.values():
                var.set("")
            self.grbl.get_settings()
        
        def apply_settings():
            """应用修改的设置"""
            for param, var in setting_vars.items():
                value = var.get().strip()
                if value:
                    self.grbl.set_setting(param, value)
        
        def on_close():
            """关闭窗口时恢复回调"""
            self.grbl.callbacks['setting'] = old_setting_callback
            canvas.unbind_all("<MouseWheel>")
            settings_win.destroy()
        
        ttk.Button(btn_frame, text="读取当前设置 ($$)", command=read_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="应用修改", command=apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=on_close).pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(btn_frame, text="提示: 先读取设置，修改后点应用", foreground='blue').pack(side=tk.LEFT, padx=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 绑定鼠标滚轮
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        settings_win.protocol("WM_DELETE_WINDOW", on_close)
    
    # === GCode发送 ===
    def _update_stream_buttons(self, state):
        """更新发送按钮状态"""
        self.stream_state = state
        if state == 'idle':
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.resume_btn.config(state=tk.DISABLED)
            self.end_btn.config(state=tk.DISABLED)
        elif state == 'streaming':
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.resume_btn.config(state=tk.DISABLED)
            self.end_btn.config(state=tk.NORMAL)
        elif state == 'paused':
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.DISABLED)
            self.resume_btn.config(state=tk.NORMAL)
            self.end_btn.config(state=tk.NORMAL)
    
    def start_streaming(self):
        """开始发送（总是重新生成GCode）"""
        if not self.grbl.connected:
            messagebox.showwarning("警告", "请先连接设备")
            return
        
        # 如果有轮廓，总是重新生成GCode（使用最新位置参数）
        if self.contours:
            ok = self.generate_gcode_from_contours()
            if not ok:
                return
        elif not self.gcode_lines:
            messagebox.showwarning("警告", "没有GCode可发送，请先处理图像或加载GCode文件")
            return
        
        # 重置进度
        self.grbl.stream_progress = 0
        self._update_stream_buttons('streaming')
        self.log("=== 开始发送GCode ===")
        self.grbl.stream_gcode(self.gcode_lines, on_complete=self._on_stream_complete)
    
    def pause_streaming(self):
        """暂停发送"""
        self.grbl.stop_stream()
        self._update_stream_buttons('paused')
        self.log(f"=== 已暂停（在第{self.grbl.stream_progress}行）===")
    
    def resume_streaming(self):
        """从上次停止的位置继续发送"""
        if not self.grbl.connected:
            messagebox.showwarning("警告", "请先连接设备")
            return
        
        if not self.gcode_lines:
            messagebox.showwarning("警告", "没有GCode可发送")
            return
        
        # 从上次停止的位置继续
        start_line = self.grbl.stream_progress
        if start_line >= len(self.gcode_lines):
            messagebox.showinfo("提示", "已发送完成")
            self._update_stream_buttons('idle')
            return
        
        remaining_lines = self.gcode_lines[start_line:]
        self._update_stream_buttons('streaming')
        self.log(f"=== 继续发送（从第{start_line + 1}行）===")
        self.grbl.stream_gcode(remaining_lines, on_complete=self._on_stream_complete)
    
    def end_streaming(self):
        """结束发送（完全停止并重置）"""
        self.grbl.stop_stream()
        self.grbl.stream_progress = 0
        self._update_stream_buttons('idle')
        self.progress_var.set(0)
        self.progress_label.config(text="0 / 0")
        self.log("=== 已结束发送 ===")
    
    def _on_stream_complete(self):
        """发送完成回调"""
        self.root.after(0, lambda: self._update_stream_buttons('idle'))
    
    def stop_streaming(self):
        """兼容旧方法"""
        self.pause_streaming()
    
    def generate_gcode_from_contours(self):
        """从轮廓生成GCode"""
        try:
            offset_x = float(self.offset_x_var.get())
            offset_y = float(self.offset_y_var.get())
            scale = float(self.scale_var.get()) * 0.1
            feed_rate = int(self.feed_rate_var.get())
            z_up = float(self.z_up_var.get())
            z_down = float(self.z_down_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效的参数")
            return False

        if scale == 0:
            messagebox.showerror("错误", "缩放比例不能为0")
            return False
        
        # 获取高度补偿参数
        z_compensation = {'enabled': False}
        if self.z_comp_enabled_var.get():
            try:
                z_compensation = {
                    'enabled': True,
                    'left': float(self.z_left_var.get()),    # 左侧
                    'center': float(self.z_center_var.get()), # 中间
                    'right': float(self.z_right_var.get()),   # 右侧
                    'curve': float(self.z_curve_var.get())    # 曲率
                }
            except ValueError:
                messagebox.showerror("错误", "请输入有效的高度补偿参数")
                return False
        
        gcode_gen = GCodeGenerator(
            feed_rate, z_up, z_down, 
            z_compensation=z_compensation,
            paper_width=self.paper_width,
            paper_height=self.paper_height
        )
        contours_to_export = self.contours
        image_rotation = getattr(self, 'image_rotation', 0)
        if image_rotation != 0:
            all_x = []
            all_y = []
            for contour in self.contours:
                for point in contour:
                    all_x.append(point[0] * scale)
                    all_y.append(point[1] * scale)
            if all_x and all_y:
                center_x = (min(all_x) + max(all_x)) / 2 + offset_x
                center_y = (min(all_y) + max(all_y)) / 2 + offset_y
            else:
                center_x = offset_x
                center_y = offset_y
            
            rotated_contours = []
            for contour in self.contours:
                rotated_contour = []
                for point in contour:
                    mm_x = point[0] * scale + offset_x
                    mm_y = point[1] * scale + offset_y
                    rx, ry = self._rotate_point(mm_x, mm_y, center_x, center_y, image_rotation)
                    rotated_contour.append(((rx - offset_x) / scale, (ry - offset_y) / scale))
                rotated_contours.append(rotated_contour)
            contours_to_export = rotated_contours
        
        self.gcode_lines = gcode_gen.generate_from_contours(
            contours_to_export, offset_x, offset_y, scale, self.paper_height
        )
        
        comp_status = "（已启用高度补偿）" if z_compensation['enabled'] else ""
        self.status_label.config(text=f"已生成 {len(self.gcode_lines)} 行GCode{comp_status}")
        return True
    
    def load_gcode_file(self):
        """加载GCode文件"""
        filetypes = [('GCode文件', '*.gcode *.nc *.ngc *.txt'), ('所有文件', '*.*')]
        path = filedialog.askopenfilename(title="选择GCode文件", filetypes=filetypes)
        
        if path:
            try:
                with open(path, 'r') as f:
                    self.gcode_lines = f.readlines()
                self.status_label.config(text=f"已加载 {len(self.gcode_lines)} 行GCode")
                self.log(f"已加载GCode文件: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("错误", f"加载失败: {e}")
    
    def save_gcode(self):
        """导出GCode文件"""
        if self.contours:
            ok = self.generate_gcode_from_contours()
            if not ok:
                return
        
        if not self.gcode_lines:
            messagebox.showwarning("警告", "没有GCode可导出，请先处理图像")
            return
        
        filetypes = [('GCode文件', '*.gcode'), ('NC文件', '*.nc'), ('所有文件', '*.*')]
        path = filedialog.asksaveasfilename(
            title="导出GCode", 
            defaultextension=".gcode",
            filetypes=filetypes
        )
        
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    for line in self.gcode_lines:
                        f.write(line if line.endswith('\n') else line + '\n')
                self.status_label.config(text=f"已导出: {os.path.basename(path)}")
                self.log(f"已导出GCode: {path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {e}")
    
    # === 图像处理（保持原有功能） ===
    def load_image(self):
        filetypes = [('图片文件', '*.png *.jpg *.jpeg *.bmp *.gif *.tiff'), ('所有文件', '*.*')]
        path = filedialog.askopenfilename(title="选择图片", filetypes=filetypes)
        
        if path:
            self.image_path = path
            self.original_image = ImageProcessor.load_image(path)
            self.crop_region = None
            self.processed_image = None
            self.contours = []
            self.gcode_lines = []
            
            # 重置原图旋转角度
            self.source_rotation = 0
            self.source_rotation_var.set(0)
            self.source_rotation_label.config(text="0°")
            
            if self.original_image is not None:
                self.status_label.config(text=f"已加载: {os.path.basename(path)}", foreground='green')
                self.display_original_image()
                self.notebook.select(0)
            else:
                messagebox.showerror("错误", "无法加载图片")
    
    def display_original_image(self):
        if self.original_image is None:
            return
        
        self.original_canvas.delete('all')
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            self.root.after(100, self.display_original_image)
            return
        
        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 应用原图旋转
        if self.source_rotation != 0:
            pil_img = pil_img.rotate(-self.source_rotation, expand=True, resample=Image.Resampling.BICUBIC)
        
        img_width, img_height = pil_img.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * 0.9
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.original_photo = ImageTk.PhotoImage(pil_img)
        
        self.original_display_scale = scale
        self.original_display_offset_x = (canvas_width - new_width) // 2
        self.original_display_offset_y = (canvas_height - new_height) // 2
        
        # 保存旋转后的图像尺寸
        self.rotated_img_width = img_width
        self.rotated_img_height = img_height
        
        self.original_canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.original_photo, anchor=tk.CENTER, tags='image'
        )
        
        if self.crop_region:
            self.draw_selection_rect()
    
    def start_selection_mode(self):
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return
        
        self.selection_mode = True
        self.selection_start = None
        self.crop_region = None
        self.notebook.select(0)
        self.status_label.config(text="框选模式: 拖动选择区域", foreground='blue')
    
    def on_original_canvas_click(self, event):
        if self.selection_mode and self.original_image is not None:
            self.selection_start = (event.x, event.y)
            self.original_canvas.delete('selection')
    
    def on_original_canvas_drag(self, event):
        if self.selection_mode and self.selection_start:
            self.original_canvas.delete('selection')
            x1, y1 = self.selection_start
            x2, y2 = event.x, event.y
            self.original_canvas.create_rectangle(
                x1, y1, x2, y2, 
                outline='red', width=2, dash=(5, 5), tags='selection'
            )
    
    def on_original_canvas_release(self, event):
        if self.selection_mode and self.selection_start:
            x1, y1 = self.selection_start
            x2, y2 = event.x, event.y
            
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            
            img_x1 = int((x1 - self.original_display_offset_x) / self.original_display_scale)
            img_y1 = int((y1 - self.original_display_offset_y) / self.original_display_scale)
            img_x2 = int((x2 - self.original_display_offset_x) / self.original_display_scale)
            img_y2 = int((y2 - self.original_display_offset_y) / self.original_display_scale)
            
            # 使用旋转后图像的尺寸
            w = getattr(self, 'rotated_img_width', self.original_image.shape[1])
            h = getattr(self, 'rotated_img_height', self.original_image.shape[0])
            img_x1 = max(0, min(img_x1, w))
            img_x2 = max(0, min(img_x2, w))
            img_y1 = max(0, min(img_y1, h))
            img_y2 = max(0, min(img_y2, h))
            
            if img_x2 - img_x1 > 10 and img_y2 - img_y1 > 10:
                self.crop_region = (img_x1, img_y1, img_x2, img_y2)
                self.selection_label.config(text=f"({img_x1},{img_y1})-({img_x2},{img_y2})")
                self.status_label.config(text="已选择区域", foreground='green')
            else:
                self.selection_label.config(text="区域太小")
            
            self.selection_start = None
    
    def draw_selection_rect(self):
        if not self.crop_region:
            return
        
        x1, y1, x2, y2 = self.crop_region
        canvas_x1 = x1 * self.original_display_scale + self.original_display_offset_x
        canvas_y1 = y1 * self.original_display_scale + self.original_display_offset_y
        canvas_x2 = x2 * self.original_display_scale + self.original_display_offset_x
        canvas_y2 = y2 * self.original_display_scale + self.original_display_offset_y
        
        self.original_canvas.delete('selection')
        self.original_canvas.create_rectangle(
            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
            outline='red', width=2, dash=(5, 5), tags='selection'
        )
    
    def confirm_selection(self):
        if self.crop_region and self.original_image is not None:
            x1, y1, x2, y2 = self.crop_region
            # 使用旋转后的图像进行裁剪
            rotated_img = self._get_rotated_image()
            self.processed_image = ImageProcessor.crop_image(rotated_img, x1, y1, x2, y2)
            self.selection_mode = False
            self.status_label.config(text="区域已确认", foreground='green')
        else:
            messagebox.showwarning("警告", "请先框选区域")
    
    def reset_selection(self):
        self.crop_region = None
        self.selection_mode = False
        self.processed_image = None
        self.original_canvas.delete('selection')
        self.selection_label.config(text="未选择")
        self.status_label.config(text="已重置", foreground='green')
    
    def _on_source_rotation_change(self, value):
        """原图旋转滑块变化"""
        angle = int(float(value))
        self.source_rotation = angle
        self.source_rotation_label.config(text=f"{angle}°")
        # 旋转后重置选区
        self.crop_region = None
        self.selection_label.config(text="未选择")
        self.display_original_image()
    
    def _rotate_source(self, angle, absolute=False):
        """旋转原图"""
        if absolute:
            self.source_rotation = angle
        else:
            self.source_rotation = (self.source_rotation + angle) % 360
            if self.source_rotation > 180:
                self.source_rotation -= 360
        self.source_rotation_var.set(self.source_rotation)
        self.source_rotation_label.config(text=f"{self.source_rotation}°")
        # 旋转后重置选区
        self.crop_region = None
        self.selection_label.config(text="未选择")
        self.display_original_image()
    
    def _get_rotated_image(self):
        """获取应用原图旋转后的图像"""
        if self.original_image is None:
            return None
        
        if self.source_rotation == 0:
            return self.original_image.copy()
        
        # 使用PIL旋转图像
        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        rotated_pil = pil_img.rotate(-self.source_rotation, expand=True, resample=Image.Resampling.BICUBIC)
        rotated_rgb = np.array(rotated_pil)
        rotated_bgr = cv2.cvtColor(rotated_rgb, cv2.COLOR_RGB2BGR)
        return rotated_bgr

    def process_image(self):
        # 判断要处理的图像
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return
        
        # 获取旋转后的原图
        rotated_img = self._get_rotated_image()
        if rotated_img is None:
            messagebox.showerror("错误", "无法获取旋转后的图像")
            return

        rotated_h, rotated_w = rotated_img.shape[:2]
        
        # 记录框选区域的偏移量（用于坐标转换）
        offset_x_px = 0
        offset_y_px = 0
        
        # 如果有框选区域，自动裁剪该区域
        if self.crop_region:
            x1, y1, x2, y2 = self.crop_region
            img_to_process = ImageProcessor.crop_image(rotated_img, x1, y1, x2, y2)
            offset_x_px = x1  # 记录偏移
            offset_y_px = y1
        else:
            # 没有框选区域，处理整个图像
            img_to_process = rotated_img

        crop_flag = bool(self.crop_region)
        img_h, img_w = img_to_process.shape[:2]
        self.debug_log(f"处理图像: rotated=({rotated_w}x{rotated_h}) crop={crop_flag} region={self.crop_region} img=({img_w}x{img_h}) ocr={self.ocr_mode_var.get()}")
        
        try:
            # 检查是否启用OCR模式
            if self.ocr_mode_var.get():
                # OCR模式：识别文字并转换为骨架图
                skeleton_img = self._process_ocr_image(img_to_process)
                if skeleton_img is None:
                    return

                try:
                    nonzero = int(np.count_nonzero(skeleton_img))
                except:
                    nonzero = -1
                self.debug_log(f"OCR结果图: skeleton_nonzero={nonzero}")
                
                # 从骨架图提取轮廓
                contours_cv, _ = cv2.findContours(skeleton_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                
                simplify = self.simplify_var.get()
                new_contours = []
                for contour in contours_cv:
                    if len(contour) < 2:
                        continue
                    
                    # 简化路径
                    epsilon = simplify * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, False)
                    
                    points = []
                    for point in approx:
                        # 加上框选区域的偏移量
                        points.append((point[0][0] + offset_x_px, point[0][1] + offset_y_px))
                    
                    if len(points) >= 2:
                        new_contours.append(points)

                new_points = sum(len(c) for c in new_contours)
                total_points = sum(len(c) for c in self.contours) + new_points
                self.debug_log(f"轮廓提取: new_contours={len(new_contours)} new_points={new_points} total_contours={len(self.contours)+len(new_contours)} total_points={total_points}")
                
                # 追加到现有轮廓
                self.contours.extend(new_contours)
                
                h, w = rotated_img.shape[:2]
                self.image_mm_width = w * 0.1
                self.image_mm_height = h * 0.1
                
                self.status_label.config(text=f"OCR识别完成, 新增 {len(new_contours)} 条路径, 共 {len(self.contours)} 条", foreground='green')
            else:
                # 普通模式：边缘检测
                low = self.low_threshold_var.get()
                high = self.high_threshold_var.get()
                simplify = self.simplify_var.get()
                
                edges = ImageProcessor.detect_edges(img_to_process, low, high)
                new_contours_raw = ImageProcessor.find_contours(edges, True, simplify)
                
                # 将新轮廓加上偏移量
                new_contours = []
                for contour in new_contours_raw:
                    adjusted_contour = [(x + offset_x_px, y + offset_y_px) for x, y in contour]
                    new_contours.append(adjusted_contour)
                
                # 追加到现有轮廓
                self.contours.extend(new_contours)
                
                h, w = rotated_img.shape[:2]
                self.image_mm_width = w * 0.1
                self.image_mm_height = h * 0.1
                
                self.status_label.config(text=f"新增 {len(new_contours)} 个轮廓, 共 {len(self.contours)} 个", foreground='green')
            
            self.center_contours_on_paper()
            self.gcode_lines = []  # 清空旧的GCode
            
            self.draw_paper()
            self.notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {str(e)}")

    def show_ocr_compare(self):
        input_bgr = getattr(self, "_last_ocr_input_bgr", None)
        output_skeleton = getattr(self, "_last_ocr_output_skeleton", None)
        if input_bgr is None or output_skeleton is None:
            messagebox.showwarning("提示", "请先在 OCR 模式下点击“处理图像”，再打开对比。")
            return

        win = tk.Toplevel(self.root)
        win.title("原图 vs OCR结果")
        win.geometry("1200x700")

        toolbar = ttk.Frame(win)
        toolbar.pack(fill=tk.X, padx=8, pady=6)

        scale_var = tk.DoubleVar(value=1.0)
        info_var = tk.StringVar(value="")

        def _clamp_scale(v):
            try:
                v = float(v)
            except Exception:
                v = 1.0
            return max(0.05, min(8.0, v))

        def _np_bgr_to_pil_rgb(arr_bgr):
            rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

        def _np_skeleton_to_pil(arr):
            a = arr
            if a is None:
                return None
            if len(a.shape) == 3:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            a = a.astype(np.uint8)
            return Image.fromarray(255 - a, mode="L")

        left_src = _np_bgr_to_pil_rgb(input_bgr)
        right_src = _np_skeleton_to_pil(output_skeleton)

        panes = ttk.Panedwindow(win, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left_frame = ttk.Frame(panes)
        right_frame = ttk.Frame(panes)
        panes.add(left_frame, weight=1)
        panes.add(right_frame, weight=1)

        def _make_view(parent, title):
            header = ttk.Frame(parent)
            header.pack(fill=tk.X)
            ttk.Label(header, text=title).pack(side=tk.LEFT)

            body = ttk.Frame(parent)
            body.pack(fill=tk.BOTH, expand=True)

            ybar = ttk.Scrollbar(body, orient=tk.VERTICAL)
            xbar = ttk.Scrollbar(body, orient=tk.HORIZONTAL)
            canvas = tk.Canvas(body, bg="white", highlightthickness=0, yscrollcommand=ybar.set, xscrollcommand=xbar.set)
            ybar.config(command=canvas.yview)
            xbar.config(command=canvas.xview)

            ybar.pack(side=tk.RIGHT, fill=tk.Y)
            xbar.pack(side=tk.BOTTOM, fill=tk.X)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            return canvas

        left_canvas = _make_view(left_frame, "原图（OCR输入）")
        right_canvas = _make_view(right_frame, "识别后（骨架结果）")

        def _render():
            s = _clamp_scale(scale_var.get())
            scale_var.set(s)

            lw, lh = left_src.size
            rw, rh = right_src.size

            left_img = left_src.resize((max(1, int(lw * s)), max(1, int(lh * s))), Image.Resampling.NEAREST)
            right_img = right_src.resize((max(1, int(rw * s)), max(1, int(rh * s))), Image.Resampling.NEAREST)

            left_photo = ImageTk.PhotoImage(left_img)
            right_photo = ImageTk.PhotoImage(right_img)

            left_canvas.delete("all")
            right_canvas.delete("all")

            left_canvas.create_image(0, 0, image=left_photo, anchor=tk.NW)
            right_canvas.create_image(0, 0, image=right_photo, anchor=tk.NW)

            left_canvas.image = left_photo
            right_canvas.image = right_photo

            left_canvas.config(scrollregion=(0, 0, left_img.width, left_img.height))
            right_canvas.config(scrollregion=(0, 0, right_img.width, right_img.height))

            nonzero = int(np.count_nonzero(output_skeleton)) if output_skeleton is not None else 0
            info_var.set(f"缩放 {s:.2f} | 原图 {lw}×{lh} | 结果 {rw}×{rh} | 骨架非零 {nonzero}")

        def _zoom_by(factor):
            scale_var.set(_clamp_scale(scale_var.get() * factor))
            _render()

        def _fit():
            self.root.update_idletasks()
            cw = max(1, min(left_canvas.winfo_width(), right_canvas.winfo_width()))
            ch = max(1, min(left_canvas.winfo_height(), right_canvas.winfo_height()))
            lw, lh = left_src.size
            rw, rh = right_src.size
            s1 = min(cw / max(1, lw), ch / max(1, lh))
            s2 = min(cw / max(1, rw), ch / max(1, rh))
            scale_var.set(_clamp_scale(min(s1, s2)))
            _render()

        ttk.Button(toolbar, text="缩小", width=8, command=lambda: _zoom_by(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="放大", width=8, command=lambda: _zoom_by(1.25)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="适配", width=8, command=_fit).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="1:1", width=8, command=lambda: (scale_var.set(1.0), _render())).pack(side=tk.LEFT, padx=2)
        ttk.Label(toolbar, textvariable=info_var).pack(side=tk.RIGHT)

        def _wheel(event):
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return
            _zoom_by(1.1 if delta > 0 else 0.9)

        left_canvas.bind("<MouseWheel>", _wheel)
        right_canvas.bind("<MouseWheel>", _wheel)

        win.after(50, _fit)
    
    def process_engineering_thinning(self):
        """工程化单线化处理：将图像转换为单像素细线"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return
        
        try:
            from skimage.morphology import skeletonize, thin
            from skimage import img_as_ubyte
        except ImportError:
            messagebox.showerror("错误", "请安装 scikit-image: pip install scikit-image")
            return
        
        try:
            if self.contours:
                self.contours = []
                self.gcode_lines = []
                self.selected_object = None
                self.contour_bounds = None
                self.image_rotation = 0
                self.image_rotation_var.set(0)
                self.image_rotation_label.config(text="0°")

            # 获取旋转后的原图
            rotated_img = self._get_rotated_image()
            
            # 记录框选区域的偏移量
            offset_x_px = 0
            offset_y_px = 0
            
            # 如果有框选区域，自动裁剪该区域
            if self.crop_region:
                x1, y1, x2, y2 = self.crop_region
                img_to_process = ImageProcessor.crop_image(rotated_img, x1, y1, x2, y2)
                offset_x_px = x1
                offset_y_px = y1
            else:
                img_to_process = rotated_img
            
            h, w = img_to_process.shape[:2]
            
            # 获取参数
            text_sensitivity = self.text_sensitivity_var.get() / 100.0  # 0-1
            denoise_strength = self.denoise_var.get()
            min_contour_len = self.min_contour_len_var.get()
            simplify = self.simplify_var.get()
            
            self.status_label.config(text="正在预处理图像...", foreground='blue')
            self.root.update()
            
            # === 步骤1: 图像预处理 ===
            gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY) if len(img_to_process.shape) == 3 else img_to_process.copy()
            
            # 去噪处理
            if denoise_strength > 0:
                # 使用双边滤波保留边缘同时去噪
                gray = cv2.bilateralFilter(gray, 9, denoise_strength * 10, denoise_strength * 10)
            
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # === 步骤3: 检测文字区域（可选） ===
            text_mask = None
            if text_sensitivity > 0 and self._ocr_is_available():
                self.status_label.config(text="正在检测文字区域...", foreground='blue')
                self.root.update()
                text_mask = self._detect_text_regions(img_to_process, text_sensitivity)
            
            # === 步骤4: 分区域处理 ===
            self.status_label.config(text="正在进行单线化处理...", foreground='blue')
            self.root.update()
            
            # 创建输出图像
            skeleton_result = np.zeros_like(binary)

            if text_mask is not None:
                text_region = cv2.bitwise_and(binary, text_mask)
                if np.any(text_region > 0):
                    text_skeleton = skeletonize(text_region > 0)
                    skeleton_result = skeleton_result | (text_skeleton.astype(np.uint8) * 255)

                non_text_region = cv2.bitwise_and(binary, cv2.bitwise_not(text_mask))
                if np.any(non_text_region > 0):
                    non_text_thinned = thin(non_text_region > 0)
                    skeleton_result = skeleton_result | (non_text_thinned.astype(np.uint8) * 255)
            else:
                thinned = thin(binary > 0)
                skeleton_result = (thinned.astype(np.uint8) * 255)
            
            # === 步骤5: 清理和优化 ===
            self.status_label.config(text="正在优化轮廓...", foreground='blue')
            self.root.update()
            
            # 移除孤立点和短线段
            skeleton_result = self._remove_small_components(skeleton_result, min_contour_len)
            
            # === 步骤6: 提取轮廓 ===
            self.status_label.config(text="正在提取路径...", foreground='blue')
            self.root.update()
            
            # 使用连通域分析提取路径
            contours_cv, _ = cv2.findContours(skeleton_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            new_contours = []
            for contour in contours_cv:
                if len(contour) < 2:
                    continue
                
                # 简化路径
                epsilon = simplify * cv2.arcLength(contour, False)
                approx = cv2.approxPolyDP(contour, epsilon, False)
                
                points = []
                for point in approx:
                    # 加上框选区域的偏移量
                    points.append((point[0][0] + offset_x_px, point[0][1] + offset_y_px))
                
                if len(points) >= 2:
                    new_contours.append(points)
            
            # 追加到现有轮廓
            self.contours.extend(new_contours)
            
            # 更新图像尺寸信息
            h_full, w_full = rotated_img.shape[:2]
            self.image_mm_width = w_full * 0.1
            self.image_mm_height = h_full * 0.1
            
            self.center_contours_on_paper()
            self.gcode_lines = []  # 清空旧的GCode
            
            self.status_label.config(
                text=f"单线化完成, 新增 {len(new_contours)} 条路径, 共 {len(self.contours)} 条", 
                foreground='green'
            )
            
            self.draw_paper()
            self.notebook.select(1)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"单线化处理失败: {str(e)}")

    def process_algo_medial_axis(self):
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return

        try:
            from skimage.morphology import medial_axis
        except ImportError:
            messagebox.showerror("错误", "请安装 scikit-image: pip install scikit-image")
            return

        try:
            if self.contours:
                self.contours = []
                self.gcode_lines = []
                self.selected_object = None
                self.contour_bounds = None
                self.image_rotation = 0
                self.image_rotation_var.set(0)
                self.image_rotation_label.config(text="0°")

            rotated_img = self._get_rotated_image()

            offset_x_px = 0
            offset_y_px = 0
            if self.crop_region:
                x1, y1, x2, y2 = self.crop_region
                img_to_process = ImageProcessor.crop_image(rotated_img, x1, y1, x2, y2)
                offset_x_px = x1
                offset_y_px = y1
            else:
                img_to_process = rotated_img

            denoise_strength = int(self.ma_denoise_var.get())
            min_contour_len = int(self.ma_min_contour_len_var.get())
            simplify = float(self.ma_simplify_var.get())
            smooth_iters = int(self.ma_bspline_iter_var.get())
            do_opt = bool(self.ma_optimize_paths_var.get())

            self.status_label.config(text="算法一：预处理...", foreground='blue')
            self.root.update()

            gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY) if len(img_to_process.shape) == 3 else img_to_process.copy()
            if denoise_strength > 0:
                gray = cv2.bilateralFilter(gray, 9, denoise_strength * 10, denoise_strength * 10)

            binary = self._binarize_to_foreground(gray, 'adaptive', 128)

            self.status_label.config(text="算法一：Medial Axis...", foreground='blue')
            self.root.update()

            sk = medial_axis(binary > 0)
            skeleton_result = (sk.astype(np.uint8) * 255)
            skeleton_result = self._remove_small_components(skeleton_result, min_contour_len)

            self.status_label.config(text="算法一：提取路径...", foreground='blue')
            self.root.update()

            contours_cv, _ = cv2.findContours(skeleton_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            new_contours = []
            for contour in contours_cv:
                if len(contour) < 2:
                    continue
                epsilon = simplify * cv2.arcLength(contour, False)
                approx = cv2.approxPolyDP(contour, epsilon, False)
                points = []
                for point in approx:
                    points.append((point[0][0] + offset_x_px, point[0][1] + offset_y_px))
                if len(points) >= 2:
                    new_contours.append(points)

            if smooth_iters > 0:
                new_contours = [self._bspline_smooth_polyline(c, smooth_iters) for c in new_contours]
            if do_opt and len(new_contours) >= 2:
                new_contours = self._optimize_contours_order(new_contours)

            self.contours.extend(new_contours)

            h_full, w_full = rotated_img.shape[:2]
            self.image_mm_width = w_full * 0.1
            self.image_mm_height = h_full * 0.1

            self.center_contours_on_paper()
            self.gcode_lines = []

            self.status_label.config(text=f"算法一完成, 新增 {len(new_contours)} 条路径, 共 {len(self.contours)} 条", foreground='green')
            self.draw_paper()
            self.notebook.select(1)

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"算法一处理失败: {str(e)}")

    def process_algo_hough_replace(self):
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return

        try:
            from skimage.morphology import skeletonize
        except ImportError:
            messagebox.showerror("错误", "请安装 scikit-image: pip install scikit-image")
            return

        try:
            if self.contours:
                self.contours = []
                self.gcode_lines = []
                self.selected_object = None
                self.contour_bounds = None
                self.image_rotation = 0
                self.image_rotation_var.set(0)
                self.image_rotation_label.config(text="0°")

            rotated_img = self._get_rotated_image()

            offset_x_px = 0
            offset_y_px = 0
            if self.crop_region:
                x1, y1, x2, y2 = self.crop_region
                img_to_process = ImageProcessor.crop_image(rotated_img, x1, y1, x2, y2)
                offset_x_px = x1
                offset_y_px = y1
            else:
                img_to_process = rotated_img

            denoise_strength = int(self.hough2_denoise_var.get())
            bin_method = str(self.hough2_binarize_method_var.get())
            fixed_th = int(self.hough2_fixed_threshold_var.get())
            min_contour_len = int(self.hough2_min_contour_len_var.get())
            min_line_len = int(self.hough2_min_len_var.get())
            max_gap = int(self.hough2_max_gap_var.get())
            simplify = float(self.hough2_simplify_var.get())
            smooth_iters = int(self.hough2_bspline_iter_var.get())
            do_opt = bool(self.hough2_optimize_paths_var.get())

            self.status_label.config(text="算法二：预处理与二值化...", foreground='blue')
            self.root.update()

            gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY) if len(img_to_process.shape) == 3 else img_to_process.copy()
            if denoise_strength > 0:
                gray = cv2.bilateralFilter(gray, 9, denoise_strength * 10, denoise_strength * 10)

            binary = self._binarize_to_foreground(gray, bin_method, fixed_th)

            self.status_label.config(text="算法二：骨架化...", foreground='blue')
            self.root.update()

            sk = skeletonize(binary > 0)
            skeleton_result = (sk.astype(np.uint8) * 255)
            skeleton_result = self._remove_small_components(skeleton_result, min_contour_len)

            self.status_label.config(text="算法二：Hough直线替换...", foreground='blue')
            self.root.update()

            skeleton_result, _ = self._replace_skeleton_by_hough_lines(
                skeleton_result,
                min_line_length=max(1, min_line_len),
                max_line_gap=max(0, max_gap),
            )

            self.status_label.config(text="算法二：提取路径...", foreground='blue')
            self.root.update()

            contours_cv, _ = cv2.findContours(skeleton_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            new_contours = []
            for contour in contours_cv:
                if len(contour) < 2:
                    continue
                epsilon = simplify * cv2.arcLength(contour, False)
                approx = cv2.approxPolyDP(contour, epsilon, False)
                points = []
                for point in approx:
                    points.append((point[0][0] + offset_x_px, point[0][1] + offset_y_px))
                if len(points) >= 2:
                    new_contours.append(points)

            if smooth_iters > 0:
                new_contours = [self._smooth_curves_preserve_lines(c, smooth_iters) for c in new_contours]
            if do_opt and len(new_contours) >= 2:
                new_contours = self._optimize_contours_order(new_contours)

            self.contours.extend(new_contours)

            h_full, w_full = rotated_img.shape[:2]
            self.image_mm_width = w_full * 0.1
            self.image_mm_height = h_full * 0.1

            self.center_contours_on_paper()
            self.gcode_lines = []

            self.status_label.config(text=f"算法二完成, 新增 {len(new_contours)} 条路径, 共 {len(self.contours)} 条", foreground='green')
            self.draw_paper()
            self.notebook.select(1)

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"算法二处理失败: {str(e)}")

    def process_algo_potrace(self):
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return

        try:
            if self.contours:
                self.contours = []
                self.gcode_lines = []
                self.selected_object = None
                self.contour_bounds = None
                self.image_rotation = 0
                self.image_rotation_var.set(0)
                self.image_rotation_label.config(text="0°")

            rotated_img = self._get_rotated_image()

            offset_x_px = 0
            offset_y_px = 0
            if self.crop_region:
                x1, y1, x2, y2 = self.crop_region
                img_to_process = ImageProcessor.crop_image(rotated_img, x1, y1, x2, y2)
                offset_x_px = x1
                offset_y_px = y1
            else:
                img_to_process = rotated_img

            denoise_strength = int(self.potrace_denoise_var.get())
            bin_method = str(self.potrace_binarize_method_var.get())
            fixed_th = int(self.potrace_fixed_threshold_var.get())
            simplify = float(self.potrace_simplify_var.get())
            min_contour_len = int(self.potrace_min_contour_len_var.get())
            angle_tol = float(self.potrace_line_angle_tol_var.get())
            dist_tol = float(self.potrace_line_dist_tol_var.get())

            self.status_label.config(text="算法三：预处理与二值化...", foreground='blue')
            self.root.update()

            gray = cv2.cvtColor(img_to_process, cv2.COLOR_BGR2GRAY) if len(img_to_process.shape) == 3 else img_to_process.copy()
            if denoise_strength > 0:
                gray = cv2.bilateralFilter(gray, 9, denoise_strength * 10, denoise_strength * 10)

            binary = self._binarize_to_foreground(gray, bin_method, fixed_th)
            if min_contour_len > 0:
                binary = self._remove_small_components(binary, min_contour_len)

            self.status_label.config(text="算法三：Potrace轮廓提取...", foreground='blue')
            self.root.update()

            contours_cv, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            new_contours = []
            for contour in contours_cv:
                if contour is None or len(contour) < 2:
                    continue
                if min_contour_len > 0 and cv2.arcLength(contour, True) < float(min_contour_len):
                    continue

                if simplify > 0:
                    eps = simplify * cv2.arcLength(contour, True)
                    contour2 = cv2.approxPolyDP(contour, eps, True)
                else:
                    contour2 = contour

                pts = [(int(p[0][0]), int(p[0][1])) for p in contour2]
                if len(pts) < 2:
                    continue

                if pts[0] != pts[-1]:
                    pts.append(pts[0])

                pts = self._line_constraint_polyline(pts, angle_tol_deg=angle_tol, dist_tol=dist_tol)
                pts = [(int(round(x + offset_x_px)), int(round(y + offset_y_px))) for x, y in pts]

                pts2 = []
                last = None
                for p in pts:
                    if last is None or p != last:
                        pts2.append(p)
                        last = p
                if len(pts2) >= 2:
                    new_contours.append(pts2)

            self.contours.extend(new_contours)

            h_full, w_full = rotated_img.shape[:2]
            self.image_mm_width = w_full * 0.1
            self.image_mm_height = h_full * 0.1

            self.center_contours_on_paper()
            self.gcode_lines = []

            self.status_label.config(text=f"算法三完成, 新增 {len(new_contours)} 条路径, 共 {len(self.contours)} 条", foreground='green')
            self.draw_paper()
            self.notebook.select(1)

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"算法三处理失败: {str(e)}")

    def process_algo_deepsvg_pix2vector(self):
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return

        try:
            if self.contours:
                self.contours = []
                self.gcode_lines = []
                self.selected_object = None
                self.contour_bounds = None
                self.image_rotation = 0
                self.image_rotation_var.set(0)
                self.image_rotation_label.config(text="0°")

            rotated_img = self._get_rotated_image()

            offset_x_px = 0
            offset_y_px = 0
            if self.crop_region:
                x1, y1, x2, y2 = self.crop_region
                img_to_process = ImageProcessor.crop_image(rotated_img, x1, y1, x2, y2)
                offset_x_px = x1
                offset_y_px = y1
            else:
                img_to_process = rotated_img

            backend = str(self.dsvg_backend_var.get())
            cmd_template = str(self.dsvg_command_var.get()).strip()
            sample_step = int(self.dsvg_sample_step_var.get())
            simplify = float(self.dsvg_simplify_var.get())
            min_contour_len = int(self.dsvg_min_contour_len_var.get())
            do_opt = bool(self.dsvg_optimize_paths_var.get())

            svg_path = None
            if cmd_template:
                self.status_label.config(text=f"算法四：{backend} 推理中...", foreground='blue')
                self.root.update()

                with tempfile.TemporaryDirectory() as tmpdir:
                    in_path = os.path.join(tmpdir, "input.png")
                    out_path = os.path.join(tmpdir, "output.svg")
                    cv2.imwrite(in_path, img_to_process)

                    cmd = cmd_template
                    cmd = cmd.replace("{in}", in_path).replace("{input}", in_path)
                    cmd = cmd.replace("{out}", out_path).replace("{output}", out_path)

                    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if p.returncode != 0:
                        msg = (p.stderr or "").strip()
                        if not msg:
                            msg = (p.stdout or "").strip()
                        messagebox.showerror("错误", f"算法四推理失败({backend})\n\n{msg or '未知错误'}")
                        return
                    if not os.path.exists(out_path):
                        messagebox.showerror("错误", f"未生成SVG输出文件: {out_path}")
                        return
                    svg_path = out_path

                    polylines_svg, viewbox = self._svg_load_polylines(svg_path, sample_step=max(1, sample_step))
            else:
                svg_path = filedialog.askopenfilename(
                    title="选择DeepSVG / Pix2Vector导出的SVG",
                    filetypes=[('SVG文件', '*.svg'), ('所有文件', '*.*')]
                )
                if not svg_path:
                    return
                self.status_label.config(text="算法四：解析SVG...", foreground='blue')
                self.root.update()
                polylines_svg, viewbox = self._svg_load_polylines(svg_path, sample_step=max(1, sample_step))

            if not polylines_svg:
                messagebox.showwarning("提示", "SVG中未解析到可用路径")
                return

            img_h, img_w = img_to_process.shape[:2]
            mapped = self._svg_map_polylines_to_image(polylines_svg, viewbox, img_w, img_h)
            mapped = [[(int(round(x + offset_x_px)), int(round(y + offset_y_px))) for x, y in poly] for poly in mapped]

            new_contours = []
            for pts in mapped:
                if not pts or len(pts) < 2:
                    continue
                pts2 = []
                last = None
                for p in pts:
                    if last is None or p != last:
                        pts2.append(p)
                        last = p
                if len(pts2) < 2:
                    continue
                if len(pts2) < min_contour_len:
                    continue

                if simplify > 0 and len(pts2) >= 3:
                    arr = np.array(pts2, dtype=np.int32).reshape(-1, 1, 2)
                    eps = simplify * cv2.arcLength(arr, False)
                    approx = cv2.approxPolyDP(arr, eps, False)
                    pts2 = [(int(p[0][0]), int(p[0][1])) for p in approx]
                    if len(pts2) < 2:
                        continue

                new_contours.append(pts2)

            if do_opt and len(new_contours) >= 2:
                new_contours = self._optimize_contours_order(new_contours)

            self.contours.extend(new_contours)

            h_full, w_full = rotated_img.shape[:2]
            self.image_mm_width = w_full * 0.1
            self.image_mm_height = h_full * 0.1

            self.center_contours_on_paper()
            self.gcode_lines = []

            self.status_label.config(text=f"算法四完成, 新增 {len(new_contours)} 条路径, 共 {len(self.contours)} 条", foreground='green')
            self.draw_paper()
            self.notebook.select(1)

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("错误", f"算法四处理失败: {str(e)}")
    
    def _ocr_is_available(self):
        if getattr(self, '_ocr_import_failed', False):
            return False
        try:
            __import__('rapidocr_onnxruntime', fromlist=['RapidOCR'])
            return True
        except ImportError:
            self._ocr_import_failed = True
            return False

    def _get_ocr_engine(self, show_error=True):
        if getattr(self, '_ocr_engine', None) is not None:
            return self._ocr_engine
        if getattr(self, '_ocr_engine_init_failed', False):
            return None

        RapidOCR = None
        try:
            mod = __import__('rapidocr_onnxruntime', fromlist=['RapidOCR'])
            RapidOCR = getattr(mod, 'RapidOCR', None)
        except ImportError:
            RapidOCR = None

        if RapidOCR is None:
            self._ocr_engine_init_failed = True
            if show_error:
                messagebox.showerror("OCR不可用", "未安装 RapidOCR，请运行:\n\npip install rapidocr-onnxruntime")
            return None

        try:
            self._ocr_engine = RapidOCR()
            return self._ocr_engine
        except Exception as e:
            import traceback
            self._ocr_engine_init_failed = True
            if show_error:
                messagebox.showerror(
                    "OCR初始化失败",
                    "RapidOCR 初始化失败：\n\n"
                    f"{type(e).__name__}: {e}\n\n"
                    "详细信息：\n"
                    f"{traceback.format_exc()}",
                )
            return None

    def _ocr_normalize_bbox(self, bbox):
        if bbox is None:
            return None
        if isinstance(bbox, np.ndarray):
            try:
                bbox = bbox.tolist()
            except:
                return None
        if not isinstance(bbox, (list, tuple)):
            return None

        if len(bbox) == 4 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in bbox):
            x1, y1, x2, y2 = bbox
            return [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]

        if len(bbox) == 8 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in bbox):
            return [[float(bbox[0]), float(bbox[1])], [float(bbox[2]), float(bbox[3])], [float(bbox[4]), float(bbox[5])], [float(bbox[6]), float(bbox[7])]]

        if len(bbox) != 4:
            return None

        pts = []
        for p in bbox:
            if isinstance(p, np.ndarray):
                try:
                    p = p.tolist()
                except:
                    return None
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                return None
            x, y = p[0], p[1]
            if not isinstance(x, (int, float, np.integer, np.floating)) or not isinstance(y, (int, float, np.integer, np.floating)):
                return None
            pts.append([float(x), float(y)])
        return pts

    def _ocr_extract_items(self, result):
        if not result:
            return []
        if isinstance(result, (tuple, list)) and len(result) == 2:
            result = result[0]
        if isinstance(result, np.ndarray):
            try:
                result = result.tolist()
            except:
                return []
        if not isinstance(result, list):
            return []

        items = []
        for it in result:
            if not isinstance(it, (list, tuple)) or len(it) < 1:
                continue
            bbox = self._ocr_normalize_bbox(it[0] if len(it) >= 1 else None)
            if bbox is None:
                continue
            text = ""
            if len(it) >= 2 and it[1] is not None:
                try:
                    text = str(it[1])
                except:
                    text = ""
            confidence = None
            if len(it) >= 3:
                v = it[2]
                if isinstance(v, (int, float, np.integer, np.floating)):
                    confidence = float(v)
                elif isinstance(v, str):
                    try:
                        confidence = float(v)
                    except:
                        confidence = None
            items.append((bbox, text, confidence))
        return items

    def _ocr_call(self, engine, img_bgr):
        try:
            return engine(img_bgr, use_det=True, use_cls=False, use_rec=True)
        except TypeError:
            return engine(img_bgr)

    def _detect_text_regions(self, img, sensitivity=0.5):
        """检测图像中的文字区域，返回文字区域的掩码"""
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            ocr = self._get_ocr_engine(show_error=False)
            if ocr is None:
                return mask
            
            h0, w0 = img.shape[:2]
            min_size = 2400
            scale_factor = 1.0
            img_for_ocr = img
            if max(h0, w0) < min_size:
                scale_factor = min_size / max(h0, w0)
                img_for_ocr = cv2.resize(
                    img,
                    (int(w0 * scale_factor), int(h0 * scale_factor)),
                    interpolation=cv2.INTER_CUBIC
                )

            if len(img_for_ocr.shape) == 2:
                img_for_ocr = cv2.cvtColor(img_for_ocr, cv2.COLOR_GRAY2BGR)
            elif img_for_ocr.shape[2] == 4:
                img_for_ocr = cv2.cvtColor(img_for_ocr, cv2.COLOR_BGRA2BGR)

            ocr_ret = self._ocr_call(ocr, img_for_ocr)

            ocr_items = self._ocr_extract_items(ocr_ret)
            if not ocr_items:
                return mask
            
            min_conf = max(0.0, min(1.0, 1.0 - float(sensitivity)))
            for bbox, text, confidence in ocr_items:
                try:
                    keep = False
                    if text and str(text).strip():
                        keep = True
                    if confidence is not None and confidence >= min_conf:
                        keep = True
                    if not keep:
                        continue

                    pts = (np.array(bbox, dtype=np.float32) / float(scale_factor)).astype(np.int32)
                    if pts.shape != (4, 2):
                        continue
                    cv2.fillPoly(mask, [pts], 255)
                except:
                    continue
        except:
            pass
        
        return mask
    
    def _remove_small_components(self, binary_img, min_size):
        """移除小于指定大小的连通域"""
        # 找到所有连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        
        # 创建输出图像
        result = np.zeros_like(binary_img)
        
        # 保留大于最小尺寸的连通域
        for i in range(1, num_labels):  # 跳过背景(0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_size:
                result[labels == i] = 255
        
        return result

    def _binarize_to_foreground(self, gray_img, method='adaptive', fixed_threshold=128):
        gray = gray_img
        if len(gray.shape) != 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        method = (method or 'adaptive').strip().lower()
        if method == 'otsu':
            _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cand = [b, 255 - b]
        elif method == 'fixed':
            th = int(max(0, min(255, fixed_threshold)))
            _, b = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
            cand = [b, 255 - b]
        else:
            b1 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            b2 = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            cand = [b1, b2]

        best = None
        best_ratio = None
        for c in cand:
            ratio = float(np.count_nonzero(c)) / float(c.size)
            if ratio <= 0.0005 or ratio >= 0.9995:
                continue
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best = c

        return best if best is not None else cand[-1]

    def _replace_skeleton_by_hough_lines(self, skeleton_img, min_line_length=60, max_line_gap=10, threshold=30, erase_radius=2):
        if skeleton_img is None:
            return skeleton_img, []
        bw = skeleton_img.copy()
        if bw.dtype != np.uint8:
            bw = bw.astype(np.uint8)
        bw = np.where(bw > 0, 255, 0).astype(np.uint8)

        lines = cv2.HoughLinesP(
            bw,
            rho=1,
            theta=np.pi / 180.0,
            threshold=int(max(1, threshold)),
            minLineLength=int(max(1, min_line_length)),
            maxLineGap=int(max(0, max_line_gap)),
        )

        if lines is None or len(lines) == 0:
            return bw, []

        out = bw.copy()
        thick_erase = int(max(1, erase_radius * 2 + 1))
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            mask = np.zeros_like(out)
            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thick_erase)
            out[mask > 0] = 0

        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)

        return out, lines.reshape(-1, 4).tolist()

    def _bspline_smooth_polyline(self, points, iterations=2):
        if not points or len(points) < 3 or iterations <= 0:
            return points
        pts = [(float(x), float(y)) for x, y in points]
        for _ in range(iterations):
            if len(pts) < 3:
                break
            if len(pts) > 2500:
                break
            out = [pts[0]]
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                qx, qy = 0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1
                rx, ry = 0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1
                out.append((qx, qy))
                out.append((rx, ry))
            out.append(pts[-1])
            pts = out
        return [(int(round(x)), int(round(y))) for x, y in pts]

    def _smooth_curves_preserve_lines(self, points, iterations=2, angle_tol_deg=8.0, dist_tol=1.2):
        if not points or len(points) < 3 or iterations <= 0:
            return points

        pts = [(float(x), float(y)) for x, y in points]
        n = len(pts)

        def _angle_deg(v1, v2):
            import math
            x1, y1 = v1
            x2, y2 = v2
            n1 = math.hypot(x1, y1)
            n2 = math.hypot(x2, y2)
            if n1 <= 1e-9 or n2 <= 1e-9:
                return 0.0
            c = (x1 * x2 + y1 * y2) / (n1 * n2)
            c = max(-1.0, min(1.0, c))
            return math.degrees(math.acos(c))

        def _perp_dist(p, a, b):
            import math
            px, py = p
            ax, ay = a
            bx, by = b
            vx, vy = bx - ax, by - ay
            wx, wy = px - ax, py - ay
            den = vx * vx + vy * vy
            if den <= 1e-9:
                return math.hypot(px - ax, py - ay)
            t = (wx * vx + wy * vy) / den
            t = max(0.0, min(1.0, t))
            cx, cy = ax + t * vx, ay + t * vy
            return math.hypot(px - cx, py - cy)

        corner_idx = [0]
        for i in range(1, n - 1):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            a = _angle_deg((x1 - x0, y1 - y0), (x2 - x1, y2 - y1))
            if a > float(angle_tol_deg):
                corner_idx.append(i)
        corner_idx.append(n - 1)

        merged = []
        for k in range(len(corner_idx) - 1):
            s = corner_idx[k]
            e = corner_idx[k + 1]
            if e <= s:
                continue
            seg = pts[s:e + 1]
            if len(seg) < 2:
                continue

            a = seg[0]
            b = seg[-1]
            max_d = 0.0
            for p in seg[1:-1]:
                d = _perp_dist(p, a, b)
                if d > max_d:
                    max_d = d
                    if max_d > dist_tol:
                        break

            if max_d <= dist_tol:
                seg_out = [a, b]
            else:
                seg_out = [(float(x), float(y)) for x, y in self._bspline_smooth_polyline([(int(round(x)), int(round(y))) for x, y in seg], iterations)]
                if seg_out:
                    seg_out[0] = a
                    seg_out[-1] = b

            if not merged:
                merged.extend(seg_out)
            else:
                if seg_out and (abs(merged[-1][0] - seg_out[0][0]) <= 1e-6 and abs(merged[-1][1] - seg_out[0][1]) <= 1e-6):
                    merged.extend(seg_out[1:])
                else:
                    merged.extend(seg_out)

        out_int = []
        last = None
        for x, y in merged:
            p = (int(round(x)), int(round(y)))
            if last is None or p != last:
                out_int.append(p)
                last = p
        return out_int if len(out_int) >= 2 else points

    def _line_constraint_polyline(self, points, angle_tol_deg=8.0, dist_tol=2.0):
        if not points or len(points) < 3:
            return points

        closed = (points[0] == points[-1])
        if closed:
            pts_in = points[:-1]
        else:
            pts_in = points[:]

        if len(pts_in) < 3:
            return points

        pts = [(float(x), float(y)) for x, y in pts_in]
        n = len(pts)

        def _angle_deg(v1, v2):
            import math
            x1, y1 = v1
            x2, y2 = v2
            n1 = math.hypot(x1, y1)
            n2 = math.hypot(x2, y2)
            if n1 <= 1e-9 or n2 <= 1e-9:
                return 0.0
            c = (x1 * x2 + y1 * y2) / (n1 * n2)
            c = max(-1.0, min(1.0, c))
            return math.degrees(math.acos(c))

        def _perp_dist(p, a, b):
            import math
            px, py = p
            ax, ay = a
            bx, by = b
            vx, vy = bx - ax, by - ay
            wx, wy = px - ax, py - ay
            den = vx * vx + vy * vy
            if den <= 1e-9:
                return math.hypot(px - ax, py - ay)
            t = (wx * vx + wy * vy) / den
            t = max(0.0, min(1.0, t))
            cx, cy = ax + t * vx, ay + t * vy
            return math.hypot(px - cx, py - cy)

        corner_idx = [0]
        for i in range(1, n - 1):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            a = _angle_deg((x1 - x0, y1 - y0), (x2 - x1, y2 - y1))
            if a > float(angle_tol_deg):
                corner_idx.append(i)
        corner_idx.append(n - 1)

        merged = []
        for k in range(len(corner_idx) - 1):
            s = corner_idx[k]
            e = corner_idx[k + 1]
            if e <= s:
                continue
            seg = pts[s:e + 1]
            if len(seg) < 2:
                continue

            a = seg[0]
            b = seg[-1]
            max_d = 0.0
            for p in seg[1:-1]:
                d = _perp_dist(p, a, b)
                if d > max_d:
                    max_d = d
                    if max_d > float(dist_tol):
                        break

            if max_d <= float(dist_tol):
                seg_out = [a, b]
            else:
                seg_out = seg

            if not merged:
                merged.extend(seg_out)
            else:
                if seg_out and (abs(merged[-1][0] - seg_out[0][0]) <= 1e-6 and abs(merged[-1][1] - seg_out[0][1]) <= 1e-6):
                    merged.extend(seg_out[1:])
                else:
                    merged.extend(seg_out)

        out_int = []
        last = None
        for x, y in merged:
            p = (int(round(x)), int(round(y)))
            if last is None or p != last:
                out_int.append(p)
                last = p

        if len(out_int) < 2:
            return points

        if closed and out_int[0] != out_int[-1]:
            out_int.append(out_int[0])
        return out_int

    def _svg_load_polylines(self, svg_path, sample_step=4):
        tree = ET.parse(svg_path)
        root = tree.getroot()

        viewbox = None
        vb = root.attrib.get('viewBox') or root.attrib.get('viewbox')
        if vb:
            nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', vb)
            if len(nums) >= 4:
                try:
                    viewbox = (float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3]))
                except:
                    viewbox = None

        if viewbox is None:
            w = self._svg_parse_length(root.attrib.get('width'))
            h = self._svg_parse_length(root.attrib.get('height'))
            if w is not None and h is not None and w > 0 and h > 0:
                viewbox = (0.0, 0.0, float(w), float(h))

        polylines = []
        self._svg_extract_polylines_from_element(root, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), polylines, sample_step=max(1, int(sample_step)))
        polylines = [p for p in polylines if p and len(p) >= 2]
        return polylines, viewbox

    def _svg_map_polylines_to_image(self, polylines, viewbox, img_w, img_h):
        if not polylines:
            return []

        if viewbox is None:
            xs = []
            ys = []
            for poly in polylines:
                for x, y in poly:
                    xs.append(float(x))
                    ys.append(float(y))
            if xs and ys:
                minx = min(xs)
                miny = min(ys)
                vw = max(1e-6, max(xs) - minx)
                vh = max(1e-6, max(ys) - miny)
                viewbox = (minx, miny, vw, vh)
            else:
                viewbox = (0.0, 0.0, 1.0, 1.0)

        vb_x, vb_y, vb_w, vb_h = viewbox
        sx = float(img_w) / float(vb_w) if vb_w else 1.0
        sy = float(img_h) / float(vb_h) if vb_h else 1.0

        out = []
        for poly in polylines:
            pts = []
            for x, y in poly:
                px = (float(x) - float(vb_x)) * sx
                py = (float(y) - float(vb_y)) * sy
                pts.append((px, py))
            out.append(pts)
        return out

    def _svg_parse_length(self, s):
        if s is None:
            return None
        try:
            s2 = str(s).strip()
        except:
            return None
        if not s2:
            return None
        m = re.search(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', s2)
        if not m:
            return None
        try:
            return float(m.group(0))
        except:
            return None

    def _svg_affine_compose(self, outer, inner):
        a2, b2, c2, d2, e2, f2 = outer
        a1, b1, c1, d1, e1, f1 = inner
        a = a2 * a1 + c2 * b1
        b = b2 * a1 + d2 * b1
        c = a2 * c1 + c2 * d1
        d = b2 * c1 + d2 * d1
        e = a2 * e1 + c2 * f1 + e2
        f = b2 * e1 + d2 * f1 + f2
        return (a, b, c, d, e, f)

    def _svg_apply_affine(self, m, p):
        a, b, c, d, e, f = m
        x, y = p
        return (a * x + c * y + e, b * x + d * y + f)

    def _svg_parse_transform(self, s):
        if not s:
            return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        try:
            text = str(s)
        except:
            return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

        cur = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        for name, args in re.findall(r'([a-zA-Z]+)\s*\(([^)]*)\)', text):
            nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', args)
            vals = []
            for n in nums:
                try:
                    vals.append(float(n))
                except:
                    pass
            name_l = name.strip().lower()
            m = None
            if name_l == 'matrix' and len(vals) >= 6:
                a, b, c, d, e, f = vals[:6]
                m = (a, b, c, d, e, f)
            elif name_l == 'translate' and len(vals) >= 1:
                tx = vals[0]
                ty = vals[1] if len(vals) >= 2 else 0.0
                m = (1.0, 0.0, 0.0, 1.0, tx, ty)
            elif name_l == 'scale' and len(vals) >= 1:
                sx = vals[0]
                sy = vals[1] if len(vals) >= 2 else sx
                m = (sx, 0.0, 0.0, sy, 0.0, 0.0)
            elif name_l == 'rotate' and len(vals) >= 1:
                import math
                ang = vals[0] * math.pi / 180.0
                ca = math.cos(ang)
                sa = math.sin(ang)
                rot = (ca, sa, -sa, ca, 0.0, 0.0)
                if len(vals) >= 3:
                    cx, cy = vals[1], vals[2]
                    t1 = (1.0, 0.0, 0.0, 1.0, -cx, -cy)
                    t2 = (1.0, 0.0, 0.0, 1.0, cx, cy)
                    m = self._svg_affine_compose(t2, self._svg_affine_compose(rot, t1))
                else:
                    m = rot
            if m is not None:
                cur = self._svg_affine_compose(m, cur)
        return cur

    def _svg_extract_polylines_from_element(self, el, parent_m, out, sample_step):
        tag = el.tag
        if '}' in tag:
            tag = tag.split('}', 1)[1]
        tag_l = tag.lower()

        local_m = self._svg_parse_transform(el.attrib.get('transform'))
        total_m = self._svg_affine_compose(parent_m, local_m)

        if tag_l == 'path':
            d = el.attrib.get('d') or ''
            polys = self._svg_parse_path_d(d, sample_step=sample_step)
            for poly in polys:
                out.append([self._svg_apply_affine(total_m, p) for p in poly])
        elif tag_l in ('polyline', 'polygon'):
            pts = self._svg_parse_points_attr(el.attrib.get('points') or '')
            if pts and len(pts) >= 2:
                if tag_l == 'polygon' and pts[0] != pts[-1]:
                    pts = list(pts) + [pts[0]]
                out.append([self._svg_apply_affine(total_m, p) for p in pts])
        elif tag_l == 'line':
            x1 = self._svg_parse_length(el.attrib.get('x1')) or 0.0
            y1 = self._svg_parse_length(el.attrib.get('y1')) or 0.0
            x2 = self._svg_parse_length(el.attrib.get('x2')) or 0.0
            y2 = self._svg_parse_length(el.attrib.get('y2')) or 0.0
            out.append([self._svg_apply_affine(total_m, (x1, y1)), self._svg_apply_affine(total_m, (x2, y2))])
        elif tag_l == 'rect':
            x = self._svg_parse_length(el.attrib.get('x')) or 0.0
            y = self._svg_parse_length(el.attrib.get('y')) or 0.0
            w = self._svg_parse_length(el.attrib.get('width')) or 0.0
            h = self._svg_parse_length(el.attrib.get('height')) or 0.0
            pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            out.append([self._svg_apply_affine(total_m, p) for p in pts])
        elif tag_l in ('circle', 'ellipse'):
            import math
            cx = self._svg_parse_length(el.attrib.get('cx')) or 0.0
            cy = self._svg_parse_length(el.attrib.get('cy')) or 0.0
            if tag_l == 'circle':
                r = self._svg_parse_length(el.attrib.get('r')) or 0.0
                rx, ry = r, r
            else:
                rx = self._svg_parse_length(el.attrib.get('rx')) or 0.0
                ry = self._svg_parse_length(el.attrib.get('ry')) or 0.0
            if rx > 0 and ry > 0:
                n = 64
                pts = []
                for i in range(n + 1):
                    t = 2.0 * math.pi * (float(i) / float(n))
                    pts.append((cx + rx * math.cos(t), cy + ry * math.sin(t)))
                out.append([self._svg_apply_affine(total_m, p) for p in pts])

        for child in list(el):
            self._svg_extract_polylines_from_element(child, total_m, out, sample_step)

    def _svg_parse_points_attr(self, s):
        nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', s or '')
        pts = []
        for i in range(0, len(nums) - 1, 2):
            try:
                x = float(nums[i])
                y = float(nums[i + 1])
                pts.append((x, y))
            except:
                continue
        return pts

    def _svg_parse_path_d(self, d, sample_step=4):
        if not d:
            return []
        tokens = re.findall(r'[A-Za-z]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', d)
        if not tokens:
            return []

        def _is_cmd(t):
            return len(t) == 1 and t.isalpha()

        def _num(i):
            try:
                return float(tokens[i])
            except:
                return 0.0

        def _len_est(ps):
            import math
            s = 0.0
            for i in range(len(ps) - 1):
                x0, y0 = ps[i]
                x1, y1 = ps[i + 1]
                s += math.hypot(x1 - x0, y1 - y0)
            return s

        def _cubic(p0, p1, p2, p3, steps):
            pts = []
            for i in range(1, steps + 1):
                t = float(i) / float(steps)
                mt = 1.0 - t
                x = (mt ** 3) * p0[0] + 3 * (mt ** 2) * t * p1[0] + 3 * mt * (t ** 2) * p2[0] + (t ** 3) * p3[0]
                y = (mt ** 3) * p0[1] + 3 * (mt ** 2) * t * p1[1] + 3 * mt * (t ** 2) * p2[1] + (t ** 3) * p3[1]
                pts.append((x, y))
            return pts

        def _quad(p0, p1, p2, steps):
            pts = []
            for i in range(1, steps + 1):
                t = float(i) / float(steps)
                mt = 1.0 - t
                x = (mt ** 2) * p0[0] + 2 * mt * t * p1[0] + (t ** 2) * p2[0]
                y = (mt ** 2) * p0[1] + 2 * mt * t * p1[1] + (t ** 2) * p2[1]
                pts.append((x, y))
            return pts

        paths = []
        cur_path = []
        cx = cy = 0.0
        sx = sy = 0.0
        last_cmd = None
        last_cubic_ctrl = None
        last_quad_ctrl = None

        i = 0
        while i < len(tokens):
            cmd = tokens[i]
            if _is_cmd(cmd):
                i += 1
                last_cmd = cmd
            else:
                if last_cmd is None:
                    break
                cmd = last_cmd

            rel = cmd.islower()
            c = cmd.lower()

            if c == 'm':
                if i + 1 >= len(tokens):
                    break
                x = _num(i)
                y = _num(i + 1)
                i += 2
                if rel:
                    x += cx
                    y += cy
                if cur_path:
                    paths.append(cur_path)
                cur_path = [(x, y)]
                cx, cy = x, y
                sx, sy = x, y
                last_cubic_ctrl = None
                last_quad_ctrl = None
                while i + 1 < len(tokens) and not _is_cmd(tokens[i]):
                    x = _num(i)
                    y = _num(i + 1)
                    i += 2
                    if rel:
                        x += cx
                        y += cy
                    cur_path.append((x, y))
                    cx, cy = x, y
                continue

            if c == 'z':
                if cur_path:
                    if (cx, cy) != (sx, sy):
                        cur_path.append((sx, sy))
                    paths.append(cur_path)
                    cur_path = []
                cx, cy = sx, sy
                last_cubic_ctrl = None
                last_quad_ctrl = None
                continue

            if c == 'l':
                while i + 1 < len(tokens) and not _is_cmd(tokens[i]):
                    x = _num(i)
                    y = _num(i + 1)
                    i += 2
                    if rel:
                        x += cx
                        y += cy
                    if not cur_path:
                        cur_path = [(cx, cy)]
                    cur_path.append((x, y))
                    cx, cy = x, y
                last_cubic_ctrl = None
                last_quad_ctrl = None
                continue

            if c == 'h':
                while i < len(tokens) and not _is_cmd(tokens[i]):
                    x = _num(i)
                    i += 1
                    if rel:
                        x += cx
                    if not cur_path:
                        cur_path = [(cx, cy)]
                    cur_path.append((x, cy))
                    cx = x
                last_cubic_ctrl = None
                last_quad_ctrl = None
                continue

            if c == 'v':
                while i < len(tokens) and not _is_cmd(tokens[i]):
                    y = _num(i)
                    i += 1
                    if rel:
                        y += cy
                    if not cur_path:
                        cur_path = [(cx, cy)]
                    cur_path.append((cx, y))
                    cy = y
                last_cubic_ctrl = None
                last_quad_ctrl = None
                continue

            if c == 'c':
                while i + 5 < len(tokens) and not _is_cmd(tokens[i]):
                    x1 = _num(i)
                    y1 = _num(i + 1)
                    x2 = _num(i + 2)
                    y2 = _num(i + 3)
                    x = _num(i + 4)
                    y = _num(i + 5)
                    i += 6
                    if rel:
                        x1 += cx
                        y1 += cy
                        x2 += cx
                        y2 += cy
                        x += cx
                        y += cy
                    if not cur_path:
                        cur_path = [(cx, cy)]
                    p0 = (cx, cy)
                    p1 = (x1, y1)
                    p2 = (x2, y2)
                    p3 = (x, y)
                    steps = max(2, int(_len_est([p0, p1, p2, p3]) / float(max(1, sample_step))))
                    cur_path.extend(_cubic(p0, p1, p2, p3, steps))
                    cx, cy = x, y
                    last_cubic_ctrl = (x2, y2)
                    last_quad_ctrl = None
                continue

            if c == 's':
                while i + 3 < len(tokens) and not _is_cmd(tokens[i]):
                    x2 = _num(i)
                    y2 = _num(i + 1)
                    x = _num(i + 2)
                    y = _num(i + 3)
                    i += 4
                    if rel:
                        x2 += cx
                        y2 += cy
                        x += cx
                        y += cy
                    if last_cubic_ctrl is None:
                        x1, y1 = cx, cy
                    else:
                        x1 = 2 * cx - last_cubic_ctrl[0]
                        y1 = 2 * cy - last_cubic_ctrl[1]
                    if not cur_path:
                        cur_path = [(cx, cy)]
                    p0 = (cx, cy)
                    p1 = (x1, y1)
                    p2 = (x2, y2)
                    p3 = (x, y)
                    steps = max(2, int(_len_est([p0, p1, p2, p3]) / float(max(1, sample_step))))
                    cur_path.extend(_cubic(p0, p1, p2, p3, steps))
                    cx, cy = x, y
                    last_cubic_ctrl = (x2, y2)
                    last_quad_ctrl = None
                continue

            if c == 'q':
                while i + 3 < len(tokens) and not _is_cmd(tokens[i]):
                    x1 = _num(i)
                    y1 = _num(i + 1)
                    x = _num(i + 2)
                    y = _num(i + 3)
                    i += 4
                    if rel:
                        x1 += cx
                        y1 += cy
                        x += cx
                        y += cy
                    if not cur_path:
                        cur_path = [(cx, cy)]
                    p0 = (cx, cy)
                    p1 = (x1, y1)
                    p2 = (x, y)
                    steps = max(2, int(_len_est([p0, p1, p2]) / float(max(1, sample_step))))
                    cur_path.extend(_quad(p0, p1, p2, steps))
                    cx, cy = x, y
                    last_quad_ctrl = (x1, y1)
                    last_cubic_ctrl = None
                continue

            if c == 't':
                while i + 1 < len(tokens) and not _is_cmd(tokens[i]):
                    x = _num(i)
                    y = _num(i + 1)
                    i += 2
                    if rel:
                        x += cx
                        y += cy
                    if last_quad_ctrl is None:
                        x1, y1 = cx, cy
                    else:
                        x1 = 2 * cx - last_quad_ctrl[0]
                        y1 = 2 * cy - last_quad_ctrl[1]
                    if not cur_path:
                        cur_path = [(cx, cy)]
                    p0 = (cx, cy)
                    p1 = (x1, y1)
                    p2 = (x, y)
                    steps = max(2, int(_len_est([p0, p1, p2]) / float(max(1, sample_step))))
                    cur_path.extend(_quad(p0, p1, p2, steps))
                    cx, cy = x, y
                    last_quad_ctrl = (x1, y1)
                    last_cubic_ctrl = None
                continue

            if c == 'a':
                while i + 6 < len(tokens) and not _is_cmd(tokens[i]):
                    x = _num(i + 5)
                    y = _num(i + 6)
                    i += 7
                    if rel:
                        x += cx
                        y += cy
                    if not cur_path:
                        cur_path = [(cx, cy)]
                    cur_path.append((x, y))
                    cx, cy = x, y
                    last_cubic_ctrl = None
                    last_quad_ctrl = None
                continue

            i += 1

        if cur_path:
            paths.append(cur_path)
        return [p for p in paths if p and len(p) >= 2]

    def _optimize_contours_order(self, contours):
        if not contours:
            return contours

        remaining = []
        for c in contours:
            if not c or len(c) < 2:
                continue
            remaining.append(c)
        if len(remaining) <= 1:
            return remaining

        def _dist2(a, b):
            dx = float(a[0]) - float(b[0])
            dy = float(a[1]) - float(b[1])
            return dx * dx + dy * dy

        cur = (0.0, 0.0)
        ordered = []
        used = [False] * len(remaining)
        left = len(remaining)

        while left > 0:
            best_i = None
            best_rev = False
            best_d2 = None
            for i, c in enumerate(remaining):
                if used[i]:
                    continue
                d2_start = _dist2(cur, c[0])
                d2_end = _dist2(cur, c[-1])
                if best_d2 is None or d2_start < best_d2:
                    best_d2 = d2_start
                    best_i = i
                    best_rev = False
                if d2_end < best_d2:
                    best_d2 = d2_end
                    best_i = i
                    best_rev = True

            c = remaining[best_i]
            if best_rev:
                c = list(reversed(c))
            ordered.append(c)
            cur = c[-1]
            used[best_i] = True
            left -= 1

        return ordered
    
    def on_paper_change(self, event=None):
        size_name = self.paper_var.get()
        if size_name in PaperSizes.SIZES:
            w, h = PaperSizes.SIZES[size_name]
            if self.orientation_var.get() == 'landscape':
                w, h = h, w
            self.paper_width, self.paper_height = w, h
            self.draw_paper()
    
    def on_orientation_change(self):
        self.on_paper_change()
    
    def apply_custom_size(self):
        try:
            w = float(self.custom_width_var.get())
            h = float(self.custom_height_var.get())
            if w > 0 and h > 0:
                self.paper_var.set('自定义')
                if self.orientation_var.get() == 'landscape':
                    w, h = h, w
                self.paper_width, self.paper_height = w, h
                self.draw_paper()
            else:
                messagebox.showwarning("警告", "尺寸必须大于0")
        except ValueError:
            messagebox.showerror("错误", "请输入有效数字")
    
    def draw_paper(self):
        self.preview_canvas.delete('all')
        
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width < 10 or canvas_height < 10:
            self.root.after(100, self.draw_paper)
            return
        
        paper_pixel_width = self.paper_width * self.pixels_per_mm
        paper_pixel_height = self.paper_height * self.pixels_per_mm
        
        scale_fit = min((canvas_width - 40) / paper_pixel_width, 
                        (canvas_height - 40) / paper_pixel_height) * 0.95
        
        paper_display_width = paper_pixel_width * scale_fit
        paper_display_height = paper_pixel_height * scale_fit
        
        paper_x = (canvas_width - paper_display_width) / 2
        paper_y = (canvas_height - paper_display_height) / 2
        
        self.preview_canvas.create_rectangle(
            paper_x, paper_y, 
            paper_x + paper_display_width, paper_y + paper_display_height,
            fill='white', outline='black', width=2, tags='paper'
        )
        
        grid_step = 10 * self.pixels_per_mm * scale_fit
        
        for i in range(int(self.paper_width / 10) + 1):
            x = paper_x + i * grid_step
            self.preview_canvas.create_line(x, paper_y, x, paper_y + paper_display_height, fill='#e0e0e0', tags='grid')
        
        for i in range(int(self.paper_height / 10) + 1):
            y = paper_y + i * grid_step
            self.preview_canvas.create_line(paper_x, y, paper_x + paper_display_width, y, fill='#e0e0e0', tags='grid')
        
        self.preview_canvas.create_text(paper_x + paper_display_width / 2, paper_y - 10,
            text=f"{self.paper_width}mm", fill='gray', tags='label')
        self.preview_canvas.create_text(paper_x - 10, paper_y + paper_display_height / 2,
            text=f"{self.paper_height}mm", fill='gray', angle=90, tags='label')
        
        self.preview_paper_x = paper_x
        self.preview_paper_y = paper_y
        self.preview_scale = scale_fit * self.pixels_per_mm
        
        self.draw_ground()  # 绘制Ground参照图
        self.draw_contours()
    
    def draw_contours(self):
        self.preview_canvas.delete('contour')
        # 只有当选中contour或未选中任何对象时才删除transform
        if self.selected_object != 'ground':
            self.preview_canvas.delete('transform')
        
        if not self.contours:
            self.contour_bounds = None
            return
        
        try:
            offset_x = float(self.offset_x_var.get())
            offset_y = float(self.offset_y_var.get())
            scale = float(self.scale_var.get())
        except ValueError:
            return
        
        # 计算轮廓边界框(mm坐标)
        all_mm_x = []
        all_mm_y = []
        for contour in self.contours:
            for point in contour:
                mm_x = point[0] * 0.1 * scale + offset_x
                mm_y = point[1] * 0.1 * scale + offset_y
                all_mm_x.append(mm_x)
                all_mm_y.append(mm_y)
        
        if all_mm_x and all_mm_y:
            # 未旋转的边界框
            min_x, max_x = min(all_mm_x), max(all_mm_x)
            min_y, max_y = min(all_mm_y), max(all_mm_y)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
        else:
            center_x, center_y = offset_x, offset_y
            min_x, max_x, min_y, max_y = offset_x, offset_x, offset_y, offset_y
        
        # 保存轮廓中心和边界
        self.contour_bounds = (min_x, min_y, max_x, max_y)
        self.transform_center_x = center_x
        self.transform_center_y = center_y
        
        # 绘制轮廓
        for contour in self.contours:
            if len(contour) < 2:
                continue
            
            points = []
            for point in contour:
                mm_x = point[0] * 0.1 * scale + offset_x
                mm_y = point[1] * 0.1 * scale + offset_y
                
                # 应用旋转
                if self.image_rotation != 0:
                    mm_x, mm_y = self._rotate_point(mm_x, mm_y, center_x, center_y, self.image_rotation)
                
                canvas_x = self.preview_paper_x + mm_x * self.preview_scale
                canvas_y = self.preview_paper_y + mm_y * self.preview_scale
                
                points.extend([canvas_x, canvas_y])
            
            if len(points) >= 4:
                self.preview_canvas.create_line(points, fill='blue', width=1, tags='contour')
        
        # 绘制变换控制框（如果选中了contour）
        if self.selected_object == 'contour':
            self._draw_transform_box('contour', min_x, min_y, max_x, max_y, center_x, center_y)
    
    def on_preview_click(self, event):
        """PS风格点击处理：检测点击对象/控制点"""
        if not hasattr(self, 'preview_scale') or self.preview_scale == 0:
            return
        
        # 检测是否点击了角度标签（用于直接输入角度）
        if self.selected_object and hasattr(self, '_angle_label_bounds'):
            x1, y1, x2, y2 = self._angle_label_bounds
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self._show_angle_input_dialog()
                return
        
        # 检测是否点击了控制点（如果已选中对象）
        if self.selected_object:
            handle = self._hit_test_transform_handle(event.x, event.y)
            if handle:
                self.transform_handle = handle
                self.transform_start_x = event.x
                self.transform_start_y = event.y
                if handle == 'rotate':
                    self.transform_mode = 'rotate'
                    # 记录初始旋转角度
                    if self.selected_object == 'contour':
                        self.transform_start_angle = self.image_rotation
                    else:
                        self.transform_start_angle = self.ground_rotation
                else:
                    self.transform_mode = 'scale'
                    if self.selected_object == 'contour':
                        self.transform_start_scale = float(self.scale_var.get())
                    else:
                        self.transform_start_scale = self.ground_scale
                return
        
        # 检测是否点击了对象内部
        clicked_obj = self._hit_test_object(event.x, event.y)
        
        if clicked_obj:
            self.selected_object = clicked_obj
            self.transform_mode = 'move'
            self.transform_start_x = event.x
            self.transform_start_y = event.y
            
            if clicked_obj == 'contour':
                self.transform_start_offset_x = float(self.offset_x_var.get())
                self.transform_start_offset_y = float(self.offset_y_var.get())
            else:  # ground
                self.transform_start_offset_x = self.ground_offset_x
                self.transform_start_offset_y = self.ground_offset_y
            
            self.draw_paper()
        else:
            # 点击空白区域，取消选中
            self.selected_object = None
            self.transform_mode = None
            self.draw_paper()
    
    def on_preview_drag(self, event):
        """PS风格拖动处理：移动/缩放/旋转"""
        if not self.transform_mode or not self.selected_object:
            return
        
        if not hasattr(self, 'preview_scale') or self.preview_scale == 0:
            return
        
        if self.transform_mode == 'move':
            # 相对位置移动
            dx_px = event.x - self.transform_start_x
            dy_px = event.y - self.transform_start_y
            dx_mm = dx_px / self.preview_scale
            dy_mm = dy_px / self.preview_scale
            
            if self.selected_object == 'contour':
                new_x = self.transform_start_offset_x + dx_mm
                new_y = self.transform_start_offset_y + dy_mm
                self.offset_x_var.set(f"{new_x:.1f}")
                self.offset_y_var.set(f"{new_y:.1f}")
                self.draw_contours()
            else:  # ground
                if not self.ground_locked:
                    self.ground_offset_x = self.transform_start_offset_x + dx_mm
                    self.ground_offset_y = self.transform_start_offset_y + dy_mm
                    self.draw_paper()
        
        elif self.transform_mode == 'scale':
            self._handle_scale_drag(event)
        
        elif self.transform_mode == 'rotate':
            self._handle_rotate_drag(event)
    
    def on_preview_release(self, event):
        """PS风格释放处理"""
        if self.transform_mode == 'move':
            # 移动完成，保持选中状态
            pass
        self.transform_mode = None
        self.transform_handle = None
        self.draw_paper()
    
    def delete_selected_object(self, event=None):
        """删除选中的对象"""
        if not self.selected_object:
            return
        
        if self.selected_object == 'contour':
            # 删除图像/轮廓/文字
            self.contours = []
            self.original_image = None
            self.processed_image = None
            self.contour_bounds = None
            self.image_rotation = 0
            self.image_rotation_var.set(0)
            self.image_rotation_label.config(text="0°")
            self.selected_object = None
            self.status_label.config(text="已删除图像/文字", foreground='orange')
            
        elif self.selected_object == 'ground':
            # 删除Ground参照图
            if self.ground_locked:
                return  # 锁定时不允许删除
            self.ground_image = None
            self.ground_photo = None
            self.ground_points = []
            self.ground_scale = 0.1
            self.ground_offset_x = 0
            self.ground_offset_y = 0
            self.ground_rotation = 0
            self.ground_rotation_var.set(0)
            self.ground_rotation_label.config(text="0°")
            self.ground_locked = False
            self.ground_lock_btn.config(text="锁定")
            self.selected_object = None
            self.ground_status_label.config(text="已删除参照图", foreground='orange')
        
        self.draw_paper()
    
    # === PS风格操作辅助方法 ===
    def _draw_transform_box(self, obj_type, min_x, min_y, max_x, max_y, center_x, center_y):
        """绘制变换控制框（8个控制点+旋转手柄）"""
        # 应用旋转获取四个角点
        rotation = self.image_rotation if obj_type == 'contour' else self.ground_rotation
        
        # 四个角点(mm)
        corners_mm = [
            (min_x, min_y),  # nw
            (max_x, min_y),  # ne
            (max_x, max_y),  # se
            (min_x, max_y),  # sw
        ]
        
        # 旋转角点
        rotated_corners = []
        for x, y in corners_mm:
            if rotation != 0:
                rx, ry = self._rotate_point(x, y, center_x, center_y, rotation)
            else:
                rx, ry = x, y
            # 转换为画布坐标
            cx = self.preview_paper_x + rx * self.preview_scale
            cy = self.preview_paper_y + ry * self.preview_scale
            rotated_corners.append((cx, cy))
        
        # 绘制边框
        for i in range(4):
            x1, y1 = rotated_corners[i]
            x2, y2 = rotated_corners[(i + 1) % 4]
            self.preview_canvas.create_line(x1, y1, x2, y2, fill='#0078D7', width=2, 
                                           dash=(4, 2), tags='transform')
        
        # 中心点（画布坐标）
        center_cx = self.preview_paper_x + center_x * self.preview_scale
        center_cy = self.preview_paper_y + center_y * self.preview_scale
        
        # 绘制8个控制点
        handle_size = 6
        handle_positions = {}
        
        # 4个角点
        handle_positions['nw'] = rotated_corners[0]
        handle_positions['ne'] = rotated_corners[1]
        handle_positions['se'] = rotated_corners[2]
        handle_positions['sw'] = rotated_corners[3]
        
        # 4个边中点
        handle_positions['n'] = ((rotated_corners[0][0] + rotated_corners[1][0]) / 2,
                                 (rotated_corners[0][1] + rotated_corners[1][1]) / 2)
        handle_positions['e'] = ((rotated_corners[1][0] + rotated_corners[2][0]) / 2,
                                 (rotated_corners[1][1] + rotated_corners[2][1]) / 2)
        handle_positions['s'] = ((rotated_corners[2][0] + rotated_corners[3][0]) / 2,
                                 (rotated_corners[2][1] + rotated_corners[3][1]) / 2)
        handle_positions['w'] = ((rotated_corners[3][0] + rotated_corners[0][0]) / 2,
                                 (rotated_corners[3][1] + rotated_corners[0][1]) / 2)
        
        for name, (hx, hy) in handle_positions.items():
            self.preview_canvas.create_rectangle(
                hx - handle_size, hy - handle_size,
                hx + handle_size, hy + handle_size,
                fill='white', outline='#0078D7', width=2, tags='transform'
            )
        
        # 旋转手柄（在顶部中心上方30像素）
        rotate_x, rotate_y = handle_positions['n']
        # 根据旋转角度调整手柄位置
        import math
        angle_rad = math.radians(-rotation - 90)  # 向上
        rotate_dist = 30
        rotate_hx = rotate_x + rotate_dist * math.cos(angle_rad)
        rotate_hy = rotate_y + rotate_dist * math.sin(angle_rad)
        
        # 连接线
        self.preview_canvas.create_line(rotate_x, rotate_y, rotate_hx, rotate_hy,
                                       fill='#0078D7', width=1, tags='transform')
        # 旋转手柄圆圈
        self.preview_canvas.create_oval(
            rotate_hx - 8, rotate_hy - 8, rotate_hx + 8, rotate_hy + 8,
            fill='#0078D7', outline='white', width=2, tags='transform'
        )
        handle_positions['rotate'] = (rotate_hx, rotate_hy)
        
        # 在旋转手柄旁边显示角度值
        angle_text_x = rotate_hx + 20
        angle_text_y = rotate_hy
        angle_text = f"{int(rotation)}°"
        self.preview_canvas.create_rectangle(
            angle_text_x - 2, angle_text_y - 10,
            angle_text_x + 35, angle_text_y + 10,
            fill='white', outline='#0078D7', tags='transform'
        )
        self.preview_canvas.create_text(
            angle_text_x + 16, angle_text_y, 
            text=angle_text, fill='#0078D7', font=('Arial', 10, 'bold'),
            tags=('transform', 'angle_label')
        )
        # 保存角度标签位置用于点击检测
        self._angle_label_bounds = (angle_text_x - 2, angle_text_y - 10, angle_text_x + 35, angle_text_y + 10)
        
        # 绘制中心点
        self.preview_canvas.create_oval(
            center_cx - 4, center_cy - 4, center_cx + 4, center_cy + 4,
            fill='#0078D7', outline='white', width=1, tags='transform'
        )
        
        # 保存控制点位置用于点击检测
        self._transform_handles = handle_positions
        self._transform_center = (center_cx, center_cy)
    
    def _show_angle_input_dialog(self):
        """显示角度输入对话框"""
        from tkinter import simpledialog
        
        current_angle = self.image_rotation if self.selected_object == 'contour' else self.ground_rotation
        
        result = simpledialog.askfloat(
            "输入旋转角度",
            "请输入旋转角度 (-180 到 180)：",
            initialvalue=int(current_angle),
            minvalue=-180,
            maxvalue=180,
            parent=self.root
        )
        
        if result is not None:
            # 限制范围
            new_angle = max(-180, min(180, result))
            
            if self.selected_object == 'contour':
                self.image_rotation = new_angle
                self.image_rotation_var.set(int(new_angle))
                self.image_rotation_label.config(text=f"{int(new_angle)}°")
            else:  # ground
                if not self.ground_locked:
                    self.ground_rotation = new_angle
                    self.ground_rotation_var.set(int(new_angle))
                    self.ground_rotation_label.config(text=f"{int(new_angle)}°")
                    self.ground_points = []  # 清除定标点
            
            self.draw_paper()
    
    def _hit_test_transform_handle(self, x, y):
        """检测是否点击了控制点"""
        if not hasattr(self, '_transform_handles'):
            return None
        
        hit_radius = 10  # 点击判定半径
        
        for name, (hx, hy) in self._transform_handles.items():
            if (x - hx) ** 2 + (y - hy) ** 2 <= hit_radius ** 2:
                return name
        
        return None
    
    def _hit_test_object(self, x, y):
        """检测点击了哪个对象"""
        # 转换为mm坐标
        mm_x = (x - self.preview_paper_x) / self.preview_scale
        mm_y = (y - self.preview_paper_y) / self.preview_scale
        
        # 检测contour边界框
        if self.contour_bounds:
            min_x, min_y, max_x, max_y = self.contour_bounds
            # 考虑旋转：简化处理，使用扩大的边界框
            padding = 10  # mm
            if min_x - padding <= mm_x <= max_x + padding and min_y - padding <= mm_y <= max_y + padding:
                return 'contour'
        
        # 检测ground边界框
        if self.ground_image is not None and self.ground_visible:
            h, w = self.ground_image.shape[:2]
            g_min_x = self.ground_offset_x
            g_min_y = self.ground_offset_y
            g_max_x = self.ground_offset_x + w * self.ground_scale
            g_max_y = self.ground_offset_y + h * self.ground_scale
            padding = 10
            if g_min_x - padding <= mm_x <= g_max_x + padding and g_min_y - padding <= mm_y <= g_max_y + padding:
                return 'ground'
        
        return None
    
    def _handle_scale_drag(self, event):
        """处理缩放拖动（绕中心等比缩放）"""
        if not hasattr(self, '_transform_center'):
            return
        
        center_cx, center_cy = self._transform_center
        
        # 计算距离比例
        start_dist = ((self.transform_start_x - center_cx) ** 2 + 
                      (self.transform_start_y - center_cy) ** 2) ** 0.5
        current_dist = ((event.x - center_cx) ** 2 + (event.y - center_cy) ** 2) ** 0.5
        
        if start_dist < 1:
            return
        
        scale_factor = current_dist / start_dist
        
        if self.selected_object == 'contour':
            new_scale = self.transform_start_scale * scale_factor
            new_scale = max(0.1, min(10.0, new_scale))
            self.scale_var.set(f"{new_scale:.2f}")
            self.draw_contours()
        else:  # ground
            if not self.ground_locked:
                self.ground_scale = self.transform_start_scale * scale_factor
                self.ground_scale = max(0.01, min(10.0, self.ground_scale))
                self.draw_paper()
    
    def _handle_rotate_drag(self, event):
        """处理旋转拖动（绕中心旋转）"""
        if not hasattr(self, '_transform_center'):
            return
        
        import math
        center_cx, center_cy = self._transform_center
        
        # 计算角度变化
        start_angle = math.atan2(self.transform_start_y - center_cy, 
                                  self.transform_start_x - center_cx)
        current_angle = math.atan2(event.y - center_cy, event.x - center_cx)
        
        delta_angle = math.degrees(current_angle - start_angle)
        new_rotation = self.transform_start_angle + delta_angle
        
        # 限制在-180到180度
        while new_rotation > 180:
            new_rotation -= 360
        while new_rotation < -180:
            new_rotation += 360
        
        if self.selected_object == 'contour':
            self.image_rotation = new_rotation
            self.image_rotation_var.set(int(new_rotation))
            self.image_rotation_label.config(text=f"{int(new_rotation)}°")
            self.draw_contours()
        else:  # ground
            if not self.ground_locked:
                self.ground_rotation = new_rotation
                self.ground_rotation_var.set(int(new_rotation))
                self.ground_rotation_label.config(text=f"{int(new_rotation)}°")
                # ground旋转时清除定标点
                self.ground_points = []
                self.draw_paper()
    
    def on_mouse_wheel(self, event):
        if self.contours:
            try:
                current_scale = float(self.scale_var.get())
                if event.delta > 0:
                    new_scale = current_scale * 1.1
                else:
                    new_scale = current_scale / 1.1
                
                new_scale = max(0.1, min(10.0, new_scale))
                self.scale_var.set(f"{new_scale:.2f}")
                self.draw_contours()
            except ValueError:
                pass
    
    def apply_position(self):
        self.draw_contours()

    def center_contours_on_paper(self):
        if not self.contours:
            return
        
        try:
            scale_factor = float(self.scale_var.get())
        except ValueError:
            return
        
        scale = scale_factor * 0.1
        if scale == 0:
            return
        
        min_x = None
        max_x = None
        min_y = None
        max_y = None
        
        for contour in self.contours:
            for px, py in contour:
                if min_x is None:
                    min_x = max_x = px
                    min_y = max_y = py
                else:
                    min_x = min(min_x, px)
                    max_x = max(max_x, px)
                    min_y = min(min_y, py)
                    max_y = max(max_y, py)
        
        if min_x is None:
            return
        
        center_x_mm = ((min_x + max_x) / 2) * scale
        center_y_mm = ((min_y + max_y) / 2) * scale
        
        offset_x = (self.paper_width / 2) - center_x_mm
        offset_y = (self.paper_height / 2) - center_y_mm
        
        self.offset_x_var.set(f"{offset_x:.2f}")
        self.offset_y_var.set(f"{offset_y:.2f}")
    
    def export_gcode(self):
        if not self.contours:
            messagebox.showwarning("警告", "没有可导出的轮廓，请先处理图像")
            return
        
        try:
            offset_x = float(self.offset_x_var.get())
            offset_y = float(self.offset_y_var.get())
            scale = float(self.scale_var.get()) * 0.1
            
            feed_rate = int(self.feed_rate_var.get())
            z_up = float(self.z_up_var.get())
            z_down = float(self.z_down_var.get())
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")
            return

        if scale == 0:
            messagebox.showerror("错误", "缩放比例不能为0")
            return
        
        gcode_gen = GCodeGenerator(feed_rate, z_up, z_down)
        
        # 应用旋转变换
        contours_to_export = self.contours
        if self.image_rotation != 0:
            # 计算中心点
            all_x = []
            all_y = []
            for contour in self.contours:
                for point in contour:
                    all_x.append(point[0] * scale)
                    all_y.append(point[1] * scale)
            if all_x and all_y:
                center_x = (min(all_x) + max(all_x)) / 2 + offset_x
                center_y = (min(all_y) + max(all_y)) / 2 + offset_y
            else:
                center_x = offset_x
                center_y = offset_y
            
            # 旋转轮廓
            rotated_contours = []
            for contour in self.contours:
                rotated_contour = []
                for point in contour:
                    mm_x = point[0] * scale + offset_x
                    mm_y = point[1] * scale + offset_y
                    rx, ry = self._rotate_point(mm_x, mm_y, center_x, center_y, self.image_rotation)
                    # 转换回像素坐标
                    rotated_contour.append(((rx - offset_x) / scale, (ry - offset_y) / scale))
                rotated_contours.append(rotated_contour)
            contours_to_export = rotated_contours
        
        gcode = gcode_gen.generate_from_contours(
            contours_to_export, offset_x, offset_y, scale, self.paper_height
        )
        
        filetypes = [('GCode文件', '*.gcode *.nc *.ngc'), ('所有文件', '*.*')]
        path = filedialog.asksaveasfilename(
            title="保存GCode文件",
            defaultextension=".gcode",
            filetypes=filetypes
        )
        
        if path:
            try:
                with open(path, 'w') as f:
                    f.write('\n'.join(gcode))
                self.status_label.config(text=f"GCode已保存: {os.path.basename(path)}", foreground='green')
                self.gcode_lines = gcode  # 保存到内存
                messagebox.showinfo("成功", f"GCode文件已保存\n路径: {path}\n总行数: {len(gcode)}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    # === 文字输入功能 ===
    def generate_text_contours(self):
        """将文字转换为骨架线"""
        text = self.text_input_var.get().strip()
        if not text:
            messagebox.showwarning("警告", "请输入文字")
            return
        
        try:
            font_name = self.font_var.get()
            font_size = self.font_size_var.get()
            stroke_width = self.stroke_width_var.get()
        except:
            font_size = 80
            stroke_width = 3
        
        try:
            from PIL import ImageFont, ImageDraw
            from skimage.morphology import skeletonize
            
            # 字体映射
            font_map = {
                'simhei': 'C:/Windows/Fonts/simhei.ttf',
                'simsun': 'C:/Windows/Fonts/simsun.ttc',
                'msyh': 'C:/Windows/Fonts/msyh.ttc',
                'arial': 'C:/Windows/Fonts/arial.ttf',
                'times': 'C:/Windows/Fonts/times.ttf',
                'consola': 'C:/Windows/Fonts/consola.ttf',
            }
            
            font = None
            font_path = font_map.get(font_name.lower())
            if font_path:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except:
                    pass
            
            if font is None:
                # 尝试其他字体
                for path in font_map.values():
                    try:
                        font = ImageFont.truetype(path, font_size)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
            
            # 获取文字边界框
            dummy_img = Image.new('L', (1, 1))
            draw = ImageDraw.Draw(dummy_img)
            bbox = draw.textbbox((0, 0), text, font=font)
            padding = stroke_width * 4 + 20
            text_width = bbox[2] - bbox[0] + padding * 2
            text_height = bbox[3] - bbox[1] + padding * 2
            
            # 创建图像并绘制粗体文字
            img = Image.new('L', (text_width, text_height), 255)
            draw = ImageDraw.Draw(img)
            
            x = padding - bbox[0]
            y = padding - bbox[1]
            
            # 绘制粗体文字（通过多次偏移绘制）
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx*dx + dy*dy <= stroke_width*stroke_width:
                        draw.text((x + dx, y + dy), text, font=font, fill=0)
            
            # 转换为OpenCV图像
            img_cv = np.array(img)
            
            # 二值化
            _, binary = cv2.threshold(img_cv, 128, 255, cv2.THRESH_BINARY_INV)
            
            # 骨架化 - 提取中心线
            skeleton = skeletonize(binary > 0)
            skeleton_img = (skeleton * 255).astype(np.uint8)
            
            # 从骨架图提取轮廓/路径
            contours_cv, _ = cv2.findContours(skeleton_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            # 转换为路径格式
            self.contours = []
            for contour in contours_cv:
                if len(contour) < 2:
                    continue
                
                # 简化路径
                epsilon = 0.5
                approx = cv2.approxPolyDP(contour, epsilon, False)
                
                points = []
                for point in approx:
                    points.append((point[0][0], point[0][1]))
                
                if len(points) >= 2:
                    self.contours.append(points)
            
            # 保存图像尺寸
            self.image_mm_width = text_width * 0.1
            self.image_mm_height = text_height * 0.1
            
            self.status_label.config(text=f"已生成文字骨架: {len(self.contours)}条路径", foreground='green')
            self.gcode_lines = []
            
            self.center_contours_on_paper()
            self.draw_paper()
            self.notebook.select(1)
            
        except ImportError:
            messagebox.showerror("错误", "请安装 scikit-image: pip install scikit-image")
        except Exception as e:
            messagebox.showerror("错误", f"生成文字失败: {str(e)}")
    
    # === Ground参照功能 ===
    def load_ground_image(self):
        """加载参照图片"""
        filetypes = [('图片文件', '*.png *.jpg *.jpeg *.bmp *.gif *.tiff'), ('所有文件', '*.*')]
        path = filedialog.askopenfilename(title="选择参照图片", filetypes=filetypes)
        
        if path:
            # 使用ImageProcessor.load_image支持中文路径
            self.ground_image = ImageProcessor.load_image(path)
            if self.ground_image is not None:
                # 重置ground状态
                self.ground_points = []
                self.ground_locked = False
                self.ground_offset_x = 0
                self.ground_offset_y = 0
                self.ground_rotation = 0
                self.ground_rotation_var.set(0)
                self.ground_rotation_label.config(text="0°")
                self.ground_mode = None
                self.ground_lock_btn.config(text="锁定")
                
                # 自动计算初始缩放比例，让图片高度占纸张的50%
                h, w = self.ground_image.shape[:2]
                target_height_mm = self.paper_height * 0.5  # 目标高度50%纸张高度
                self.ground_scale = target_height_mm / h  # mm/像素
                
                self.ground_status_label.config(text=f"已加载: {os.path.basename(path)}", foreground='green')
                self.draw_paper()
                self.notebook.select(1)
            else:
                messagebox.showerror("错误", "无法加载图片")
    
    def set_ground_mode(self, mode):
        """设置ground操作模式"""
        if self.ground_image is None:
            messagebox.showwarning("警告", "请先加载参照图片")
            return
        
        if self.ground_locked:
            messagebox.showwarning("警告", "Ground已锁定，请先解锁")
            return
        
        # 设置定标点时要求旋转为0
        if mode in ['point1', 'point2'] and self.ground_rotation != 0:
            messagebox.showwarning("警告", "请先将旋转角度设为0°后再设置定标点")
            return
        
        self.ground_mode = mode
        
        if mode == 'point1':
            self.ground_status_label.config(text="请在预览图上点击上顶点", foreground='blue')
        elif mode == 'point2':
            self.ground_status_label.config(text="请在预览图上点击下顶点", foreground='blue')
        elif mode == 'move':
            self.ground_status_label.config(text="拖动参照图移动位置", foreground='blue')
    
    def calculate_ground_scale(self):
        """计算ground缩放比例"""
        if len(self.ground_points) < 2:
            messagebox.showwarning("警告", "请先设置上下两个顶点")
            return
        
        if self.ground_rotation != 0:
            messagebox.showwarning("警告", "请先将旋转角度设为0°后再计算缩放")
            return
        
        try:
            real_size = float(self.ground_size_var.get())
            if real_size <= 0:
                messagebox.showerror("错误", "尺寸必须大于0")
                return
        except ValueError:
            messagebox.showerror("错误", "请输入有效数字")
            return
        
        # 计算两点之间的像素距离（使用欧几里得距离）
        p1, p2 = self.ground_points
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        pixel_distance = (dx*dx + dy*dy) ** 0.5
        
        if pixel_distance < 1:
            messagebox.showerror("错误", "两点距离太近")
            return
        
        # 计算每像素对应的毫米数
        self.ground_scale = real_size / pixel_distance
        
        self.ground_status_label.config(
            text=f"缩放: {self.ground_scale:.4f} mm/px (已计算)", 
            foreground='green'
        )
        self.draw_paper()
    
    def toggle_ground_lock(self):
        """切换ground锁定状态"""
        if self.ground_image is None:
            return
        
        self.ground_locked = not self.ground_locked
        self.ground_mode = None
        
        if self.ground_locked:
            self.ground_lock_btn.config(text="解锁")
            self.ground_status_label.config(text="Ground已锁定", foreground='green')
        else:
            self.ground_lock_btn.config(text="锁定")
            self.ground_status_label.config(text="Ground已解锁", foreground='orange')
    
    def toggle_ground_visible(self):
        """切换ground显示状态"""
        self.ground_visible = self.ground_visible_var.get()
        self.draw_paper()
    
    def clear_ground(self):
        """清除ground"""
        self.ground_image = None
        self.ground_photo = None
        self.ground_points = []
        self.ground_locked = False
        self.ground_scale = 0.1  # 默认缩放比例
        self.ground_offset_x = 0
        self.ground_offset_y = 0
        self.ground_rotation = 0
        self.ground_rotation_var.set(0)
        self.ground_rotation_label.config(text="0°")
        self.ground_mode = None
        self.ground_lock_btn.config(text="锁定")
        self.ground_status_label.config(text="未加载参照图", foreground='gray')
        self.draw_paper()
    
    def on_ground_preview_click(self, event):
        """处理ground在预览画布上的点击"""
        if self.ground_image is None or not hasattr(self, 'preview_scale'):
            return False
        
        if self.ground_mode in ['point1', 'point2']:
            # 设置定标点时必须旋转为0
            if self.ground_rotation != 0:
                return False
            
            # 画布坐标 -> 纸张mm坐标 -> 图片像素坐标
            mm_x = (event.x - self.preview_paper_x) / self.preview_scale - self.ground_offset_x
            mm_y = (event.y - self.preview_paper_y) / self.preview_scale - self.ground_offset_y
            # mm坐标转为图片像素坐标
            img_x = mm_x / self.ground_scale
            img_y = mm_y / self.ground_scale
            
            if self.ground_mode == 'point1':
                if len(self.ground_points) == 0:
                    self.ground_points.append((img_x, img_y))
                else:
                    self.ground_points[0] = (img_x, img_y)
                self.ground_status_label.config(text="上顶点已设置", foreground='green')
                self.ground_mode = None
                self.draw_paper()
                return True
            
            elif self.ground_mode == 'point2':
                if len(self.ground_points) == 0:
                    messagebox.showwarning("警告", "请先设置上顶点")
                    return True
                if len(self.ground_points) == 1:
                    self.ground_points.append((img_x, img_y))
                else:
                    self.ground_points[1] = (img_x, img_y)
                self.ground_status_label.config(text="下顶点已设置", foreground='green')
                self.ground_mode = None
                self.draw_paper()
                return True
        
        elif self.ground_mode == 'move':
            self.ground_drag_start_x = event.x
            self.ground_drag_start_y = event.y
            return True
        
        return False
    
    def on_ground_preview_drag(self, event):
        """处理ground移动"""
        if self.ground_mode == 'move' and self.ground_image is not None:
            dx = event.x - self.ground_drag_start_x
            dy = event.y - self.ground_drag_start_y
            
            # 转换为mm
            dx_mm = dx / self.preview_scale
            dy_mm = dy / self.preview_scale
            
            self.ground_offset_x += dx_mm
            self.ground_offset_y += dy_mm
            
            self.ground_drag_start_x = event.x
            self.ground_drag_start_y = event.y
            
            self.draw_paper()
            return True
        return False
    
    def on_ground_preview_release(self, event):
        """处理ground移动结束"""
        if self.ground_mode == 'move':
            self.ground_status_label.config(
                text=f"位置: ({self.ground_offset_x:.1f}, {self.ground_offset_y:.1f}) mm", 
                foreground='green'
            )
            return True
        return False
    
    def _on_image_rotation_change(self, value):
        """图像旋转滑块变化"""
        angle = int(float(value))
        self.image_rotation = angle
        self.image_rotation_label.config(text=f"{angle}°")
        self.draw_paper()
    
    def _rotate_image(self, angle, absolute=False):
        """旋转图像"""
        if absolute:
            self.image_rotation = angle
        else:
            self.image_rotation = (self.image_rotation + angle) % 360
            if self.image_rotation > 180:
                self.image_rotation -= 360
        self.image_rotation_var.set(self.image_rotation)
        self.image_rotation_label.config(text=f"{self.image_rotation}°")
        self.draw_paper()
    
    def _on_ground_rotation_change(self, value):
        """Ground旋转滑块变化"""
        if self.ground_locked:
            return
        angle = int(float(value))
        self.ground_rotation = angle
        self.ground_rotation_label.config(text=f"{angle}°")
        self.draw_paper()
    
    def _rotate_point(self, x, y, cx, cy, angle_deg):
        """绕中心点旋转一个点"""
        import math
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        dx = x - cx
        dy = y - cy
        new_x = cx + dx * cos_a - dy * sin_a
        new_y = cy + dx * sin_a + dy * cos_a
        return new_x, new_y
    
    def _rotate_contours(self, contours, angle_deg, center_x, center_y):
        """旋转所有轮廓"""
        if angle_deg == 0:
            return contours
        
        rotated = []
        for contour in contours:
            rotated_contour = []
            for x, y in contour:
                rx, ry = self._rotate_point(x, y, center_x, center_y, angle_deg)
                rotated_contour.append((rx, ry))
            rotated.append(rotated_contour)
        return rotated
    
    def draw_ground(self):
        """在预览画布上绘制ground参照图"""
        self.preview_canvas.delete('ground')
        
        if self.ground_image is None or not self.ground_visible:
            return
        
        if not hasattr(self, 'preview_scale') or self.preview_scale == 0:
            return
        
        # 计算图片显示尺寸
        h, w = self.ground_image.shape[:2]
        
        # 图片在纸张上的实际尺寸(mm)
        img_width_mm = w * self.ground_scale
        img_height_mm = h * self.ground_scale
        
        # 转换为画布像素
        display_width = img_width_mm * self.preview_scale
        display_height = img_height_mm * self.preview_scale
        
        # 限制最大尺寸防止内存溢出
        MAX_SIZE = 2000
        scale_ratio = 1.0
        if display_width > MAX_SIZE or display_height > MAX_SIZE:
            scale_ratio = min(MAX_SIZE / max(display_width, 1), MAX_SIZE / max(display_height, 1))
            display_width *= scale_ratio
            display_height *= scale_ratio
        
        display_width = max(1, int(display_width))
        display_height = max(1, int(display_height))
        
        # 计算图片左上角位置
        x = self.preview_paper_x + self.ground_offset_x * self.preview_scale
        y = self.preview_paper_y + self.ground_offset_y * self.preview_scale
        
        # 转换图片
        img_rgb = cv2.cvtColor(self.ground_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        # 计算图片中心（未旋转时）
        center_x = x + display_width / 2
        center_y = y + display_height / 2
        
        # 旋转图片
        if self.ground_rotation != 0:
            pil_img = pil_img.rotate(-self.ground_rotation, expand=True, resample=Image.Resampling.BICUBIC)
        
        # 设置透明度
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')
        alpha = Image.new('L', pil_img.size, 128)  # 50%透明
        pil_img.putalpha(alpha)
        
        self.ground_photo = ImageTk.PhotoImage(pil_img)
        
        # 计算旋转后的显示位置（以中心旋转）
        new_w, new_h = pil_img.size
        draw_x = center_x - new_w / 2
        draw_y = center_y - new_h / 2
        
        self.preview_canvas.create_image(draw_x, draw_y, image=self.ground_photo, anchor=tk.NW, tags='ground')
        
        # 绘制定标点（只在旋转为0时显示，因为定标点只能在旋转为0时设置）
        if self.ground_rotation == 0:
            for i, point in enumerate(self.ground_points):
                # 定标点是图片像素坐标，转换为画布坐标
                mm_x = point[0] * self.ground_scale + self.ground_offset_x
                mm_y = point[1] * self.ground_scale + self.ground_offset_y
                px = self.preview_paper_x + mm_x * self.preview_scale
                py = self.preview_paper_y + mm_y * self.preview_scale
                
                color = 'red' if i == 0 else 'blue'
                self.preview_canvas.create_oval(px-6, py-6, px+6, py+6, fill=color, outline='white', width=2, tags='ground')
                label = "上" if i == 0 else "下"
                self.preview_canvas.create_text(px+12, py, text=label, fill=color, font=('Arial', 10, 'bold'), tags='ground')
        else:
            # 旋转时显示提示
            if len(self.ground_points) > 0:
                self.preview_canvas.create_text(
                    center_x, center_y - new_h/2 - 15,
                    text="定标点已隐藏(旋转中)", 
                    fill='orange', font=('Arial', 9), tags='ground'
                )
        
        # 绘制边框（如果未选中）
        if self.selected_object != 'ground':
            self.preview_canvas.create_rectangle(
                draw_x, draw_y, draw_x + new_w, draw_y + new_h,
                outline='orange' if not self.ground_locked else 'green',
                width=2, dash=(5, 5) if not self.ground_locked else None,
                tags='ground'
            )
        else:
            # 如果选中了ground，绘制变换控制框
            min_x_mm = self.ground_offset_x
            min_y_mm = self.ground_offset_y
            max_x_mm = self.ground_offset_x + w * self.ground_scale
            max_y_mm = self.ground_offset_y + h * self.ground_scale
            center_x_mm = (min_x_mm + max_x_mm) / 2
            center_y_mm = (min_y_mm + max_y_mm) / 2
            self._draw_transform_box('ground', min_x_mm, min_y_mm, max_x_mm, max_y_mm, center_x_mm, center_y_mm)
        
        # 绘制零点标记
        self.preview_canvas.create_line(
            self.preview_paper_x - 10, self.preview_paper_y,
            self.preview_paper_x + 20, self.preview_paper_y,
            fill='red', width=2, tags='ground'
        )
        self.preview_canvas.create_line(
            self.preview_paper_x, self.preview_paper_y - 10,
            self.preview_paper_x, self.preview_paper_y + 20,
            fill='red', width=2, tags='ground'
        )
        self.preview_canvas.create_text(
            self.preview_paper_x + 25, self.preview_paper_y + 10,
            text="(0,0)", fill='red', font=('Arial', 9), tags='ground'
        )
        ###

    def _process_ocr_image(self, img):
        try:
            from PIL import Image, ImageDraw, ImageFont
            from skimage.morphology import skeletonize, thin
        except ImportError as e:
            messagebox.showerror("依赖缺失", str(e))
            return None

        try:
            self.status_label.config(text="初始化 OCR...", foreground='blue')
            self.root.update()

            ocr = self._get_ocr_engine(show_error=True)
            if ocr is None:
                return None

            if img is None:
                return None

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            h, w = img.shape[:2]
            self.debug_log(f"OCR输入: img=({w}x{h})")

            min_size = 2400
            scale_factor = 1.0
            img_for_ocr = img
            if max(h, w) < min_size:
                scale_factor = float(min_size) / float(max(h, w))
                img_for_ocr = cv2.resize(
                    img,
                    (int(w * scale_factor), int(h * scale_factor)),
                    interpolation=cv2.INTER_CUBIC
                )
            self.debug_log(f"OCR预处理: scale_factor={scale_factor:.4f}")

            variants = [("原图", img_for_ocr)]
            try:
                gray = cv2.cvtColor(img_for_ocr, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                variants.append(("增强灰度", cv2.cvtColor(cv2.convertScaleAbs(gray, alpha=1.8, beta=0), cv2.COLOR_GRAY2BGR)))

                binary = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 10
                )
                variants.append(("自适应二值", cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))
                variants.append(("自适应反色", cv2.cvtColor(255 - binary, cv2.COLOR_GRAY2BGR)))
            except:
                pass

            try:
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
                sharpened = cv2.filter2D(img_for_ocr, -1, kernel)
                variants.append(("锐化", sharpened))
            except:
                pass

            chosen_variant = None
            ocr_items = []
            for variant_name, img_variant in variants:
                self.status_label.config(text=f"OCR 识别中（{variant_name}）...", foreground='blue')
                self.root.update()

                ocr_ret = None
                try:
                    ocr_ret = self._ocr_call(ocr, img_variant)
                except:
                    ocr_ret = None

                ocr_items = self._ocr_extract_items(ocr_ret)
                self.debug_log(f"OCR返回: variant={variant_name} items={len(ocr_items)}")
                if ocr_items:
                    chosen_variant = variant_name
                    break

            if not ocr_items:
                self.status_label.config(text="OCR未识别到文字，进行单线化提取...", foreground='orange')
                self.root.update()
                self.debug_log("OCR返回为空: 进入单线化fallback")

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                try:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray = clahe.apply(gray)
                except:
                    pass
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

                binary_inv = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 21, 8
                )
                binary_norm = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 21, 8
                )

                inv_ratio = float(np.count_nonzero(binary_inv)) / float(binary_inv.size)
                norm_ratio = float(np.count_nonzero(binary_norm == 0)) / float(binary_norm.size)
                binary = binary_inv if inv_ratio < norm_ratio else (255 - binary_norm)

                kernel = np.ones((2, 2), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                min_keep = max(20, int(binary.size * 0.00002))
                binary = self._remove_small_components(binary, min_keep)

                skeleton_img = (thin(binary > 0).astype(np.uint8) * 255)
                nonzero = int(np.count_nonzero(skeleton_img))
                self.debug_log(f"fallback结果: skeleton_nonzero={nonzero}")

                if nonzero == 0:
                    messagebox.showwarning("提示", "未识别到文字，骨架化也没有提取到有效笔画。")
                    return None
                try:
                    self._last_ocr_input_bgr = img.copy()
                    self._last_ocr_text_binary = None
                    self._last_ocr_base_binary = binary.copy()
                    self._last_ocr_output_skeleton = skeleton_img.copy()
                except:
                    pass
                return skeleton_img

            output_img = Image.new('L', (w, h), 255)
            draw = ImageDraw.Draw(output_img)

            font_map = {
                'simhei': 'C:/Windows/Fonts/simhei.ttf',
                'simsun': 'C:/Windows/Fonts/simsun.ttc',
                'msyh': 'C:/Windows/Fonts/msyh.ttc',
                'arial': 'C:/Windows/Fonts/arial.ttf',
                'times': 'C:/Windows/Fonts/times.ttf',
            }
            font_name = self.ocr_font_var.get().lower()
            font_path = font_map.get(font_name)

            def _load_font(font_size):
                try:
                    if font_path:
                        return ImageFont.truetype(font_path, int(font_size))
                except:
                    pass
                for p in font_map.values():
                    try:
                        return ImageFont.truetype(p, int(font_size))
                    except:
                        continue
                return ImageFont.load_default()

            def _fit_font_size(text, target_w, target_h):
                if target_w <= 1 or target_h <= 1:
                    return 1
                lo = 1
                hi = min(600, max(8, int(target_h * 3)))
                best = 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    font = _load_font(mid)
                    tb = draw.textbbox((0, 0), text, font=font)
                    tw = tb[2] - tb[0]
                    th = tb[3] - tb[1]
                    if tw <= target_w and th <= target_h:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return best

            self.status_label.config(text="重绘文字...", foreground='blue')
            self.root.update()

            filtered_ocr_items = []
            for bbox, text, _confidence in ocr_items:
                try:
                    if not text or not str(text).strip():
                        continue
                    parts = re.findall(r"[A-Za-z\u4e00-\u9fff]+", str(text))
                    if not parts:
                        continue
                    filtered_text = "".join(parts)
                    if not filtered_text:
                        continue
                    filtered_ocr_items.append((bbox, filtered_text))
                except:
                    continue

            def _tokenize_zh_en(s):
                tokens = []
                buf = []
                for ch in s:
                    if 'A' <= ch <= 'Z' or 'a' <= ch <= 'z':
                        buf.append(ch)
                        continue
                    if buf:
                        tokens.append(''.join(buf))
                        buf = []
                    if '\u4e00' <= ch <= '\u9fff':
                        tokens.append(ch)
                if buf:
                    tokens.append(''.join(buf))
                return tokens

            draw_jobs = []
            for bbox, text in filtered_ocr_items:
                try:
                    pts = (np.array(bbox, dtype=np.float32) / float(scale_factor))
                    x1 = int(np.floor(np.min(pts[:, 0])))
                    x2 = int(np.ceil(np.max(pts[:, 0])))
                    y1 = int(np.floor(np.min(pts[:, 1])))
                    y2 = int(np.ceil(np.max(pts[:, 1])))

                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    tokens = _tokenize_zh_en(text)
                    if len(tokens) <= 1:
                        draw_jobs.append((x1, y1, x2, y2, text))
                        continue

                    box_w = x2 - x1
                    box_h = y2 - y1
                    horizontal = box_w >= box_h
                    nominal = int(max(10, min(200, min(box_w, box_h) * 0.9)))
                    nominal_font = _load_font(nominal)
                    weights = []
                    for t in tokens:
                        try:
                            tb = draw.textbbox((0, 0), t, font=nominal_font)
                            tw = max(1, tb[2] - tb[0])
                            th = max(1, tb[3] - tb[1])
                            weights.append(tw if horizontal else th)
                        except:
                            weights.append(1)
                    total_w = float(sum(weights)) if weights else 1.0

                    if horizontal:
                        cursor = x1
                        for t, wt in zip(tokens, weights):
                            seg = int(round(box_w * (float(wt) / total_w)))
                            seg = max(1, seg)
                            nx1 = cursor
                            nx2 = min(x2, cursor + seg)
                            if nx2 > nx1:
                                draw_jobs.append((nx1, y1, nx2, y2, t))
                            cursor = nx2
                        if cursor < x2 and draw_jobs:
                            lx1, ly1, lx2, ly2, lt = draw_jobs[-1]
                            if ly1 == y1 and ly2 == y2 and lx2 <= x2:
                                draw_jobs[-1] = (lx1, ly1, x2, ly2, lt)
                    else:
                        cursor = y1
                        for t, wt in zip(tokens, weights):
                            seg = int(round(box_h * (float(wt) / total_w)))
                            seg = max(1, seg)
                            ny1 = cursor
                            ny2 = min(y2, cursor + seg)
                            if ny2 > ny1:
                                draw_jobs.append((x1, ny1, x2, ny2, t))
                            cursor = ny2
                        if cursor < y2 and draw_jobs:
                            lx1, ly1, lx2, ly2, lt = draw_jobs[-1]
                            if lx1 == x1 and lx2 == x2 and ly2 <= y2:
                                draw_jobs[-1] = (lx1, ly1, lx2, y2, lt)
                except:
                    continue

            drawn = 0
            for x1, y1, x2, y2, text in draw_jobs:
                try:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    font_size = _fit_font_size(text, box_w, box_h)
                    font = _load_font(font_size)
                    tb = draw.textbbox((0, 0), text, font=font)
                    tw = tb[2] - tb[0]
                    th = tb[3] - tb[1]
                    draw_x = int(x1 + (box_w - tw) / 2 - tb[0])
                    draw_y = int(y1 + (box_h - th) / 2 - tb[1])
                    draw.text((draw_x, draw_y), text, font=font, fill=0)
                    drawn += 1
                except:
                    continue

            self.debug_log(f"OCR绘制: variant={chosen_variant} items={len(filtered_ocr_items)} drawn={drawn}")

            output_cv = np.array(output_img)
            _, text_binary = cv2.threshold(output_cv, 128, 255, cv2.THRESH_BINARY_INV)
            try:
                text_mask_margin_px = 1
                k = int(text_mask_margin_px) * 2 + 1
                text_mask = cv2.dilate(text_binary, np.ones((k, k), np.uint8), iterations=1)
            except:
                text_mask = text_binary.copy()

            self.status_label.config(text="文字骨架化...", foreground='blue')
            self.root.update()

            text_skeleton_img = (skeletonize(text_binary > 0).astype(np.uint8) * 255)

            self.status_label.config(text="单线化未识别区域...", foreground='blue')
            self.root.update()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            binary_inv = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 21, 8
            )
            binary_norm = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 21, 8
            )
            inv_ratio = float(np.count_nonzero(binary_inv)) / float(binary_inv.size)
            norm_ratio = float(np.count_nonzero(binary_norm == 0)) / float(binary_norm.size)
            base_binary = binary_inv if inv_ratio < norm_ratio else (255 - binary_norm)

            kernel = np.ones((2, 2), np.uint8)
            base_binary = cv2.morphologyEx(base_binary, cv2.MORPH_OPEN, kernel, iterations=1)
            min_keep = max(20, int(base_binary.size * 0.00002))
            base_binary = self._remove_small_components(base_binary, min_keep)

            exclude_mask = text_mask if np.any(text_mask > 0) else np.zeros_like(base_binary)
            non_text_region = cv2.bitwise_and(base_binary, cv2.bitwise_not(exclude_mask))
            non_text_thin_img = (thin(non_text_region > 0).astype(np.uint8) * 255)

            combined = cv2.bitwise_or(text_skeleton_img, non_text_thin_img)
            combined = self._remove_small_components(combined, min_keep)
            self.debug_log(f"OCR合并结果: variant={chosen_variant} items={len(filtered_ocr_items)} skeleton_nonzero={int(np.count_nonzero(combined))}")
            try:
                self._last_ocr_input_bgr = img.copy()
                self._last_ocr_text_binary = text_binary.copy()
                self._last_ocr_base_binary = base_binary.copy()
                self._last_ocr_output_skeleton = combined.copy()
            except:
                pass
            return combined

        except Exception as e:
            messagebox.showerror("OCR处理失败", str(e))
            return None


def main():
    root = tk.Tk()
    app = GCodeApp(root)
    
    def on_closing():
        if app.grbl.connected:
            app.grbl.disconnect()
        if app.status_timer:
            root.after_cancel(app.status_timer)
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
