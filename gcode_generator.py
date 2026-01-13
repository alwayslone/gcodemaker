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
import threading
import queue
import time
import re

# 尝试导入串口库
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# 尝试导入OCR库 (PaddleOCR)
try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


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
        
        # 右下：控制台
        console_frame = ttk.LabelFrame(bottom_right, text="控制台")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.console = scrolledtext.ScrolledText(console_frame, height=10, font=('Consolas', 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.config(state=tk.DISABLED)
        
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
    
  ###  def _process_ocr_image(self, img):
        """使用PaddleOCR识别文字并转换为骨架图"""
        if not OCR_AVAILABLE:
            messagebox.showerror("错误", "未安装PaddleOCR库，请运行: pip install paddlepaddle paddleocr")
            return None
        
        try:
            from PIL import ImageFont, ImageDraw, ImageEnhance
            from skimage.morphology import skeletonize
            import warnings
            warnings.filterwarnings("ignore")
            
            # 初始化PaddleOCR（支持中英文）
            self.status_label.config(text="正在初始化OCR...", foreground='blue')
            self.root.update()
            
            # 使用PaddleOCR，启用方向分类器
            ocr = PaddleOCR(use_angle_cls=True, lang='ch')
            
            # 图像预处理：提高识别精度
            self.status_label.config(text="正在预处理图像...", foreground='blue')
            self.root.update()
            
            # 记录原始尺寸
            h, w = img.shape[:2]
            
            # 转换为RGB图像
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # 放大小图像以提高识别率
            min_size = 800
            if max(h, w) < min_size:
                scale_factor = min_size / max(h, w)
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            else:
                scale_factor = 1.0
            
            # 增强对比度
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.3)
            
            # 增强锐度
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(1.5)
            
            img_enhanced = np.array(pil_img)
            
            # OCR识别
            self.status_label.config(text="正在识别文字...", foreground='blue')
            self.root.update()
            
            results = ocr.ocr(img_enhanced)
            
            if not results or not results[0]:
                messagebox.showwarning("警告", "未识别到文字")
                return None
            
            # 创建与原图相同尺寸的白色图像
            output_img = Image.new('L', (w, h), 255)
            draw = ImageDraw.Draw(output_img)
            
            # 字体映射
            font_name = self.ocr_font_var.get()
            font_map = {
                'simhei': 'C:/Windows/Fonts/simhei.ttf',
                'simsun': 'C:/Windows/Fonts/simsun.ttc',
                'msyh': 'C:/Windows/Fonts/msyh.ttc',
                'arial': 'C:/Windows/Fonts/arial.ttf',
                'times': 'C:/Windows/Fonts/times.ttf',
            }
            
            self.status_label.config(text="正在重绘文字...", foreground='blue')
            self.root.update()
            
            # 绘制每个识别到的文字
            for line in results[0]:
                try:
                    # PaddleOCR返回格式可能是: [bbox, (text, confidence)]
                    bbox = line[0]
                    text_info = line[1]
                    
                    # 解析文字和置信度
                    if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                        text = str(text_info[0])
                        try:
                            confidence = float(text_info[1])
                        except:
                            confidence = 0.9  # 默认置信度
                    elif isinstance(text_info, str):
                        text = text_info
                        confidence = 0.9
                    else:
                        text = str(text_info)
                        confidence = 0.9
                except:
                    continue
                
                if not text or len(text.strip()) == 0:
                    continue
                
                # 获取边界框坐标（还原到原始尺寸）
                pts = np.array(bbox, dtype=np.float32) / scale_factor
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]
                x1, x2 = int(min(x_coords)), int(max(x_coords))
                y1, y2 = int(min(y_coords)), int(max(y_coords))
                
                # 计算文字大小（基于边界框高度）
                text_height = y2 - y1
                text_width = x2 - x1
                
                # 根据文字长度和宽度计算字体大小
                chars = len(text)
                if chars > 0:
                    char_width = text_width / chars
                    font_size = max(12, int(min(text_height * 0.9, char_width * 1.2)))
                else:
                    font_size = max(12, int(text_height * 0.85))
                
                # 加载字体
                font = None
                font_path = font_map.get(font_name.lower())
                if font_path:
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                    except:
                        pass
                
                if font is None:
                    for path in font_map.values():
                        try:
                            font = ImageFont.truetype(path, font_size)
                            break
                        except:
                            continue
                
                if font is None:
                    font = ImageFont.load_default()
                
                # 绘制文字（黑色字在白色背景）
                draw.text((x1, y1), text, font=font, fill=0)
            
            # 转换为OpenCV图像
            output_cv = np.array(output_img)
            
            # 二值化
            _, binary = cv2.threshold(output_cv, 128, 255, cv2.THRESH_BINARY_INV)
            
            # 骨架化文字
            self.status_label.config(text="正在骨架化...", foreground='blue')
            self.root.update()
            
            skeleton = skeletonize(binary > 0)
            skeleton_img = (skeleton * 255).astype(np.uint8)
            
            # 同时对原图做边缘检测，保留其他内容
            self.status_label.config(text="正在提取其他轮廓...", foreground='blue')
            self.root.update()
            
            # 边缘检测原图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 合并文字骨架和边缘检测结果
            combined = cv2.bitwise_or(skeleton_img, edges)
            
            # 返回合并后的图像
            return combined
            
        except ImportError:
            messagebox.showerror("错误", "请安装 scikit-image: pip install scikit-image")
            return None
        except Exception as e:
            messagebox.showerror("错误", f"OCR处理失败: {str(e)}")
            return None
    
    def process_image(self):
        # 判断要处理的图像
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图片")
            return
        
        # 获取旋转后的原图
        rotated_img = self._get_rotated_image()
        
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
        
        try:
            # 检查是否启用OCR模式
            if self.ocr_mode_var.get():
                # OCR模式：识别文字并转换为骨架图
                skeleton_img = self._process_ocr_image(img_to_process)
                if skeleton_img is None:
                    return
                
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
            
            self.gcode_lines = []  # 清空旧的GCode
            
            self.draw_paper()
            self.notebook.select(1)
            
        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {str(e)}")
    
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
            
            # === 步骤2: 自适应二值化 ===
            # 使用自适应阈值处理不同光照条件
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # === 步骤3: 检测文字区域（可选） ===
            text_mask = None
            if text_sensitivity > 0 and OCR_AVAILABLE:
                self.status_label.config(text="正在检测文字区域...", foreground='blue')
                self.root.update()
                text_mask = self._detect_text_regions(img_to_process, text_sensitivity)
            
            # === 步骤4: 分区域处理 ===
            self.status_label.config(text="正在进行单线化处理...", foreground='blue')
            self.root.update()
            
            # 创建输出图像
            skeleton_result = np.zeros_like(binary)
            
            if text_mask is not None:
                # 文字区域：使用骨架化
                text_region = binary & text_mask
                if np.any(text_region > 0):
                    text_skeleton = skeletonize(text_region > 0)
                    skeleton_result = skeleton_result | (text_skeleton.astype(np.uint8) * 255)
                
                # 非文字区域：使用细化
                non_text_region = binary & ~text_mask
                if np.any(non_text_region > 0):
                    non_text_thinned = thin(non_text_region > 0)
                    skeleton_result = skeleton_result | (non_text_thinned.astype(np.uint8) * 255)
            else:
                # 没有文字检测，对整个图像进行细化
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
    
    def _detect_text_regions(self, img, sensitivity=0.5):
        """检测图像中的文字区域，返回文字区域的掩码"""
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # 使用PaddleOCR检测文字
            ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
            
            # 转换为RGB
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            results = ocr.ocr(img_rgb, cls=True)
            
            if results and results[0]:
                for line in results[0]:
                    try:
                        bbox = line[0]
                        text_info = line[1]
                        
                        # 获取置信度
                        if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                            confidence = float(text_info[1])
                        else:
                            confidence = 0.9
                        
                        # 根据灵敏度过滤
                        if confidence < (1 - sensitivity):
                            continue
                        
                        # 获取边界框
                        pts = np.array(bbox, dtype=np.int32)
                        
                        # 扩展边界框
                        x_coords = [p[0] for p in pts]
                        y_coords = [p[1] for p in pts]
                        x1, x2 = max(0, min(x_coords) - 5), min(w, max(x_coords) + 5)
                        y1, y2 = max(0, min(y_coords) - 5), min(h, max(y_coords) + 5)
                        
                        # 填充文字区域
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
        """工业级稳定版：OCR → 重绘 → 骨架化（永不因 OCR 输出崩溃）"""
        if not OCR_AVAILABLE:
            messagebox.showerror(
                "错误",
                "未安装 PaddleOCR，请运行:\n\npip install paddlepaddle paddleocr"
            )
            return None

        try:
            from PIL import Image, ImageDraw, ImageFont, ImageEnhance
            from skimage.morphology import skeletonize
            import warnings
            warnings.filterwarnings("ignore")

            # ------------------------------
            # 1. 初始化 OCR
            # ------------------------------
            self.status_label.config(text="初始化 OCR...", foreground='blue')
            self.root.update()

            ocr = PaddleOCR(use_angle_cls=True, lang='ch')

            # ------------------------------
            # 2. 图像预处理
            # ------------------------------
            self.status_label.config(text="图像预处理...", foreground='blue')
            self.root.update()

            h, w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # 放大过小图片
            min_size = 800
            scale_factor = 1.0
            if max(h, w) < min_size:
                scale_factor = min_size / max(h, w)
                pil_img = pil_img.resize(
                    (int(w * scale_factor), int(h * scale_factor)),
                    Image.Resampling.LANCZOS
                )

            # 增强
            pil_img = ImageEnhance.Contrast(pil_img).enhance(1.3)
            pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)

            img_enhanced = np.array(pil_img)

            # ------------------------------
            # 3. OCR 识别
            # ------------------------------
            self.status_label.config(text="OCR 识别中...", foreground='blue')
            self.root.update()

            ocr_result = ocr.ocr(img_enhanced)

            if not ocr_result or not ocr_result[0]:
                messagebox.showwarning("提示", "未识别到文字")
                return None

            # ------------------------------
            # 4. 创建输出画布
            # ------------------------------
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

            self.status_label.config(text="重绘文字...", foreground='blue')
            self.root.update()

            # ------------------------------
            # 5. 安全解析并绘制 OCR 行
            # ------------------------------
            for idx, line in enumerate(ocr_result[0]):
                try:
                    # ---------- 解析 bbox ----------
                    bbox = line[0]
                    pts = np.array(bbox, dtype=np.float32) / scale_factor

                    x_coords = pts[:, 0]
                    y_coords = pts[:, 1]
                    x1, x2 = int(x_coords.min()), int(x_coords.max())
                    y1, y2 = int(y_coords.min()), int(y_coords.max())

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # ---------- 解析 text / confidence（防御性） ----------
                    text = ""
                    confidence = 0.9

                    text_info = line[1]

                    if isinstance(text_info, (list, tuple)):
                        if len(text_info) >= 1:
                            text = str(text_info[0])

                        if len(text_info) >= 2 and isinstance(text_info[1], (int, float)):
                            confidence = float(text_info[1])
                    elif isinstance(text_info, str):
                        text = text_info
                    else:
                        text = str(text_info)

                    if not text.strip():
                        continue

                    # ---------- 字体大小估算 ----------
                    box_h = y2 - y1
                    box_w = x2 - x1
                    char_count = max(len(text), 1)

                    est_char_w = box_w / char_count
                    font_size = max(
                        12,
                        int(min(box_h * 0.9, est_char_w * 1.3))
                    )

                    # ---------- 加载字体（永不失败） ----------
                    font = None
                    try:
                        if font_path:
                            font = ImageFont.truetype(font_path, font_size)
                    except:
                        font = None

                    if font is None:
                        for p in font_map.values():
                            try:
                                font = ImageFont.truetype(p, font_size)
                                break
                            except:
                                continue

                    if font is None:
                        font = ImageFont.load_default()

                    # ---------- 绘制 ----------
                    draw.text((x1, y1), text, font=font, fill=0)

                except Exception as e:
                    # 单行 OCR 出错，直接跳过，绝不影响整体
                    print(f"[OCR WARN] line {idx} skipped: {e}")
                    continue

            # ------------------------------
            # 6. 二值化 + 骨架化
            # ------------------------------
            output_cv = np.array(output_img)
            _, binary = cv2.threshold(
                output_cv, 128, 255, cv2.THRESH_BINARY_INV
            )

            self.status_label.config(text="文字骨架化...", foreground='blue')
            self.root.update()

            skeleton = skeletonize(binary > 0)
            skeleton_img = (skeleton * 255).astype(np.uint8)

            # ------------------------------
            # 7. 原图边缘融合
            # ------------------------------
            self.status_label.config(text="提取原图轮廓...", foreground='blue')
            self.root.update()

            return skeleton_img

        except ImportError as e:
            messagebox.showerror("依赖缺失", str(e))
            return None

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
