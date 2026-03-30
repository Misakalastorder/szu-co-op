'''
实时读取串口传感器数据并驱动真实 LinkerHand 机器手（多进程）
参考自 new_low.py, main_h5_realtime.py
'''
import multiprocessing as mp
import re
import time
from queue import Empty, Full
from typing import Dict, Optional

import numpy as np
import serial

# 项目导入
from LinkerHand.linker_hand_api import LinkerHandApi
from LinkerHand.utils.load_write_yaml import LoadWriteYaml
from LinkerHand.utils.color_msg import ColorMsg

# 18个关节的角度限制 (弧度) - 来自 main_h5_realtime.py
angle_limit_rob = [
    [0.0, 0.0],           # hand_base_link
    [-0.18, 0.18],       # index_mcp_roll
    [0.0, 1.57],         # index_mcp_pitch
    [0.0, 1.57],         # index_pip
    [-0.18, 0.18],       # middle_mcp_roll
    [0.0, 1.57],         # middle_mcp_pitch
    [0.0, 1.57],         # middle_pip
    [-0.18, 0.18],       # ring_mcp_roll
    [0.0, 1.57],         # ring_mcp_pitch
    [0.0, 1.57],         # ring_pip
    [-0.18, 0.18],       # pinky_mcp_roll
    [0.0, 1.57],         # pinky_mcp_pitch
    [0.0, 1.57],         # pinky_pip
    [-0.6, 0.6],         # thumb_cmc_roll
    [0.0, 1.6],          # thumb_cmc_yaw
    [0.0, 1.0],          # thumb_cmc_pitch
    [0.0, 1.57],         # thumb_mcp
    [0.0, 1.57]          # thumb_ip
]

# 25个驱动器映射到 18 个关节 - 来自 main_h5_realtime.py
joint_map = {
    0: 15, 1: 2, 2: 5, 3: 8, 4: 11, 5: 14, 6: 1, 7: 4, 8: 7, 9: 10, 10: 13, 
    11: 0, 12: 0, 13: 0, 14: 0, 15: 16, 16: 0, 17: 0, 18: 0, 19: 0,
    20: 17, 21: 3, 22: 6, 23: 9, 24: 12
}

# 传感器标定配置（与 new_low.py 保持一致）
CALIBRATION_CONFIG = {
    'thumb': [
        (0, 2330, 0),
        (2540, 2730, 30),
        (2730, 2870, 60),
        (2870, 4095, 90)
    ],
    'index': [
        (0, 1985, 0),
        (2000, 2160, 30),
        (2160, 2320, 60),
        (2320, 4095, 90)
    ],
    'middle': [
        (0, 2020, 0),
        (2020, 2090, 30),
        (2090, 2180, 60),
        (2280, 4095, 90)
    ],
    'ring': [
        (0, 1940, 0),
        (1990, 2110, 30),
        (2110, 2210, 60),
        (2210, 4095, 90)
    ],
    'pinky': [
        (0, 2400, 0),
        (2500, 2600, 30),
        (2600, 2680, 60),
        (2680, 4095, 90)
    ]
}

# 数字ID到手指名称
ID_TO_FINGER = {
    1: 'thumb',
    2: 'index',
    3: 'middle',
    4: 'ring',
    5: 'back_hand',
    6: 'pinky'
}

# 手指对应AD索引
AD_INDEX_FOR_FINGER = {
    'thumb': 0,
    'index': 1,
    'middle': 2,
    'ring': 3,
    'pinky': 4
}

# 传感器手指到 Linker 18关节索引
finger_to_joint_indices = {
    "index":  [1, 2, 3],  
    "middle": [4, 5, 6],
    "ring":   [7, 8, 9],
    "pinky":  [10, 11, 12],
    "thumb":  [13, 14, 15, 16, 17],
}

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

AD_PATTERN = re.compile(r'AD:\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)')
MPU_PATTERN = re.compile(
    r'M(\d+):\s*'
    r'P=([-+]?\d+\.\d+)\s+'
    r'R=([-+]?\d+\.\d+)\s+'
    r'Y=([-+]?\d+\.\d+)\s+'
    r'aX=([-+]?\d+\.\d+)\s+'
    r'aY=([-+]?\d+\.\d+)\s+'
    r'aZ=([-+]?\d+\.\d+)'
)


def calibrate_sensor(finger_name: str, ad_value: int) -> Optional[float]:
    if finger_name not in CALIBRATION_CONFIG:
        return None

    calibration_ranges = CALIBRATION_CONFIG[finger_name]
    for min_ad, max_ad, angle in calibration_ranges:
        if min_ad <= ad_value < max_ad:
            return float(angle)

    return float(calibration_ranges[-1][2])


def try_put_latest(queue, frame: Dict):
    try:
        queue.put_nowait(frame)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        queue.put_nowait(frame)

def unit(num):
    return 0 if num < 0 else 255 if num > 255 else int(num)

def trans2realworld_linker(angle_rad_18):
    '''
    将 18 个关节的弧度转换为 25 路驱动器的 0-255 整数值
    '''
    angle_norm = [0] * 18
    for i in range(18):
        low, high = angle_limit_rob[i]
        norm = (angle_rad_18[i] - low) / (high - low) if high > low else 0.0
        angle_norm[i] = unit(norm * 255)
    
    angle_mapped = [0] * 25
    for drive_idx, joint_idx in joint_map.items():
        angle_mapped[drive_idx] = 255 - angle_norm[joint_idx]

    # 特点：Linker 手特定电机取反
    # for idx in [0, 5, 10, 15, 20]:
    #     angle_mapped[idx] = 255 - angle_mapped[idx]
        
    return angle_mapped


def build_angle_rad_18(finger2vals: Dict[str, Dict[str, float]]) -> np.ndarray:
    angle_rad_18 = np.zeros(18, dtype=np.float32)
    for f, indices in finger_to_joint_indices.items():
        data = finger2vals.get(f)
        if not data:
            continue

        if f == "thumb":
            angle_rad_18[indices[0]] = np.deg2rad(data.get("yaw", 0))
            angle_rad_18[indices[1]] = np.deg2rad(data.get("roll", 0))
            angle_rad_18[indices[2]] = np.deg2rad(data.get("pitch", 0))
            angle_rad_18[indices[3]] = np.deg2rad(data.get("angle", 0))
            angle_rad_18[indices[4]] = np.deg2rad(data.get("angle", 0))
        else:
            angle_rad_18[indices[0]] = np.deg2rad(data.get("yaw", 0))
            angle_rad_18[indices[1]] = np.deg2rad(data.get("pitch", 0))
            angle_rad_18[indices[2]] = np.deg2rad(data.get("angle", 0))

    return angle_rad_18


def collector_process(frame_queue, stop_event, port: str, baudrate: int):
    ser = None
    current_ad_values = None
    current_data_group = {}

    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            stopbits=1,
            parity=serial.PARITY_NONE,
            timeout=0.1
        )
        print(f"[采集进程] 串口已打开: {port}, 波特率: {baudrate}")

        while not stop_event.is_set():
            serial_line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not serial_line:
                continue

            ad_match = AD_PATTERN.search(serial_line)
            if ad_match:
                current_ad_values = [
                    int(ad_match.group(1)),
                    int(ad_match.group(2)),
                    int(ad_match.group(3)),
                    int(ad_match.group(4)),
                    int(ad_match.group(5))
                ]
                current_data_group = {}
                continue

            mpu_match = MPU_PATTERN.search(serial_line)
            if not mpu_match or current_ad_values is None:
                continue

            mpu_id = int(mpu_match.group(1))
            finger_name = ID_TO_FINGER.get(mpu_id, f"unknown_{mpu_id}")
            if finger_name not in FINGERS:
                continue

            ad_idx = AD_INDEX_FOR_FINGER[finger_name]
            ad_value = current_ad_values[ad_idx]
            current_data_group[finger_name] = {
                "angle": calibrate_sensor(finger_name, ad_value),
                "pitch": round(float(mpu_match.group(2)), 2),
                "roll": round(float(mpu_match.group(3)), 2),
                "yaw": round(float(mpu_match.group(4)), 2)
            }

            if len(current_data_group) == 5:
                frame = {
                    "timestamp": time.time(),
                    "fingers": current_data_group
                }
                try_put_latest(frame_queue, frame)
                current_data_group = {}

    except Exception as e:
        print(f"[采集进程] 异常: {e}")
        stop_event.set()
    finally:
        if ser is not None and ser.is_open:
            ser.close()
            print("[采集进程] 串口已关闭")


def control_process(frame_queue, stop_event):
    controller = None
    sample_count = 0
    last_log_time = time.time()

    try:
        controller = HandController()
        print("[控制进程] LinkerHand 控制初始化完成")

        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.2)
            except Empty:
                continue

            finger2vals = frame.get("fingers", {})
            angle_rad_18 = build_angle_rad_18(finger2vals)
            real_pos = trans2realworld_linker(angle_rad_18)
            controller.control_hand(real_pos)

            sample_count += 1
            now = time.time()
            if now - last_log_time >= 1.0:
                print(f"[控制进程] 已控制 {sample_count} 帧")
                last_log_time = now

    except Exception as e:
        print(f"[控制进程] 异常: {e}")
        stop_event.set()
    finally:
        if controller is not None:
            controller.close()
        print("[控制进程] 已安全退出")

class HandController:
    def __init__(self):
        self.yaml = LoadWriteYaml()
        self.left_setting = self.yaml.load_setting_yaml(config="setting")
        self.hands = {}
        self._init_hands()
        if self.hands:
            self._set_default_speeds()

    def _init_hands(self):
        hand_type = "left"
        hand_config = self.left_setting['LINKER_HAND']['LEFT_HAND']
        if hand_config.get('EXISTS', False):
            try:
                api = LinkerHandApi(
                    hand_type=hand_type,
                    hand_joint=hand_config['JOINT'],
                    can=hand_config.get('CAN_CHANNEL', 'PCAN_USBBUS1')
                )
                self.hands[hand_type] = {"api": api}
                ColorMsg(msg="左手 Linker API 初始化成功", color="green")
            except Exception as e:
                ColorMsg(msg=f"初始化失败: {e}", color="red")

    def _set_default_speeds(self):
        for hand_info in self.hands.values():
            hand_info["api"].set_speed([60, 220, 220, 220, 220])

    def control_hand(self, positions):
        if "left" in self.hands:
            self.hands["left"]["api"].finger_move(pose=positions)

    def close(self):
        for hand_info in self.hands.values():
            if hasattr(hand_info["api"].hand, 'bus'):
                hand_info["api"].hand.bus.shutdown()

def main():
    port = "COM9"
    baudrate = 115200

    stop_event = mp.Event()
    frame_queue = mp.Queue(maxsize=3)

    collector = mp.Process(
        target=collector_process,
        args=(frame_queue, stop_event, port, baudrate),
        name="sensor_collector"
    )
    controller = mp.Process(
        target=control_process,
        args=(frame_queue, stop_event),
        name="hand_controller"
    )

    collector.start()
    controller.start()

    print("启动完成：串口采集进程 + 机器手控制进程")
    print("按 Ctrl+C 停止")

    try:
        while collector.is_alive() and controller.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n主进程收到 Ctrl+C，准备停止...")
    finally:
        stop_event.set()

        collector.join(timeout=3)
        controller.join(timeout=3)

        if collector.is_alive():
            collector.terminate()
        if controller.is_alive():
            controller.terminate()

        print("全部进程已退出")

if __name__ == "__main__":
    mp.freeze_support()
    main()
