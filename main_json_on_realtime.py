'''
调用 JSON 传感器数据驱动真实 LinkerHand 机器手
参考自 main_h5_realtime.py, main_json_on_simulate.py
'''
import numpy as np
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import can
import keyboard

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

# JSON 手指到 Linker 18关节索引的映射
finger_to_joint_indices = {
    "index":  [1, 2, 3],  
    "middle": [4, 5, 6],
    "ring":   [7, 8, 9],
    "pinky":  [10, 11, 12],
    "thumb":  [13, 14, 15, 16, 17],
}

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

def load_frames(json_path: Path) -> List[Tuple[str, Dict[str, Dict[str, float]]]]:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    frames = []
    for ts, records in raw.items():
        if not isinstance(records, list): continue
        finger2vals = {}
        for rec in records:
            name = rec.get("finger_name")
            if name in FINGERS:
                vals = {"angle": float(rec.get("angle", rec.get("angel", 0)))}
                for k in ("pitch", "roll", "yaw"):
                    if k in rec: vals[k] = float(rec[k])
                finger2vals[name] = vals
        frames.append((str(ts), finger2vals))
    frames.sort(key=lambda x: x[0])
    return frames

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
    json_path = Path(__file__).parent / "finger_sensor_data_20260312_200006.json"
    frames_data = load_frames(json_path)
    controller = HandController()
    
    print("开始 JSON 实时控制... 按 'Esc' 退出")
    
    try:
        while True:
            for ts, finger2vals in frames_data:
                angle_rad_18 = np.zeros(18, dtype=np.float32)
                for f, indices in finger_to_joint_indices.items():
                    data = finger2vals.get(f)
                    if data:
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

                real_pos = trans2realworld_linker(angle_rad_18)
                controller.control_hand(real_pos)
                
                if keyboard.is_pressed('esc'): return
                time.sleep(0.04)
            print("循环播放结束，重新开始")
    except KeyboardInterrupt: pass
    finally: controller.close()

if __name__ == "__main__":
    main()
