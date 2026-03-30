"""
实时读取串口传感器数据并驱动真实 LinkerHand 机器手（共享变量 + 多进程）

说明：
1. 采集进程持续从串口读取数据，组包后写入共享 current_data_group。
2. 控制进程持续读取共享 current_data_group，并实时下发到机器手。
"""

import multiprocessing as mp
import re
import time
from typing import Dict, Optional

import numpy as np
import serial

from LinkerHand.linker_hand_api import LinkerHandApi
from LinkerHand.utils.color_msg import ColorMsg
from LinkerHand.utils.load_write_yaml import LoadWriteYaml

# 18个关节角度限制（弧度）
angle_limit_rob = [
	[0.0, 0.0],
	[-0.18, 0.18],
	[0.0, 1.57],
	[0.0, 1.57],
	[-0.18, 0.18],
	[0.0, 1.57],
	[0.0, 1.57],
	[-0.18, 0.18],
	[0.0, 1.57],
	[0.0, 1.57],
	[-0.18, 0.18],
	[0.0, 1.57],
	[0.0, 1.57],
	[-0.6, 0.6],
	[0.0, 1.6],
	[0.0, 1.0],
	[0.0, 1.57],
	[0.0, 1.57],
]

# 25个驱动器到18关节映射
joint_map = {
	0: 15,
	1: 2,
	2: 5,
	3: 8,
	4: 11,
	5: 14,
	6: 1,
	7: 4,
	8: 7,
	9: 10,
	10: 13,
	11: 0,
	12: 0,
	13: 0,
	14: 0,
	15: 16,
	16: 0,
	17: 0,
	18: 0,
	19: 0,
	20: 17,
	21: 3,
	22: 6,
	23: 9,
	24: 12,
}

# 传感器标定配置
CALIBRATION_CONFIG = {
	"thumb": [(0, 2330, 0), (2540, 2730, 30), (2730, 2870, 60), (2870, 4095, 90)],
	"index": [(0, 1985, 0), (2000, 2160, 30), (2160, 2320, 60), (2320, 4095, 90)],
	"middle": [(0, 2020, 0), (2020, 2090, 30), (2090, 2180, 60), (2280, 4095, 90)],
	"ring": [(0, 1940, 0), (1990, 2110, 30), (2110, 2210, 60), (2210, 4095, 90)],
	"pinky": [(0, 2400, 0), (2500, 2600, 30), (2600, 2680, 60), (2680, 4095, 90)],
}

ID_TO_FINGER = {
	1: "thumb",
	2: "index",
	3: "middle",
	4: "ring",
	5: "back_hand",
	6: "pinky",
}

AD_INDEX_FOR_FINGER = {
	"thumb": 0,
	"index": 1,
	"middle": 2,
	"ring": 3,
	"pinky": 4,
}

finger_to_joint_indices = {
	"index": [1, 2, 3],
	"middle": [4, 5, 6],
	"ring": [7, 8, 9],
	"pinky": [10, 11, 12],
	"thumb": [13, 14, 15, 16, 17],
}

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

AD_PATTERN = re.compile(r"AD:\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)")
MPU_PATTERN = re.compile(
	r"M(\d+):\s*"
	r"P=([-+]?\d+\.\d+)\s+"
	r"R=([-+]?\d+\.\d+)\s+"
	r"Y=([-+]?\d+\.\d+)\s+"
	r"aX=([-+]?\d+\.\d+)\s+"
	r"aY=([-+]?\d+\.\d+)\s+"
	r"aZ=([-+]?\d+\.\d+)"
)


def calibrate_sensor(finger_name: str, ad_value: int) -> Optional[float]:
	if finger_name not in CALIBRATION_CONFIG:
		return None

	calibration_ranges = CALIBRATION_CONFIG[finger_name]
	for min_ad, max_ad, angle in calibration_ranges:
		if min_ad <= ad_value < max_ad:
			return float(angle)

	return float(calibration_ranges[-1][2])


def unit(num: float) -> int:
	return 0 if num < 0 else 255 if num > 255 else int(num)

# 虚拟机器手关节角到实体机器手驱动器位置的映射函数
def trans2realworld_linker(angle_rad_18: np.ndarray):
	angle_norm = [0] * 18
	for i in range(18):
		low, high = angle_limit_rob[i]
		norm = (angle_rad_18[i] - low) / (high - low) if high > low else 0.0
		angle_norm[i] = unit(norm * 255)

	angle_mapped = [0] * 25
	for drive_idx, joint_idx in joint_map.items():
		angle_mapped[drive_idx] = 255 - angle_norm[joint_idx]
	return angle_mapped


# 构建 18 个关节角度，可用于虚拟机器手控制
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
		hand_config = self.left_setting["LINKER_HAND"]["LEFT_HAND"]
		if hand_config.get("EXISTS", False):
			try:
				api = LinkerHandApi(
					hand_type=hand_type,
					hand_joint=hand_config["JOINT"],
					can=hand_config.get("CAN_CHANNEL", "PCAN_USBBUS1"),
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
			if hasattr(hand_info["api"].hand, "bus"):
				hand_info["api"].hand.bus.shutdown()


def collector_process(current_data_group, state_lock, stop_event, port: str, baudrate: int):
	ser = None
	current_ad_values = None
	local_data_group = {}
	sample_id = 0
	last_log_time = time.time()

	try:
		ser = serial.Serial(
			port=port,
			baudrate=baudrate,
			bytesize=8,
			stopbits=1,
			parity=serial.PARITY_NONE,
			timeout=0.1,
		)
		print(f"[采集进程] 串口已打开: {port}, 波特率: {baudrate}")

		while not stop_event.is_set():
			serial_line = ser.readline().decode("utf-8", errors="ignore").strip()
			if not serial_line:
				continue

			ad_match = AD_PATTERN.search(serial_line)
			if ad_match:
				current_ad_values = [
					int(ad_match.group(1)),
					int(ad_match.group(2)),
					int(ad_match.group(3)),
					int(ad_match.group(4)),
					int(ad_match.group(5)),
				]
				local_data_group = {}
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
			local_data_group[finger_name] = {
				"angle": calibrate_sensor(finger_name, ad_value),
				"pitch": round(float(mpu_match.group(2)), 2),
				"roll": round(float(mpu_match.group(3)), 2),
				"yaw": round(float(mpu_match.group(4)), 2),
			}

			if len(local_data_group) == 5:
				sample_id += 1
				with state_lock:
					current_data_group["sample_id"] = sample_id
					current_data_group["timestamp"] = time.time()
					current_data_group["fingers"] = dict(local_data_group)

				local_data_group = {}
				now = time.time()
				if now - last_log_time >= 1.0:
					print(f"[采集进程] 已更新共享数据 {sample_id} 帧")
					last_log_time = now

	except Exception as e:
		print(f"[采集进程] 异常: {e}")
		stop_event.set()
	finally:
		if ser is not None and ser.is_open:
			ser.close()
			print("[采集进程] 串口已关闭")


def control_process(current_data_group, state_lock, stop_event):
	controller = None
	handled_sample_id = -1
	control_count = 0
	last_log_time = time.time()

	try:
		controller = HandController()
		print("[控制进程] LinkerHand 控制初始化完成")
        # 控制开始循环
		while not stop_event.is_set():
			with state_lock:
				sample_id = int(current_data_group.get("sample_id", -1))
				finger2vals = dict(current_data_group.get("fingers", {}))

			if sample_id <= handled_sample_id or not finger2vals:
				time.sleep(0.01)
				continue

			handled_sample_id = sample_id
			angle_rad_18 = build_angle_rad_18(finger2vals)
			real_pos = trans2realworld_linker(angle_rad_18)
			controller.control_hand(real_pos)

			control_count += 1
			now = time.time()
			if now - last_log_time >= 1.0:
				print(f"[控制进程] 已控制 {control_count} 帧")
				last_log_time = now

	except Exception as e:
		print(f"[控制进程] 异常: {e}")
		stop_event.set()
	finally:
		if controller is not None:
			controller.close()
		print("[控制进程] 已安全退出")


def main():
	port = "COM9"
	baudrate = 115200

	manager = mp.Manager()
	current_data_group = manager.dict()
	current_data_group["sample_id"] = -1
	current_data_group["timestamp"] = 0.0
	current_data_group["fingers"] = {}

	state_lock = mp.Lock()
	stop_event = mp.Event()

	collector = mp.Process(
		target=collector_process,
		args=(current_data_group, state_lock, stop_event, port, baudrate),
		name="sensor_collector",
	)
	controller = mp.Process(
		target=control_process,
		args=(current_data_group, state_lock, stop_event),
		name="hand_controller",
	)

	collector.start()
	controller.start()

	print("启动完成：采集进程(写共享变量) + 控制进程(读共享变量)")
	print("按 Ctrl+C 停止")

	try:
		while collector.is_alive() and controller.is_alive():
			time.sleep(0.1)
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

