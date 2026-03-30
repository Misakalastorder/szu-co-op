"""
实时读取 STM32 串口传感器数据，直接驱动仿真手（低延迟）
参考：main_json_on_simulate.py + new_low.py
"""

from collections import defaultdict
import re
import signal
import sys
import time

import gym
import numpy as np
import pybullet as p
import serial
import yumi_gym

from config.variables_define import angle_limit_rob


stop_running = False

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

CALIBRATION_CONFIG = {
	"thumb": [
		(0, 2330, 0),
		(2540, 2730, 30),
		(2730, 2870, 60),
		(2870, 4095, 90),
	],
	"index": [
		(0, 1985, 0),
		(2000, 2160, 30),
		(2160, 2320, 60),
		(2320, 4095, 90),
	],
	"middle": [
		(0, 2020, 0),
		(2020, 2090, 30),
		(2090, 2180, 60),
		(2280, 4095, 90),
	],
	"ring": [
		(0, 1940, 0),
		(1990, 2110, 30),
		(2110, 2210, 60),
		(2210, 4095, 90),
	],
	"pinky": [
		(0, 2400, 0),
		(2500, 2600, 30),
		(2600, 2680, 60),
		(2680, 4095, 90),
	],
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


def calibrate_sensor(finger_name, ad_value):
	if finger_name not in CALIBRATION_CONFIG:
		return None

	calibration_ranges = CALIBRATION_CONFIG[finger_name]
	for min_ad, max_ad, angle in calibration_ranges:
		if min_ad <= ad_value < max_ad:
			return float(angle)
	return float(calibration_ranges[-1][2])


def trans2realworld(angle):
	for i in range(len(angle_limit_rob)):
		if i >= len(angle):
			break
		if angle[i] < angle_limit_rob[i][0]:
			angle[i] = angle_limit_rob[i][0]
		elif angle[i] > angle_limit_rob[i][1]:
			angle[i] = angle_limit_rob[i][1]
	return angle


def signal_handler(sig, frame):
	del sig, frame
	global stop_running
	stop_running = True
	print("\n检测到停止信号，准备退出...")


def handle_keyboard(camera_distance, camera_yaw, paused):
	global stop_running
	keys = p.getKeyboardEvents()
	for k, v in keys.items():
		if v & p.KEY_WAS_TRIGGERED:
			if k == ord("w"):
				camera_distance -= 0.05
			elif k == ord("s"):
				camera_distance += 0.05
			elif k == ord("a"):
				camera_yaw -= 10
			elif k == ord("d"):
				camera_yaw += 10
			elif k == ord(" "):
				paused = not paused
				print("暂停" if paused else "继续")
			elif k == 27:
				stop_running = True
	return camera_distance, camera_yaw, paused


def main():
	global stop_running
	signal.signal(signal.SIGINT, signal_handler)

	port = "COM9"
	baudrate = 115200

	num_joints = 23
	finger_to_joint_indices = {
		"thumb": [13, 14, 15, 16, 17],
		"index": [1, 0, 2, 3, 3],
		"middle": [4, 0, 5, 6, 6],
		"ring": [7, 0, 8, 9, 9],
		"pinky": [10, 0, 11, 12, 12],
	}

	env = gym.make("yumi-v0")
	env.reset()

	camera_distance = 0.5
	camera_yaw = 90
	camera_pitch = -20
	camera_target_position = [0, 0, 0.05]

	paused = False
	v_rate = 1.0

	current_ad_values = None
	current_data_group = defaultdict(dict)
	frame_count = 0
	ser = None

	try:
		ser = serial.Serial(
			port=port,
			baudrate=baudrate,
			bytesize=8,
			stopbits=1,
			parity=serial.PARITY_NONE,
			timeout=0.02,
		)
		print(f"串口已打开: {port}, 波特率: {baudrate}")
		print("开始实时仿真：W/S/A/D控制相机, Space暂停, ESC退出")

		while not stop_running:
			serial_line = ser.readline().decode("utf-8", errors="ignore").strip()

			if not serial_line:
				camera_distance, camera_yaw, paused = handle_keyboard(
					camera_distance, camera_yaw, paused
				)
				time.sleep(0.005)
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
				current_data_group.clear()
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
				"yaw": round(float(mpu_match.group(4)), 2),
			}

			if len(current_data_group) != 5:
				continue

			camera_distance, camera_yaw, paused = handle_keyboard(
				camera_distance, camera_yaw, paused
			)
			if stop_running:
				break

			while paused and not stop_running:
				camera_distance, camera_yaw, paused = handle_keyboard(
					camera_distance, camera_yaw, paused
				)
				time.sleep(0.05)

			if stop_running:
				break

			current_angles = np.zeros(num_joints, dtype=np.float32)
			for finger, indices in finger_to_joint_indices.items():
				data = current_data_group.get(finger)
				if not data:
					continue
				current_angles[indices[0]] = np.deg2rad(data.get("yaw", 0))
				current_angles[indices[1]] = np.deg2rad(data.get("roll", 0))
				current_angles[indices[2]] = np.deg2rad(data.get("pitch", 0))
				current_angles[indices[3]] = np.deg2rad(data.get("angle", 0))
				current_angles[indices[4]] = np.deg2rad(data.get("angle", 0))

			current_angles = trans2realworld(current_angles)
			action = current_angles.tolist()

			p.resetDebugVisualizerCamera(
				cameraDistance=camera_distance,
				cameraYaw=camera_yaw,
				cameraPitch=camera_pitch,
				cameraTargetPosition=camera_target_position,
			)

			env.step(action)
			frame_count += 1
			if frame_count % 50 == 0:
				print(f"已仿真 {frame_count} 帧")

			time.sleep(0.01 * v_rate)

			current_data_group.clear()
			current_ad_values = None

	except Exception as e:
		print(f"程序异常: {e}")
		import traceback

		traceback.print_exc()
	finally:
		if ser is not None and ser.is_open:
			ser.close()
			print("串口已关闭")

		try:
			env.close()
		except Exception:
			pass

		print("程序已退出")
		sys.exit(0)


if __name__ == "__main__":
	main()
