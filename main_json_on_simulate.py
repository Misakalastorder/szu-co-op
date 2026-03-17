'''
调用 JSON 传感器数据驱动虚拟环境中的人手模型进行仿真
参考自 main_h5_simulate.py
'''
import gym
import yumi_gym
import pybullet as p
import numpy as np
import json
import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple

# 项目导入
from angle2real import create_hand_kinematics
from config.variables_define import hand_cfg, urdf_file, angle_limit_rob 

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

def load_frames(json_path: Path) -> List[Tuple[str, Dict[str, Dict[str, float]]]]:
    """加载 JSON 数据帧"""
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

def trans2realworld(angle):
    '''
    要将虚拟角度转换为真实角度,主要是检查是否超限,输入为弧度,下限,上限
    '''
    # 遍历上下限，如果超出则进行裁剪
    for i in range(len(angle_limit_rob)):
        if angle[i] < angle_limit_rob[i][0]:
            angle[i] = angle_limit_rob[i][0]
        elif angle[i] > angle_limit_rob[i][1]:
            angle[i] = angle_limit_rob[i][1]
    return angle
def main():
    # 1. 初始化运动学模型
    hand_fk = create_hand_kinematics(urdf_file, hand_cfg)
    num_joints = 23  #linkerhand 的 out_num_joint 

    # 2. 定义映射 (JSON Finger -> 关节索引)
    # 基于 humanhand.urdf 的可动关节顺序
    finger_to_joint_indices = {
        "thumb":  [13, 14, 15, 16, 17],
        "index":  [1,   0, 2, 3, 3],
        "middle": [4,   0, 5, 6, 6],
        "ring":   [7,   0, 8, 9, 9],
        "pinky":  [10,  0, 11, 12, 12],
    }

    # 3. 加载 JSON 数据
    json_path = Path(__file__).parent / "finger_sensor_data_20260312_200006.json"
    if not json_path.exists():
        print(f"找不到数据文件: {json_path}")
        return
    frames_data = load_frames(json_path)
    total_frames = len(frames_data)
    print(f"成功加载数据，共 {total_frames} 帧")

    # 4. 初始化虚拟环境
    env = gym.make('yumi-v0')
    observation = env.reset()

    # 相机初始参数
    camera_distance = 0.5
    camera_yaw = 90
    camera_pitch = -20
    camera_target_position = [0, 0, 0.05]
    
    paused = False
    v_rate = 1
    stop = False

    print("开始仿真控制：'W/S/A/D' 控制相机, 'Space' 暂停, 'ESC' 退出")

    while not stop:
        for t in range(total_frames):
            ts, finger2vals = frames_data[t]
            
            # 构造 action (弧度)
            current_angles = np.zeros(num_joints, dtype=np.float32)
            
            for finger_name, indices in finger_to_joint_indices.items():
                data = finger2vals.get(finger_name)
                if data:
                    # if finger_name == "thumb":
                    #     # 拇指: index0:yaw, index1:pitch, index2:angle
                    #     current_angles[indices[0]] = np.deg2rad(data.get("yaw", 0))
                    #     current_angles[indices[1]] = np.deg2rad(data.get("pitch", 0))
                    #     current_angles[indices[2]] = np.deg2rad(data.get("angle", 0))
                    # else:
                    #     # 其他手指: index0:yaw, index1:pitch, index2&3:angle (联动)
                    current_angles[indices[0]] = np.deg2rad(data.get("yaw", 0))
                    current_angles[indices[1]] = np.deg2rad(data.get("roll", 0))
                    current_angles[indices[2]] = np.deg2rad(data.get("pitch", 0))
                    current_angles[indices[3]] = np.deg2rad(data.get("angle", 0))
                    current_angles[indices[4]] = np.deg2rad(data.get("angle", 0))
                    # current_angles[indices[4]] = np.deg2rad(data.get("angle", 0))
            # 检查是否超上下限
            current_angles = trans2realworld(current_angles)

            action = current_angles.tolist()

            # PyBullet 交互处理
            keys = p.getKeyboardEvents()
            for k, v in keys.items():
                if v & p.KEY_WAS_TRIGGERED:
                    if k == ord('w'): camera_distance -= 0.05
                    elif k == ord('s'): camera_distance += 0.05
                    elif k == ord('a'): camera_yaw -= 10
                    elif k == ord('d'): camera_yaw += 10
                    elif k == ord(' '):
                        paused = not paused
                        print('暂停' if paused else '继续')
                    elif k == 27: # ESC
                        stop = True

            if stop: break

            if paused:
                time.sleep(0.1)
                # 保持在当前帧循环等待
                # 注意：这里逻辑上需要让 t 保持不变，故简单的实现是重复执行 sleep
                while paused and not stop:
                    keys = p.getKeyboardEvents()
                    if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                        paused = False
                        print('继续')
                    if 27 in keys and keys[27] & p.KEY_WAS_TRIGGERED:
                        stop = True
                    time.sleep(0.1)

            # 更新相机并执行仿真步
            p.resetDebugVisualizerCamera(
                cameraDistance=camera_distance,
                cameraYaw=camera_yaw,
                cameraPitch=camera_pitch,
                cameraTargetPosition=camera_target_position
            )
            
            observation, reward, done, info = env.step(action)
            time.sleep(0.02 * v_rate)

            if t % 50 == 0:
                print(f"当前时间戳: {ts} [{t}/{total_frames}]")

        if stop: break
        print("数据播放完成，重新播放...")

if __name__ == "__main__":
    main()
