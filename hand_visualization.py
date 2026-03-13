"""
手部运动学可视化工具

从 angle2real.py 中抽离，便于在不同脚本中直接调用。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


def infer_edges_from_parents(joint_parents: Sequence[int]) -> List[List[int]]:
    edges: List[List[int]] = []
    for child_idx, parent_idx in enumerate(joint_parents):
        if parent_idx != -1:
            edges.append([parent_idx, child_idx])
    return edges


def visualize_hand_kinematics(hand_fk, joint_angles: torch.Tensor, edges: Optional[Sequence[Sequence[int]]] = None):
    """
    可视化单帧手部运动学结果（骨架连线 + 关节编号）
    """
    _, _, global_positions = hand_fk.forward(joint_angles)
    joint_positions = global_positions[0].detach().cpu().numpy()

    if edges is None:
        edges = infer_edges_from_parents(hand_fk.joint_parents)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 范围给一个较通用的默认值
    ax.set_xlim3d(-0.15, 0.15)
    ax.set_ylim3d(-0.15, 0.15)
    ax.set_zlim3d(-0.05, 0.35)

    for parent_idx, child_idx in edges:
        line_x = [joint_positions[parent_idx][0], joint_positions[child_idx][0]]
        line_y = [joint_positions[parent_idx][1], joint_positions[child_idx][1]]
        line_z = [joint_positions[parent_idx][2], joint_positions[child_idx][2]]
        ax.plot(line_x, line_y, line_z, "royalblue", marker="o", linewidth=1.5, markersize=3)

    for i, (x, y, z) in enumerate(joint_positions):
        ax.text(float(x), float(y), float(z), f"{i}", fontsize=8, color="black")

    plt.tight_layout()
    plt.show()


def sweep_angles(lower: float, upper: float, steps: int) -> np.ndarray:
    """
    生成 下限->上限->下限 的扫动序列
    """
    if steps < 2:
        return np.array([lower, upper, lower], dtype=np.float32)
    up = np.linspace(lower, upper, steps, dtype=np.float32)
    down = np.linspace(upper, lower, steps, dtype=np.float32)[1:]  # 避免重复upper
    return np.concatenate([up, down], axis=0)


def visualize_joint_activity_region(
    hand_fk,
    base_angles: torch.Tensor,
    edges: Sequence[Sequence[int]],
    joint_index: int,
    joint_name: str,
    angle_lower: float,
    angle_upper: float,
    steps: int,
    end_effector_indices: Sequence[int],
    end_effector_names: Sequence[str],
):
    """
    固定除一个关节外的所有关节，将该关节从下限扫到上限再回到下限，
    绘制各指尖(end-effectors)的轨迹，作为“可活动区域”的直观展示。
    """
    seq = sweep_angles(angle_lower, angle_upper, steps)
    tip_paths = {name: [] for name in end_effector_names}

    with torch.no_grad():
        for a in seq:
            angles = base_angles.clone()
            angles[0, joint_index] = float(a)
            _, _, global_positions = hand_fk.forward(angles)
            pts = global_positions[0].detach().cpu().numpy()
            for name, idx in zip(end_effector_names, end_effector_indices):
                tip_paths[name].append(pts[idx].copy())

        # 画一帧基准骨架（base_angles）
        _, _, global_positions0 = hand_fk.forward(base_angles)
        skel = global_positions0[0].detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Joint sweep: {joint_name} ({joint_index})  [{angle_lower:.3f}, {angle_upper:.3f}] rad")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 骨架
    for parent_idx, child_idx in edges:
        ax.plot(
            [skel[parent_idx][0], skel[child_idx][0]],
            [skel[parent_idx][1], skel[child_idx][1]],
            [skel[parent_idx][2], skel[child_idx][2]],
            color="black",
            linewidth=1.0,
            alpha=0.5,
        )

    # 指尖轨迹
    colors = {
        "thumb": "crimson",
        "index": "dodgerblue",
        "middle": "seagreen",
        "ring": "darkorange",
        "pinky": "purple",
    }

    for name, path in tip_paths.items():
        arr = np.asarray(path)
        key = next((k for k in colors.keys() if k in name), None)
        c = colors.get(key, "royalblue")
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=c, linewidth=2.0, label=name)

    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim3d(-0.15, 0.15)
    ax.set_ylim3d(-0.15, 0.15)
    ax.set_zlim3d(-0.05, 0.35)
    plt.tight_layout()
    plt.show()

