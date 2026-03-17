"""
查看 human_hand 的可活动空间（关节逐个扫动）

从首个关节开始遍历：下限 -> 上限 -> 下限
以动画形式展示（不显示轨迹，只显示当前姿态）。
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from angle2real import create_hand_kinematics
from config.variables_define import angle_limit_rob, hand_cfg, out_num_joint, urdf_file
from config.hand_visualization import infer_edges_from_parents, sweep_angles


def animate_single_joint_sweep(
    hand_fk,
    base_angles: torch.Tensor,
    edges,
    joint_index: int,
    joint_name: str,
    lower: float,
    upper: float,
    steps: int = 60,
    interval_ms: int = 30,
):
    """
    单关节扫动动画：下限 -> 上限 -> 下限
    只展示当前骨架姿态，不绘制轨迹。
    """
    seq = sweep_angles(lower, upper, steps)

    # 预计算每一帧的关节位置（避免动画回调里频繁做CPU/GPU切换）
    num_fk_joints = len(hand_fk.joint_names)
    positions_all = []
    with torch.no_grad():
        for a in seq:
            angles = base_angles.clone()
            angles[0, joint_index] = float(a)
            _, _, global_positions = hand_fk.forward(angles)
            positions_all.append(global_positions[0].detach().cpu().numpy())
    positions_all = np.asarray(positions_all)  # [T, J, 3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Animate joint: {joint_name} ({joint_index})  [{lower:.3f}, {upper:.3f}] rad")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim3d(-0.15, 0.15)
    ax.set_ylim3d(-0.15, 0.15)
    ax.set_zlim3d(-0.05, 0.35)

    # 为每条边创建一条Line对象，后续仅更新数据（不留轨迹）
    lines = []
    for _ in edges:
        (ln,) = ax.plot([0, 0], [0, 0], [0, 0], color="royalblue", linewidth=2.0, alpha=0.9)
        lines.append(ln)

    # 关节点散点
    scat = ax.scatter([], [], [], s=12, c="black", alpha=0.8)

    def init():
        scat._offsets3d = ([], [], [])
        for ln in lines:
            ln.set_data([0, 0], [0, 0])
            ln.set_3d_properties([0, 0])
        return [scat, *lines]

    def update(frame_idx: int):
        pts = positions_all[frame_idx]  # [J, 3]

        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        scat._offsets3d = (xs, ys, zs)

        for (parent_idx, child_idx), ln in zip(edges, lines):
            ln.set_data([pts[parent_idx, 0], pts[child_idx, 0]], [pts[parent_idx, 1], pts[child_idx, 1]])
            ln.set_3d_properties([pts[parent_idx, 2], pts[child_idx, 2]])

        return [scat, *lines]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(positions_all),
        init_func=init,
        interval=interval_ms,
        blit=False,  # 3D下blit不稳定
        repeat=True,
    )

    plt.tight_layout()
    plt.show()
    return ani


def main():
    hand_fk = create_hand_kinematics(urdf_file, hand_cfg)
    edges = infer_edges_from_parents(hand_fk.joint_parents)

    # FK 需要完整 joints_name 长度的角度向量；超出 out_num_joint 的（指尖fixed joint）保持为0
    num_fk_joints = len(hand_fk.joint_names)
    print(num_fk_joints)
    base_angles = torch.zeros((1, num_fk_joints), dtype=torch.float32)

    # 为 out_num_joint 内的关节逐个扫动
    for j in range(out_num_joint):
        joint_name = hand_fk.joint_names[j]
        lower, upper = angle_limit_rob[j]

        animate_single_joint_sweep(
            hand_fk=hand_fk,
            base_angles=base_angles,
            edges=edges,
            joint_index=j,
            joint_name=joint_name,
            lower=float(lower),
            upper=float(upper),
            steps=60,
            interval_ms=30,
        )


if __name__ == "__main__":
    main()

