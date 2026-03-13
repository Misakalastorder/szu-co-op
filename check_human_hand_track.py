"""
查看 human_hand 的可活动空间（关节逐个扫动）

从首个关节开始遍历：下限 -> 上限 -> 下限
并可视化各指尖(end-effectors)的活动轨迹。
"""

import torch

from angle2real import create_hand_kinematics
from config.variables_define import angle_limit_rob, hand_cfg, out_num_joint, urdf_file
from hand_visualization import infer_edges_from_parents, visualize_joint_activity_region


def main():
    hand_fk = create_hand_kinematics(urdf_file, hand_cfg)
    edges = infer_edges_from_parents(hand_fk.joint_parents)

    # FK 需要完整 joints_name 长度的角度向量；超出 out_num_joint 的（指尖fixed joint）保持为0
    num_fk_joints = len(hand_fk.joint_names)
    base_angles = torch.zeros((1, num_fk_joints), dtype=torch.float32)

    # end-effectors 在 hand_cfg 里是关节名，这里转成索引
    name_to_index = {n: i for i, n in enumerate(hand_fk.joint_names)}
    end_effector_names = list(hand_cfg["end_effectors"])
    end_effector_indices = [name_to_index[n] for n in end_effector_names]

    # 为 out_num_joint 内的关节逐个扫动
    for j in range(out_num_joint):
        joint_name = hand_fk.joint_names[j]
        lower, upper = angle_limit_rob[j]

        visualize_joint_activity_region(
            hand_fk=hand_fk,
            base_angles=base_angles,
            edges=edges,
            joint_index=j,
            joint_name=joint_name,
            angle_lower=float(lower),
            angle_upper=float(upper),
            steps=60,
            end_effector_indices=end_effector_indices,
            end_effector_names=end_effector_names,
        )


if __name__ == "__main__":
    main()

