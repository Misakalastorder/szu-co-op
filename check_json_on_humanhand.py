import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib.animation import FuncAnimation

# Re-using your project imports
from angle2real import create_hand_kinematics
from config.variables_define import hand_cfg, urdf_file
from config.hand_visualization import infer_edges_from_parents

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

def load_frames(json_path: Path) -> List[Tuple[str, Dict[str, Dict[str, float]]]]:
    """Existing loading logic from your script."""
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

def main():
    # 1. Setup Kinematics
    hand_fk = create_hand_kinematics(urdf_file, hand_cfg)
    edges = infer_edges_from_parents(hand_fk.joint_parents)
    num_fk_joints = len(hand_fk.joint_names) # 25
    
    # 2. Define Mapping (JSON Finger -> Hand Model Indices)
    # TODO: Adjust these indices based on your specific URDF joint order
    # Typically: 4 joints per finger + 5 fixed tips = 25
    finger_to_joint_indices = {
        "thumb":  [1, 2, 3, 3], #大拇指少一个关节，最后一个重复使用
        "index":  [4, 5, 6, 7],
        "middle": [8, 9, 10, 11],
        "ring":   [12, 13, 14, 15],
        "pinky":  [16, 17, 18, 19],
    }
    
    # 3. Load Data 数据加载点
    # json_path = Path(__file__).with_name("finger_sensor_data_sample.json")
    json_path = Path(__file__).with_name("finger_sensor_data_20260312_200006.json")
    frames_data = load_frames(json_path)
    
    # 4. Pre-calculate positions for animation
    positions_all = []
    for ts, finger2vals in frames_data:
        # Initialize all 25 joints to 0.0 (handles the 5 fixed tips automatically)
        current_angles = torch.zeros((1, num_fk_joints), dtype=torch.float32)
        
        for finger_name, indices in finger_to_joint_indices.items():
            data = finger2vals.get(finger_name)
            if data:
                # Example: mapping 'angle' to the first joint of the finger
                # Adjust which data (pitch/roll/yaw) maps to which index here
                
                current_angles[0, indices[0]] = np.deg2rad(data["yaw"])
                # print(f"{finger_name}: {np.rad2deg(np.deg2rad(data['yaw'])):.2f}")
                current_angles[0, indices[1]] = np.deg2rad(data["pitch"])
                current_angles[0, indices[2]] = np.deg2rad(data["angle"])
                current_angles[0, indices[3]] = np.deg2rad(data["angle"])
               
        print(f"{ts}:")
        with torch.no_grad():
            _, _, global_positions = hand_fk.forward(current_angles)
            positions_all.append(global_positions[0].cpu().numpy())
            
    positions_all = np.asarray(positions_all)

    # 5. Visualization Setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim3d(-0.15, 0.15); ax.set_ylim3d(-0.15, 0.15); ax.set_zlim3d(-0.05, 0.35)
    
    lines = [ax.plot([0,0],[0,0],[0,0], color="royalblue", lw=2)[0] for _ in edges]
    scat = ax.scatter([], [], [], s=12, c="black")

    def update(frame_idx):
        pts = positions_all[frame_idx]
        scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        for (p_idx, c_idx), ln in zip(edges, lines):
            ln.set_data([pts[p_idx, 0], pts[c_idx, 0]], [pts[p_idx, 1], pts[c_idx, 1]])
            ln.set_3d_properties([pts[p_idx, 2], pts[c_idx, 2]])
        return [scat, *lines]
    ani = FuncAnimation(fig, update, frames=len(positions_all), interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    main()