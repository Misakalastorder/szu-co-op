"""
读取 finger_sensor_data_sample.json，并逐帧按手指输出角度信息（angle/pitch/roll/yaw）。

用途：快速测试 JSON -> 角度序列 的格式转换是否正确。
注意：此脚本忽略加速度(ax/ay/az)等其它字段，也忽略 back_hand 条目。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def load_frames(json_path: Path) -> List[Tuple[str, Dict[str, Dict[str, float]]]]:
    """
    返回：[(timestamp_str, {finger: {"angle":..,"pitch":..,"roll":..,"yaw":..}, ...}), ...]
    """
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("JSON 顶层必须是 dict: {timestamp: [records...]}")

    frames: List[Tuple[str, Dict[str, Dict[str, float]]]] = []
    for ts, records in raw.items():
        if not isinstance(records, list):
            continue

        finger2vals: Dict[str, Dict[str, float]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            name = rec.get("finger_name")
            if name in FINGERS:
                vals: Dict[str, float] = {}

                # 文档里写 angel，但不同数据可能是 angle/angel 二选一
                if "angle" in rec:
                    vals["angle"] = float(rec["angle"])
                elif "angel" in rec:
                    vals["angle"] = float(rec["angel"])

                # 姿态角（单位一般为度；这里只做原样提取用于格式核对）
                for k in ("pitch", "roll", "yaw"):
                    if k in rec:
                        vals[k] = float(rec[k])

                if vals:
                    finger2vals[name] = vals

        frames.append((str(ts), finger2vals))

    # 时间戳形如 HH:MM:SS.mmm，直接字典序排序即可
    frames.sort(key=lambda x: x[0])
    return frames


def main():
    json_path = Path(__file__).with_name("finger_sensor_data_sample.json")
    frames = load_frames(json_path)

    print(f"Loaded frames: {len(frames)} from {json_path}")
    print("Output format: timestamp | finger: angle pitch roll yaw")

    for ts, finger2vals in frames:
        print(ts)
        ok_cnt = 0
        for f in FINGERS:
            v = finger2vals.get(f)
            if not v:
                print(f"  {f}: None")
                continue
            ok_cnt += 1
            a = v.get("angle", None)
            p = v.get("pitch", None)
            r = v.get("roll", None)
            y = v.get("yaw", None)

            def fmt(x):
                return "None" if x is None else f"{x:.2f}"

            print(f"  {f}: angle={fmt(a)} pitch={fmt(p)} roll={fmt(r)} yaw={fmt(y)}")
        print(f"  (ok={ok_cnt}/5)")


if __name__ == "__main__":
    main()

