import serial
import re
import os
import json
import signal
import sys
from collections import defaultdict
from datetime import datetime
# 实时高
# 全局变量：标记是否停止采集
stop_collection = False

# ===================== 传感器标定配置（核心：方便你修改） =====================
# 格式：{手指名称: [(阈值下限, 阈值上限, 对应角度), ...]}
# 每个手指的AD值区间对应5度间隔的角度（0°,5°,10°,...,90°）
CALIBRATION_CONFIG = {
    'thumb': [
        (0, 1600, 0),    # AD值 0-1000 → 0度
        (1609, 1632, 5),  # AD值 1000-1200 → 5度
        (1632, 1640, 10), # AD值 1200-1400 → 10度
        (1640, 1691, 15), # AD值 1400-1600 → 15度
        (1691, 1725, 20), # AD值 1600-1660 → 20度
        (1725, 1760, 25), # AD值 1660-1720 → 25度
        (1760, 1800, 30), # AD值 1720-1780 → 30度
        (1800, 1841, 35), # AD值 1780-1880 → 35度
        (1841, 1910, 40), # AD值 1880-1980 → 40度
        (1910, 1923, 45), # AD值 1980-2060 → 45度
        (1923, 1965, 50), # AD值 2060-2200 → 50度
        (1965, 2000, 55), # AD值 2200-2350 → 55度
        (2000, 2044, 60), # AD值 2350-2500 → 60度
        (2044, 2082, 65), # AD值 2500-2600 → 65度
        (2082, 2100, 70), # AD值 2600-2700 → 70度
        (2100, 2149, 75), # AD值 2700-2900 → 75度
        (2149, 2177, 80), # AD值 2900-3100 → 80度
        (2177, 2210, 85), # AD值 3100-3500 → 85度
        (2210, 4095, 90)  # AD值 3500-4095 → 90度
    ],
    'index': [
        (0, 2600, 0),
        (2614, 2630, 5),
        (2630, 2638, 10),
        (2638, 2679, 15),
        (2679, 2699, 20),
        (2699, 2730, 25),
        (2730, 2738, 30),
        (2739, 2759, 35),
        (2759, 2800, 40),
        (2800, 2808, 45),
        (2808, 2837, 50),
        (2837, 2870, 55),
        (2870, 2907, 60),
        (2907, 2951, 65),
        (2951, 2980, 70),
        (2980, 3056, 75),
        (3056, 3119, 80),
        (3119, 3200, 85),
        (3200, 4095, 90)
    ],
    'middle': [
        (0, 2400, 0),
        (2409, 2425, 5),
        (2425, 2440, 10),
        (2440, 2467, 15),
        (2467, 2492, 20),
        (2492, 2510, 25),
        (2510, 2549, 30),
        (2549, 2581, 35),
        (2581, 2640, 40),
        (2640, 2650, 45),
        (2650, 2686, 50),
        (2686, 2710, 55),
        (2710, 2762, 60),
        (2762, 2801, 65),
        (2801, 2840, 70),
        (2840, 2880, 75),
        (2880, 2919, 80),
        (2919, 2960, 85),
        (2960, 4095, 90)
    ],
    'ring': [
        (0, 1600, 0),
        (1615, 1635, 5),
        (1635, 1640, 10),
        (1640, 1676, 15),
        (1676, 1698, 20),
        (1698, 1730, 25),
        (1730, 1747, 30),
        (1747, 1775, 35),
        (1775, 1820, 40),
        (1820, 1844, 45),
        (1844, 1885, 50),
        (1885, 1920, 55),
        (1920, 1984, 60),
        (1984, 2043, 65),
        (2043, 2110, 70),
        (2110, 2183, 75),
        (2183, 2266, 80),
        (2266, 2360, 85),
        (2360, 4095, 90)
    ],
    'pinky': [
        (0, 2000, 0),
        (2057, 2098, 5),
        (2098, 2160, 10),
        (2160, 2163, 15),
        (2163, 2188, 20),
        (2188, 2200, 25),
        (2200, 2229, 30),
        (2229, 2245, 35),
        (2245, 2246, 40),
        (2246, 2273, 45),
        (2273, 2286, 50),
        (2286, 2300, 55),
        (2300, 2312, 60),
        (2312, 2326, 65),
        (2326, 2360, 70),
        (2360, 2361, 75),
        (2361, 2382, 80),
        (2382, 2400, 85),
        (2400, 4095, 90)
    ]
}

# 数字ID到手指名称的映射表
ID_TO_FINGER = {
    1: 'thumb',      # 拇指
    2: 'index',      # 食指
    3: 'middle',     # 中指
    4: 'ring',       # 无名指
    5: 'back_hand',  # 手背
    6: 'pinky'       # 小指
}

# 自定义MPU排序顺序 - 核心新增配置
CUSTOM_MPU_ORDER = [1, 2, 3, 4, 6, 5]  # 拇指→食指→中指→无名指→小指→手背


def calibrate_sensor(finger_name, ad_value):
    """
    传感器标定函数：根据AD值和预设配置返回对应的角度值
    :param finger_name: 手指名称 (thumb/index/middle/ring/pinky)
    :param ad_value: 传感器的原始AD值
    :return: 标定后的角度值
    """
    # 检查手指名称是否有效
    if finger_name not in CALIBRATION_CONFIG:
        return None

    # 获取该手指的标定配置
    calibration_ranges = CALIBRATION_CONFIG[finger_name]

    # 遍历标定范围，找到匹配的角度
    for min_ad, max_ad, angle in calibration_ranges:
        if min_ad <= ad_value < max_ad:
            return angle

    # 如果没有匹配到，返回最后一个角度
    return calibration_ranges[-1][2]


def signal_handler(sig, frame):
    """捕获Ctrl+C的信号处理函数"""
    global stop_collection
    stop_collection = True
    print("\n🛑 检测到停止信号，正在保存数据并退出...")


def main():
    global stop_collection
    # 注册Ctrl+C信号处理函数
    signal.signal(signal.SIGINT, signal_handler)

    # ===================== 核心配置 =====================
    port = "COM9"  # 请修改为你的实际串口号
    baudrate = 115200  # 请和单片机串口波特率保持一致（之前优化建议改成115200）
    save_folder = r"D:\\2026\\code\\szu\\26.3.2-type"

    # ===================== 初始化工作 =====================
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file_path = os.path.join(save_folder, f"finger_sensor_data_{file_timestamp}.json")

    # 字典存储：时间戳为键，手指列表为值
    all_collected_data = {}
    current_data_group = defaultdict(dict)
    current_ad_values = None
    current_sample_time = None

    # ===================== 【已适配新格式】正则匹配规则 =====================
    # 适配新的AD行格式：AD:1222,1870,1986,1125,1686
    ad_pattern = re.compile(r'AD:\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)')
    # 适配新的MPU行格式：M1: P=0.15 R=0.65 Y=8.53 aX=0.01 aY=0.00 aZ=0.02
    mpu_pattern = re.compile(
        r'M(\d+):\s*'
        r'P=([-+]?\d+\.\d+)\s+'
        r'R=([-+]?\d+\.\d+)\s+'
        r'Y=([-+]?\d+\.\d+)\s+'
        r'aX=([-+]?\d+\.\d+)\s+'
        r'aY=([-+]?\d+\.\d+)\s+'
        r'aZ=([-+]?\d+\.\d+)'
    )

    # ===================== 打开串口 =====================
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            stopbits=1,
            parity=serial.PARITY_NONE,
            timeout=0.1
        )
        print(f"✅ 成功打开串口：{port}，波特率：{baudrate}")
        print(f"📄 数据将保存到：{json_file_path}")
        print("💡 按 Ctrl+C 停止采集")
    except Exception as e:
        print(f"❌ 打开串口失败：{e}")
        return

    # ===================== 核心采集逻辑 =====================
    try:
        while not stop_collection:
            serial_line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not serial_line:
                continue

            # 1. 解析AD值（适配新格式，对应关系完全不变）
            ad_match = ad_pattern.search(serial_line)
            if ad_match:
                current_ad_values = [
                    int(ad_match.group(1)),  # AD1 - 拇指thumb
                    int(ad_match.group(2)),  # AD2 - 食指index
                    int(ad_match.group(3)),  # AD3 - 中指middle
                    int(ad_match.group(4)),  # AD4 - 无名指ring
                    int(ad_match.group(5))   # AD5 - 小指pinky
                ]
                current_data_group.clear()
                current_sample_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                continue

            # 2. 解析MPU数据（适配新格式，数据结构完全不变）
            mpu_match = mpu_pattern.search(serial_line)
            if mpu_match and current_ad_values is not None and current_sample_time is not None:
                mpu_id = int(mpu_match.group(1))
                pitch = round(float(mpu_match.group(2)), 2)
                roll = round(float(mpu_match.group(3)), 2)
                yaw = round(float(mpu_match.group(4)), 2)
                ax = round(float(mpu_match.group(5)), 2)
                ay = round(float(mpu_match.group(6)), 2)
                az = round(float(mpu_match.group(7)), 2)

                # 获取对应的手指名称
                finger_name = ID_TO_FINGER.get(mpu_id, f"unknown_{mpu_id}")

                # 构建finger数据（完全沿用原有逻辑，无任何修改）
                if 1 <= mpu_id <= 4:
                    # MPU1-4：对应AD1-AD4，正常处理
                    ad_value = current_ad_values[mpu_id - 1]
                    calibrated_angle = calibrate_sensor(finger_name, ad_value)
                    current_data_group[mpu_id] = {
                        "finger_name": finger_name,
                        "AD": ad_value,
                        "angle": calibrated_angle,
                        "pitch": pitch,
                        "roll": roll,
                        "yaw": yaw,
                        "ax": ax,
                        "ay": ay,
                        "az": az
                    }
                elif mpu_id == 5:
                    # MPU5：手背数据，无AD值和角度
                    current_data_group[mpu_id] = {
                        "finger_name": finger_name,
                        "pitch": pitch,
                        "roll": roll,
                        "yaw": yaw,
                        "ax": ax,
                        "ay": ay,
                        "az": az
                    }
                elif mpu_id == 6:
                    # MPU6：小指数据，使用AD5的值
                    ad_value = current_ad_values[4]  # AD5对应索引4
                    calibrated_angle = calibrate_sensor(finger_name, ad_value)
                    current_data_group[mpu_id] = {
                        "finger_name": finger_name,
                        "AD": ad_value,
                        "angle": calibrated_angle,
                        "pitch": pitch,
                        "roll": roll,
                        "yaw": yaw,
                        "ax": ax,
                        "ay": ay,
                        "az": az
                    }

                # 3. 凑齐6个MPU数据，保存样本（原有逻辑完全不变）
                if len(current_data_group) == 6:
                    # 核心：使用自定义顺序排序，而非默认升序
                    finger_list = [current_data_group[i] for i in CUSTOM_MPU_ORDER]
                    all_collected_data[current_sample_time] = finger_list

                    # 拼接JSON字符串
                    json_parts = []
                    for time_key, fingers in all_collected_data.items():
                        fingers_str = json.dumps(fingers, separators=(',', ':'))
                        json_parts.append(f'"{time_key}":{fingers_str}')

                    final_json = '{' + ',\n  '.join(json_parts) + '}'

                    # 写入文件
                    with open(json_file_path, 'w', encoding='utf-8') as f:
                        f.write(final_json)

                    print(f"✅ 已保存样本 | 时间：{current_sample_time} | 累计样本数：{len(all_collected_data)}")

                    # 重置临时变量
                    current_data_group.clear()
                    current_ad_values = None
                    current_sample_time = None

    except Exception as e:
        print(f"\n❌ 程序异常：{e}")
        import traceback
        traceback.print_exc()
    finally:
        # 最终保存数据
        if all_collected_data:
            json_parts = []
            for time_key, fingers in all_collected_data.items():
                fingers_str = json.dumps(fingers, separators=(',', ':'))
                json_parts.append(f'"{time_key}":{fingers_str}')
            final_json = '{' + ',\n  '.join(json_parts) + '}'
            with open(json_file_path, 'w', encoding='utf-8') as f:
                f.write(final_json)

            print(f"\n📊 采集完成！共保存 {len(all_collected_data)} 个有效样本")
            print(f"📂 数据文件位置：{json_file_path}")
        else:
            print("\n⚠️  未采集到任何有效样本")

        if ser.is_open:
            ser.close()
        print("🔌 串口已关闭")
        sys.exit(0)


if __name__ == "__main__":
    # 检查pyserial依赖
    try:
        import serial
    except ImportError:
        print("❌ 请先安装依赖：pip install pyserial")
        sys.exit(1)
    main()
