import os
import numpy as np
from scipy.spatial.distance import euclidean

# 定义计算IoU的函数
def compute_iou(box1, box2):
    # 计算两个矩形框的IoU
    # box1和box2的格式应为(x1, y1, w1, h1)和(x2, y2, w2, h2)
    b1_x1, b1_y1, b1_w, b1_h = box1
    b2_x1, b2_y1, b2_w, b2_h = box2

    b1_x2 = b1_x1 + b1_w
    b1_y2 = b1_y1 + b1_h
    b2_x2 = b2_x1 + b2_w
    b2_y2 = b2_y1 + b2_h

    # 计算重叠区域的坐标
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    # 如果没有重叠，则IoU为0
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union_area = (b1_w * b1_h) + (b2_w * b2_h) - inter_area
    return inter_area / union_area


import os
from math import sqrt
import re

def euclidean(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def extract_coordinates(file_path):
    coordinates = []

    with open(file_path, 'r') as f:
        for line in f:
            try:
                # 尝试按照带帧号和冒号的格式解析
                match = re.search(r':\s*(\d+\.\d+),(\d+\.\d+),(\d+\.\d+),(\d+\.\d+)', line)
                if match:
                    x, y, w, h = map(float, match.groups())
                    coordinates.append([x, y, w, h])
                else:
                    raise ValueError(f"Invalid coordinate format in line (with frame number): {line}")
            except ValueError:
                # 如果带帧号和冒号的格式解析失败，尝试仅解析四个坐标值
                try:
                    values = line.strip().split(',')[:4]
                    if len(values) == 4:
                        x, y, w, h = map(float, values)
                        coordinates.append([x, y, w, h])
                    else:
                        raise ValueError(f"Invalid coordinate format in line (without frame number): {line}")
                except ValueError as e:
                    print(f"Failed to parse coordinates from line: {line}\nError: {e}")

    return coordinates

def calculate_mpr_msr_with_visible_gt(prediction_folder, original_data_folder):
    # 读取预测坐标
    prediction_files = [file for file in os.listdir(prediction_folder) if file.endswith('.txt')]
    mpr_results = []
    msr_results = []
    fps_values = []

    for pred_file in prediction_files:
        predictions = []
        with open(os.path.join(prediction_folder, pred_file), 'r') as f:
            for line in f:
                if line.startswith('FPS'):
                    fps = float(line.split(':')[1].strip())
                    fps_values.append(fps)
                else:
                    bbox = tuple(map(float, line.strip().split(',')[:4]))
                    predictions.append(bbox)

        pred_name = os.path.splitext(pred_file)[0]
        visible_gt_folder = os.path.join(original_data_folder, pred_name, 'visible')
        visible_gt_file = os.path.join(visible_gt_folder, 'groundtruth.txt')

        if not os.path.exists(visible_gt_file):
            print(f"Skipped sequence '{pred_name}': missing visible GT file.")
            continue  # 如果可见光GT文件不存在，则跳过

        visible_gts = []

        visible_gts = extract_coordinates(visible_gt_file)

        # print('visible_gts:', len(visible_gts))
        # print('predictions:', len(predictions))
        assert len(predictions) == len(visible_gts), f"Check sequence '{pred_name}': prediction and visible GT count mismatch."

        # 计算当前序列的MPR和MSR
        seq_mpr_results = []
        seq_msr_results = []

        for i in range(len(predictions)):
            pred_bbox = predictions[i]
            visible_gt_bbox = visible_gts[i]

            pred_center = (pred_bbox[0] + pred_bbox[2] / 2, pred_bbox[1] + pred_bbox[3] / 2)
            visible_gt_center = (visible_gt_bbox[0] + visible_gt_bbox[2] / 2, visible_gt_bbox[1] + visible_gt_bbox[3] / 2)

            mpr = euclidean(pred_center, visible_gt_center)
            if mpr <= 10:
                seq_mpr_results.append(1)
            else:
                seq_mpr_results.append(0)

            visible_iou = compute_iou(pred_bbox, visible_gt_bbox)
            msr = visible_iou
            if msr >= 0.75:
                seq_msr_results.append(1)
            else:
                seq_msr_results.append(0)

        # 将当前序列的MPR和MSR结果添加到总结果列表
        mpr_results.extend(seq_mpr_results)
        msr_results.extend(seq_msr_results)

    # 计算平均MPR和MSR
    avg_mpr = sum(mpr_results) / len(mpr_results) if mpr_results else 0
    avg_msr = sum(msr_results) / len(msr_results) if msr_results else 0
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

    return avg_mpr, avg_msr, avg_fps


def calculate_mpr_msr(prediction_folder, original_data_folder):
    # 读取预测坐标
    prediction_files = [file for file in os.listdir(prediction_folder) if file.endswith('.txt')]
    mpr_results = []
    msr_results = []
    fps_values = []

    for pred_file in prediction_files:
        predictions = []
        with open(os.path.join(prediction_folder, pred_file), 'r') as f:
            for line in f:
                if line.startswith('FPS'):
                    fps = float(line.split(':')[1].strip())
                    fps_values.append(fps)
                else:
                    bbox = tuple(map(float, line.strip().split(',')[:4]))
                    predictions.append(bbox)

        pred_name = os.path.splitext(pred_file)[0]
        visible_gt_folder = os.path.join(original_data_folder, pred_name, 'visible')
        infrared_gt_folder = os.path.join(original_data_folder, pred_name, 'infrared')
        visible_gt_file = os.path.join(visible_gt_folder, 'groundtruth.txt')
        infrared_gt_file = os.path.join(infrared_gt_folder, 'groundtruth.txt')

        if not os.path.exists(visible_gt_file) or not os.path.exists(infrared_gt_file):
            print(f"Skipped sequence '{pred_name}': missing visible or infrared GT file.")
            continue  # 如果可见光或红外文件不存在，则跳过

        visible_gts = []
        infrared_gts = []

        visible_gts = extract_coordinates(visible_gt_file)
        infrared_gts = extract_coordinates(infrared_gt_file)
        # with open(visible_gt_file, 'r') as f:
        #     for line in f:
        #         bbox = tuple(map(float, line.strip().split(',')[:4]))
        #         visible_gts.append(bbox)
        # with open(infrared_gt_file, 'r') as f:
        #     for line in f:
        #         bbox = tuple(map(float, line.strip().split(',')[:4]))
        #         infrared_gts.append(bbox)

        # print('visible_gts:', len(visible_gts))
        # print('infrared_gts:', len(infrared_gts))
        # print('predictions:', len(predictions))
        assert len(predictions) == len(visible_gts) == len(infrared_gts), f"Check sequence '{pred_name}': prediction, visible GT, and infrared GT count mismatch."

        # 计算当前序列的MPR和MSR
        seq_mpr_results = []
        seq_msr_results = []

        for i in range(len(predictions)):
            pred_bbox = predictions[i]
            visible_gt_bbox = visible_gts[i]
            infrared_gt_bbox = infrared_gts[i]

            pred_center = (pred_bbox[0] + pred_bbox[2] / 2, pred_bbox[1] + pred_bbox[3] / 2)
            visible_gt_center = (visible_gt_bbox[0] + visible_gt_bbox[2] / 2, visible_gt_bbox[1] + visible_gt_bbox[3] / 2)
            infrared_gt_center = (infrared_gt_bbox[0] + infrared_gt_bbox[2] / 2, infrared_gt_bbox[1] + infrared_gt_bbox[3] / 2)

            mpr = min(euclidean(pred_center, visible_gt_center), euclidean(pred_center, infrared_gt_center))
            if mpr <= 10:
                seq_mpr_results.append(1)
            else:
                seq_mpr_results.append(0)

            visible_iou = compute_iou(pred_bbox, visible_gt_bbox)
            infrared_iou = compute_iou(pred_bbox, infrared_gt_bbox)
            msr = max(visible_iou, infrared_iou)
            if msr >= 0.75:
                seq_msr_results.append(1)
            else:
                seq_msr_results.append(0)

        # 将当前序列的MPR和MSR结果添加到总结果列表
        mpr_results.extend(seq_mpr_results)
        msr_results.extend(seq_msr_results)

    # 计算平均MPR和MSR
    avg_mpr = sum(mpr_results) / len(mpr_results) if mpr_results else 0
    avg_msr = sum(msr_results) / len(msr_results) if msr_results else 0
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

    return avg_mpr, avg_msr, avg_fps
# 设置路径
prediction_path = '/home/helm/tracker/ProContEXT-main/lib/test/tracking_results/procontext/cmc_gtot_fine5_210'
original_data_folder = '/media/helm/C4E1CE1E0192B573/udata/RGBT234+GTOT'
dd = '/media/helm/C4E1CE1E0192B573/udata/Siamcsr_test'

groundtruth_path = '/home/helm/tracker/ProContEXT-main/data'
#groundtruth_path = '/home/helm/tracker/ProContEXT-main/data/mpr_plane1dark'
predictions_path = '/home/helm/tracker/ProContEXT-main/lib/test/tracking_results/procontext/cmc3_974'
# 计算MPR和MSR
#mpr, msr, fps = calculate_mpr_msr_with_visible_gt(prediction_path, original_data_folder)
mpr, msr, fps = calculate_mpr_msr_with_visible_gt(predictions_path, groundtruth_path)
print(f'Average MPR: {mpr:.2f}')
print(f'Average MSR: {msr:.2f}')
print(f'Average FPS: {fps:.2f}')