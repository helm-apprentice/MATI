import os
from typing import List, Tuple
import numpy as np
import re

def calculate_metrics(prediction_folder: str, original_data_folder: str) -> Tuple[float, float, float, float]:
    # 读取预测坐标
    prediction_files = [file for file in os.listdir(prediction_folder) if file.endswith('.txt')]
    precision_results = []
    success_rate_thresholds = [0.5, 0.75, 0.9]
    success_rates_by_threshold = {threshold: [] for threshold in success_rate_thresholds}
    aor_results = []
    ape_results = []

    for pred_file in prediction_files:
        predictions = []
        with open(os.path.join(prediction_folder, pred_file), 'r') as f:
            for line in f:
                if line.startswith('FPS'):
                    fps = float(line.split(':')[1].strip())
                else:
                    bbox = tuple(map(float, line.strip().split(',')[:4]))
                    predictions.append(bbox)

        pred_name = os.path.splitext(pred_file)[0]
        visible_gt_folder = os.path.join(original_data_folder, pred_name)
        visible_gt_file = os.path.join(visible_gt_folder, 'groundtruth.txt')
        visible_gt_file_alter = os.path.join(visible_gt_folder, 'groundtruth_rect.txt')
        visible_gt_file = visible_gt_file if os.path.exists(visible_gt_file) else visible_gt_file_alter

        if not os.path.exists(visible_gt_file):
            print(f"Skipped sequence '{pred_name}': missing GT file.")
            continue  # 如果可见光GT文件不存在，则跳过

        visible_gts = []
        with open(visible_gt_file, 'r') as f:
            for line in f:
                line = line.replace(' ,', ',')  # 去除多余的空格
                line_parts = re.split(',|\s+', line.strip())[:4]
                non_empty_parts = [part for part in line_parts if part != '']
                if len(non_empty_parts) < 4:
                    print(f"{visible_gt_file} Skipped invalid line with fewer than 4 valid values: {line.strip()}")
                else:
                    bbox = tuple(map(float, non_empty_parts))
                visible_gts.append(bbox)

        assert len(predictions) == len(visible_gts), f"Check sequence '{pred_name}': prediction and visible GT count mismatch."

        # 计算当前序列的Precision、Success Rate、AOR和APE
        seq_precision = []
        seq_success_rates_by_threshold = {threshold: [] for threshold in success_rate_thresholds}
        seq_aor = []
        seq_ape = []

        for i in range(len(predictions)):
            pred_bbox = predictions[i]
            visible_gt_bbox = visible_gts[i]

            pred_center = (pred_bbox[0] + pred_bbox[2] / 2, pred_bbox[1] + pred_bbox[3] / 2)
            visible_gt_center = (visible_gt_bbox[0] + visible_gt_bbox[2] / 2, visible_gt_bbox[1] + visible_gt_bbox[3] / 2)

            # 精确度（Precision）
            iou = compute_iou(pred_bbox, visible_gt_bbox)
            seq_precision.append(iou)

            # 成功率（Success Rate）
            
            for threshold in success_rate_thresholds:
                if iou >= threshold:
                    seq_success_rates_by_threshold[threshold].append(1)
                else:
                    seq_success_rates_by_threshold[threshold].append(0)

            # 平均重叠率（AOR）
            seq_aor.append(iou)

            # 平均像素误差（APE）
            ape = euclidean(pred_center, visible_gt_center)
            seq_ape.append(ape)

        # 将当前序列的Precision、Success Rate、AOR和APE结果添加到总结果列表
        precision_results.extend(seq_precision)
        for threshold, seq_success_rates in seq_success_rates_by_threshold.items():
            success_rates_by_threshold[threshold].extend(seq_success_rates)
        aor_results.extend(seq_aor)
        ape_results.extend(seq_ape)

    # 计算平均Precision、Success Rate、AOR和APE
    avg_precision = np.mean(precision_results) if precision_results else 0
    avg_success_rates = {threshold: np.mean(results) if results else 0 for threshold, results in success_rates_by_threshold.items()}
    avg_aor = np.mean(aor_results) if aor_results else 0
    avg_ape = np.mean(ape_results) if ape_results else 0

    return avg_precision, avg_success_rates, avg_aor, avg_ape


def compute_iou(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """
    计算两个边界框的交并比（IoU）。
    """
    # 计算交集矩形的左上角坐标和右下角坐标
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y_bottom = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算交集面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算并集面积
    union_area = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - intersection_area

    return intersection_area / union_area


def euclidean(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    计算两点间的欧氏距离。
    """
    x_diff = point1[0] - point2[0]
    y_diff = point1[1] - point2[1]
    return np.sqrt(x_diff ** 2 + y_diff ** 2)

#prediction_folder = "/home/helm/tracker/mine/tests/tracking_results/ostrack/vitb_256_mae_ce_32x4_got10k_ep100_002"
prediction_folder = "/home/helm/tracker/ProContEXT-main/lib/test/tracking_results/procontext/cmc3_1053"
original_data_folder = "/home/helm/tracker/mine/data/val"
# original_data_folder = "/media/helm/C4E1CE1E0192B573/udata/otb100/otb100"
avg_precision, avg_success_rates, avg_aor, avg_ape = calculate_metrics(prediction_folder, original_data_folder)
print("Evaluation Metrics:")
# print(f"Average Precision: {avg_precision:.2f}")
print("Average Success Rates by Threshold:")
for threshold, success_rate in avg_success_rates.items():
    print(f"Threshold {threshold}: {success_rate:.2f}")
print(f"Average Overlap Rate (AOR): {avg_aor:.2f}")
print(f"Average Pixel Error (APE): {avg_ape:.2f}")