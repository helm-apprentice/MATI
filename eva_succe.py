import numpy as np
import os
import re

def load_coordinates(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    coordinates = []
    for line in lines:
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
                values = line.split(',')
                if len(values) == 4:
                    x, y, w, h = map(float, values)
                    coordinates.append([x, y, w, h])
                else:
                    raise ValueError(f"Invalid coordinate format in line (without frame number): {line}")
            except ValueError as e:
                pass
                #print(f"Failed to parse coordinates from line: {line}\nError: {e}")

    return np.array(coordinates)

def calculate_center_coordinates(x, y, w, h):
    """计算矩形框的中心坐标"""
    return x + w / 2, y + h / 2
def calculate_distance(x1, y1, x2, y2):
    """计算两点之间的欧氏距离"""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def track_evaluation(groundtruth_path, predictions_path, delta, inertia_frames, stability_frames):
    gt_coords = load_coordinates(groundtruth_path)
    pred_coords = load_coordinates(predictions_path)
    n_frames = len(gt_coords)

    misdetection_count = 0
    currently_in_misdetection = False
    last_misdetection_start = -1
    tracking_success = True

    i = 0
    while i < n_frames:
        gt_x, gt_y, gt_w, gt_h = gt_coords[i]
        pred_x, pred_y, pred_w, pred_h = pred_coords[i]
        distance = calculate_distance(gt_x + gt_w / 2, gt_y + gt_h / 2, pred_x + pred_w / 2, pred_y + pred_h / 2)
        ratio = distance / (gt_w * gt_h)

        if ratio > delta:
            if not currently_in_misdetection:
                currently_in_misdetection = True
                last_misdetection_start = i
                misdetection_count += 1

        # Check if the misdetection recovers at the end of inertia frames
        if currently_in_misdetection and i == last_misdetection_start + inertia_frames:
            if ratio <= delta:
                # Check stability for the subsequent stability_frames
                all_stable = True
                for j in range(1, stability_frames + 1):
                    if i + j >= n_frames:  # Ensure we do not go out of index
                        break
                    next_distance = calculate_distance(
                        gt_coords[i + j][0] + gt_coords[i + j][2] / 2, gt_coords[i + j][1] + gt_coords[i + j][3] / 2,
                        pred_coords[i + j][0] + pred_coords[i + j][2] / 2, pred_coords[i + j][1] + pred_coords[i + j][3] / 2)
                    next_ratio = next_distance / (gt_coords[i + j][2] * gt_coords[i + j][3])
                    if next_ratio > delta:
                        all_stable = False
                        break
                if all_stable:
                    currently_in_misdetection = False  # Misdetection resolved and stable
                else:
                    tracking_success = False
                    break
            else:
                tracking_success = False
                break
        i += 1


    return tracking_success, misdetection_count

def sequences_track_evaluation(original_data_folder, prediction_folder, delta, inertia_frames, stability_frames):
    prediction_files = [f for f in os.listdir(prediction_folder) if f.endswith('.txt')]
    success_counts = 0
    fail_counts = 0
    for pred_file in prediction_files:
        prediction_path = os.path.join(prediction_folder, pred_file)
        pred_name = os.path.splitext(pred_file)[0]
        gt_folder = os.path.join(original_data_folder, pred_name, 'visible')
        gt_path = os.path.join(gt_folder, 'groundtruth.txt')
        tracking_success, misdetection_count = track_evaluation(gt_path, prediction_path, delta, inertia_frames, stability_frames)
        if tracking_success == True:
            success_counts += 1
        else:
            fail_counts += 1
        print(f'{pred_name} Tracking:', tracking_success, 'Miss count:', misdetection_count)
    print(f"success rate: {success_counts / (success_counts + fail_counts)}")
# Example usage:
#groundtruth_path = '/media/helm/C4E1CE1E0192B573/udata/RGBT234+GTOT'
#predictions_path = '/home/helm/tracker/ProContEXT-main/lib/test/tracking_results/procontext/cmc_gtot_fine2_200'
#predictions_path = '/media/helm/C4E1CE1E0192B573/udata/Siamcsr_test'

delta = 0.02  # Distance threshold
inertia_frames = 5  # Number of inertia frames
stability_frames = 10  # Number of stability frames after recovery

# sequences_track_evaluation(groundtruth_path, predictions_path, delta, inertia_frames, stability_frames)


import numpy as np

def max_distance_to_area_ratio(groundtruth_path, predictions_path):

    def calculate_area(gt_w, gt_h):
        return gt_w * gt_h

    gt_coords = load_coordinates(groundtruth_path)
    pred_coords = load_coordinates(predictions_path)

    max_ratio = 0.0
    for i in range(len(gt_coords)):
        gt_x, gt_y, gt_w, gt_h = gt_coords[i]
        pred_x, pred_y, pred_w, pred_h = pred_coords[i]

        gt_center_x = gt_x + gt_w / 2
        gt_center_y = gt_y + gt_h / 2
        pred_center_x = pred_x + pred_w / 2
        pred_center_y = pred_y + pred_h / 2

        distance = calculate_distance(gt_center_x, gt_center_y, pred_center_x, pred_center_y)
        area = calculate_area(gt_w, gt_h)

        ratio = distance / area if area != 0 else 0  # Prevent division by zero
        max_ratio = max(max_ratio, ratio)

    return round(max_ratio, 4)

groundtruth_path = '/home/helm/tracker/ProContEXT-main/data/sequence_plane2/visible/groundtruth.txt'
groundtruth_path = '/home/helm/tracker/ProContEXT-main/data/plane1dark/visible/groundtruth.txt'
predictions_path = '/home/helm/tracker/ProContEXT-main/lib/test/tracking_results/procontext/cmc3_974/sequence_visible.txt'
predictions_path = '/home/helm/tracker/ProContEXT-main/lib/test/tracking_results/procontext/cmc3_975/sequence_plane1dark.txt'
#predictions_path = '/home/helm/tracker/ProContEXT-main/lib/test/tracking_results/procontext/cmc3_983/sequence_visible.txt'
#print(f"Maximum distance to area ratio: {max_distance_to_area_ratio(groundtruth_path, predictions_path)}")
tracking_success, misdetection_count = track_evaluation(groundtruth_path, predictions_path, delta, inertia_frames, stability_frames)
max_ratio = max_distance_to_area_ratio(groundtruth_path, predictions_path)
print(f"Tracking: {tracking_success}, miss: {misdetection_count}, max_ratio: {max_ratio}")
def _track_evaluation(groundtruth_path, predictions_path, delta, inertia_frames):
    gt_coords = load_coordinates(groundtruth_path)
    pred_coords = load_coordinates(predictions_path)
    n_frames = len(gt_coords)

    misdetection_count = 0
    currently_in_misdetection = False
    
    tracking_success = True

    i = 0
    while i < n_frames:
        gt_x, gt_y, gt_w, gt_h = gt_coords[i]
        pred_x, pred_y, pred_w, pred_h = pred_coords[i]
        distance = calculate_distance(gt_x + gt_w / 2, gt_y + gt_h / 2, pred_x + pred_w / 2, pred_y + pred_h / 2)

        if distance > delta:
            if not currently_in_misdetection:
                currently_in_misdetection = True
                
                misdetection_count += 1
        else:
            currently_in_misdetection = False

        if currently_in_misdetection:
            # Check all subsequent frames within inertia frames to confirm recovery or failure
            j = i + 1
            recovered = False
            recover_counts = 0
            while j <= i + inertia_frames and j < n_frames:
                next_distance = calculate_distance(gt_coords[j][0] + gt_coords[j][2] / 2, gt_coords[j][1] + gt_coords[j][3] / 2,
                                                  pred_coords[j][0] + pred_coords[j][2] / 2, pred_coords[j][1] + pred_coords[j][3] / 2)
                if next_distance <= delta:
                    recover_start = j
                    while recover_start <= i + inertia_frames:
                        next_distance = calculate_distance(gt_coords[recover_start][0] + gt_coords[recover_start][2] / 2, gt_coords[recover_start][1] + gt_coords[recover_start][3] / 2,
                                                          pred_coords[recover_start][0] + pred_coords[recover_start][2] / 2, pred_coords[recover_start][1] + pred_coords[recover_start][3] / 2)
                        if next_distance <= delta:
                            recover_counts += 1
                            j = recover_start
                        else: recover_counts=0
                        recover_start += 1
                    #currently_in_misdetection = False
                    if j-1 + recover_counts >= i+inertia_frames:
                        recovered = True
                    i = recover_start + 1 # Update 'i' to skip checked frames

                else: j += 1

            # If we reach the end of the inertia frames without recovery, tracking fails
            if j == i + inertia_frames and not recovered:
                tracking_success = False
                break

        i += 1

    return tracking_success, misdetection_count
# 使用示例

