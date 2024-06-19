import os
from xml.etree import ElementTree as ET
from PIL import Image
import shutil
from pathlib import Path
import cv2
import numpy as np
import random

def convert_annotations(folder_path, output_file_path):
    """
    把所有的xml文件里的voc坐标转换为coco坐标并放在一个文本文档内
    voc → coco
    xmin, ymin, xmax, ymax → xmin, ymin, w, h
    """
    # 打开输出文件
    with open(output_file_path, 'w') as output_file:
        # 遍历指定文件夹中的所有文件
        for filename in os.listdir(folder_path):
            # 检查文件是否为XML文件
            if filename.endswith('.xml'):
                file_path = os.path.join(folder_path, filename)
                
                # 解析XML文件
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # 获取图像文件名（无扩展名）
                image_filename = root.find('filename').text.split('.')[0]
                
                # 遍历XML树，找到所有的bndbox元素，并进行转换
                for obj in root.iter('object'):
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    # 计算宽度和高度
                    w = xmax - xmin
                    h = ymax - ymin
                    
                    # 转换为所需格式，并保留四位小数
                    converted_line = f"{image_filename}: {xmin:.4f},{ymin:.4f},{w:.4f},{h:.4f}"
                    output_file.write(converted_line + '\n')

def save_fourfollwingPoint(source_path, dest_path):
    """
    将文档内的所有坐标都保留四位小数
    """
    with open(source_path, 'r') as input:
        lines = input.readlines()
    processed_lines = []
    for line in lines:
        parts = line.strip().split(' ', 1)
        print(parts)
        if len(parts) == 2:
            label, coords = parts
            coords_parts = coords.split(',')
            if len(coords_parts) == 4:
                rounded_coords = [f"{float(coord):.4f}" for coord in coords_parts]
                processed_line = f"{label}: {','.join(rounded_coords)}"
                processed_lines.append(processed_line)

    with open(dest_path, 'w') as out:
        for processed_line in processed_lines:
            out.write(processed_line + '\n')

def partCover_anno(source_path, dest_path, out_path):
    """
    将之前修改的部分图像的坐标替换到原始坐标里
    并且删掉标签
    """
    source_content = {}
    with open(source_path, 'r') as a:
        for line in a:
            key, value = line.strip().split(': ')
            source_content[key] = value
    updated_lines = []
    with open(dest_path, 'r') as b:
        for line in b:
            if ': ' in line:
                key, _ = line.strip().split(': ')
                if key in source_content:
                    # updated_lines.append(f"{source_content[key]}")
                    updated_lines.append(f"{key}: {source_content[key]}")
                else:
                    # updated_lines.append(line.strip().split(': ')[1])
                    updated_lines.append(line.strip())
            else:
                updated_lines.append(line.strip())
    with open(out_path, 'w') as out:
        for line in updated_lines:
            out.write(line + '\n')

def resize_Image(source_path, dest_path, resolution):
    """
    调整文件夹下图像的分辨率
    """
    os.makedirs(dest_path, exist_ok=True)
    for filename in os.listdir(source_path):
        file_path = os.path.join(source_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', 'bmp')):
            try:
                with Image.open(file_path) as img:
                    resized_img = img.resize(resolution)
                    dest_file_path = os.path.join(dest_path, filename)
                    resized_img.save(dest_file_path)
                    print(f"resized done and saved: {filename}")
            except Exception as e:
                print(f"Error processing {filename} : {e}")

def convert_coordinates(input_file, output_file, original_resolution, target_resolution):
    """
    Convert bounding box coordinates from original resolution to target resolution.

    Parameters:
    - input_file: Path to the input file containing original coordinates.
    - output_file: Path to the output file for converted coordinates.
    - original_resolution: A tuple specifying the original resolution (width, height).
    - target_resolution: A tuple specifying the target resolution (width, height).
    """
    # 计算宽度和高度的缩放比例
    scale_width = target_resolution[0] / original_resolution[0]
    scale_height = target_resolution[1] / original_resolution[1]

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split(': ')
            label = parts[0]
            coords = parts[1].split(',')
            if len(coords) == 4:
                # 转换坐标
                xmin = float(coords[0]) * scale_width
                ymin = float(coords[1]) * scale_height
                w = float(coords[2]) * scale_width
                h = float(coords[3]) * scale_height

                # 写入转换后的坐标到输出文件
                outfile.write(f"{label}: {xmin:.4f},{ymin:.4f},{w:.4f},{h:.4f}\n")

def count_images_in_folder(folder):
    """计算指定文件夹中的图像数量。"""
    return sum([1 for item in os.listdir(folder) if item.endswith('.bmp')])

def align_and_copy_images(visible_start_index, infrared_start_index, visible_img_folder, infrared_img_folder, target_folder):
    """
    对齐并复制图像到目标文件夹。
    
    参数:
    - visible_start_index: 可见光图像起始编号
    - infrared_start_index: 红外图像起始编号
    - N_v: 可见光图像总数
    - N_i: 红外图像总数
    - visible_img_folder: 可见光图像文件夹路径
    - infrared_img_folder: 红外图像文件夹路径
    - target_folder: 目标文件夹路径
    """
    # 计算图像总数
    N_v = count_images_in_folder(visible_img_folder)
    N_i = count_images_in_folder(infrared_img_folder)

    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 对每个可见光图像找到对应的红外图像
    for n_v in range(visible_start_index, visible_start_index + N_v):
        n_i_estimated = round((n_v - visible_start_index + 1) * (N_i / N_v))
        n_i_actual = n_i_estimated + infrared_start_index - 1
        infrared_img_filename = f'image{str(n_i_actual).zfill(6)}.bmp'
        infrared_img_path = os.path.join(infrared_img_folder, infrared_img_filename)
        target_img_path = os.path.join(target_folder, infrared_img_filename)
        
        # 复制图像
        if os.path.exists(infrared_img_path):
            shutil.copy(infrared_img_path, target_img_path)
            print(f'Copied {infrared_img_filename} to {target_folder}')
        else:
            print(f'Image {infrared_img_filename} does not exist.')


def filter_coordinates(input_file, output_file, image_folder):
    """
    Filter the coordinates in the input file, keeping only those corresponding to images
    that exist in the specified image folder.

    Parameters:
    - input_file: Path to the input file containing labels and coordinates.
    - output_file: Path to the output file for filtered coordinates.
    - image_folder: Path to the folder containing the images.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Assuming each line starts with "imageXXXXXX:"
            image_id = line.split(':')[0]  # Extract image ID
            image_file = f"{image_id}.bmp"  # Construct image file name
            
            # Check if the image file exists in the given image folder
            if os.path.exists(os.path.join(image_folder, image_file)):
                # If the image exists, write the line to the output file
                outfile.write(line)


def split_image_dataset_in_order(source_dir, target_dir, subset_size):
    """
    按顺序分割图像数据集到多个子集中。
    
    :param source_dir: 原始图像集的目录。
    :param target_dir: 分割后的子集存储目录。
    :param subset_size: 每个子集的图像数量。
    """
    # 确保目标目录存在
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像路径，并按文件名排序以保持顺序
    images = [f for f in sorted(os.listdir(source_dir)) if os.path.isfile(os.path.join(source_dir, f))]
    
    total_images = len(images)
    for i in range(0, total_images, subset_size):
        subset_images = images[i:i+subset_size]
        subset_dir = os.path.join(target_dir, f'subset_a{i//subset_size}')
        os.makedirs(subset_dir, exist_ok=True)
        
        # 将选中的图像复制到新的子集目录
        for img in subset_images:
            shutil.copy(os.path.join(source_dir, img), os.path.join(subset_dir, img))
        
        print(f'Created {subset_dir} with {len(subset_images)} images ordered.')


def copy_selected_images(label_file, source_folder, target_folder):
    """
    Copy images with names listed in the label file from the source folder to the target folder.
    根据标签选择图像
    Parameters:
    - label_file: Path to the file containing labels and coordinates.
    - source_folder: Path to the folder containing the source images.
    - target_folder: Path to the folder where selected images will be copied.
    """
    # Ensure target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    with open(label_file, 'r') as file:
        for line in file:
            image_name = line.split(':')[0] + '.bmp'  # Assuming image names end with '.bmp'
            source_path = os.path.join(source_folder, image_name)
            
            if os.path.exists(source_path):
                target_path = os.path.join(target_folder, image_name)
                shutil.copy2(source_path, target_path)  # Copy image to target folder



# 使用示例
folder_path = 'E:/vitb_384_mae_ce_32x4_ep300/Annotations_可见02'  # 修改为您的XML文件所在的文件夹路径
part_file_path = 'E:/vitb_384_mae_ce_32x4_ep300/Annotations_可见02.txt'  # 修改为您希望保存结果的文件路径
source_path = "E:/vitb_384_mae_ce_32x4_ep300/upupdated_sequence_可见02.txt"
dest_path = "E:/vitb_384_mae_ce_32x4_ep300/last_sequence_可见02.txt"
# convert_annotations(folder_path, part_file_path)
# save_fourfollwingPoint(source_path, dest_path)
# partCover_anno(part_file_path, source_path, dest_path)
img_sourcePath = "/media/helm/T7/可见＋红外/可见/可见02/[高速相机存储-光口存储-0004-00000001-00009953]20231201-125752"
img_destPath = "/media/helm/T7/可见＋红外/可见/可见02(640, 512)"
resolution = (640, 512)
# resize_Image(img_sourcePath, img_destPath, resolution)
input_file = 'E:/vitb_384_mae_ce_32x4_ep300/last_sequence_可见01.txt'  # 输入文件路径
output_file = 'E:/vitb_384_mae_ce_32x4_ep300/last_sequence_可见01(640, 512).txt'  # 输出文件路径
original_resolution = (1920, 1080)  # 原始分辨率
target_resolution = (640, 512)  # 目标分辨率
# convert_coordinates(input_file, output_file, original_resolution, target_resolution)
visible_start_index = 1  # 可见图像起始编号
infrared_start_index = 51  # 红外图像起始编号

# 文件夹路径，根据实际情况修改
visible_img_folder = 'E:/vitb_384_mae_ce_32x4_ep300/sequence_可见02(640, 512)'
infrared_img_folder = 'E:/vitb_384_mae_ce_32x4_ep300/sequence_fenjie'
target_folder = 'E:/vitb_384_mae_ce_32x4_ep300/sequence_02(align)'

# align_and_copy_images(visible_start_index, infrared_start_index, visible_img_folder, infrared_img_folder, target_folder)
input_file = 'E:/vitb_384_mae_ce_32x4_ep300/processed/visible_01.txt'  # Input file path
output_file = 'E:/vitb_384_mae_ce_32x4_ep300/processed/visible/subset_a3.txt'  # Output file path
image_folder = 'E:/vitb_384_mae_ce_32x4_ep300/processed/visible/subset_a3'  # Image folder path

# filter_coordinates(input_file, output_file, image_folder)
# 示例用法
source_dir = 'E:/vitb_384_mae_ce_32x4_ep300/processed/visible_01'
target_dir = 'E:/vitb_384_mae_ce_32x4_ep300/processed/visible'
subset_size = 500  # 你希望每个子集的大小

# split_image_dataset_in_order(source_dir, target_dir, subset_size)

label_file = '/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/infrared_02.txt'  # Path to the label file
source_folder = '/media/helm/T7/可见＋红外/红外/红外02/fenjie'  # Path to the source images folder
target_folder = '/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/infrared002'  # Path to the target folder where images will be copied

# copy_selected_images(label_file, source_folder, target_folder)

dataset_details = {
    'Dataset1': {
        'Ground': {'start': 1, 'end': 800},
        'Climb': {'start': 801, 'end': 1100},
        'HighAltitude': {'start': 1101, 'end': 1895}
    },
    'Dataset2': {
        'Ground': {'start': 1, 'end': 600},
        'Climb': {'start': 601, 'end': 1000},
        'HighAltitude': {'start': 1001, 'end': 7520}
    }
}




def apply_low_light_correction(image, gamma=10):
    """
    Apply low light correction to simulate low light condition on an image.
    """
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, look_up_table)

def process_low_light_frames(stages, source_dir, target_dir, train_percentage=0.8, validate_test_percentage=0.4, select_method='random'):
    """
    Process frames for low light condition and save them to the target directory.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for stage_name, phase_info in stages.items():
        for phase, frame_range in phase_info.items():
            start, end = frame_range
            total_frames = end - start + 1
            num_frames_to_select = int(total_frames * (train_percentage if phase == 'train' else validate_test_percentage))
            if select_method == 'random':
                selected_frames = sorted(random.sample(range(start, end + 1), num_frames_to_select))
            else:
                step = total_frames // num_frames_to_select
                selected_frames = [i for i in range(start, end+1, step)]

            for frame_number in selected_frames:
                image_path = os.path.join(source_dir, f"{frame_number:08d}.jpg")
                if os.path.exists(image_path):
                    img = cv2.imread(image_path)
                    low_light_img = apply_low_light_correction(img)
                    cv2.imwrite(os.path.join(target_dir, f"{frame_number:08d}.jpg"), low_light_img)

# Example usage
source_dir = '/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/visible001(jpg)'
target_random_dir = '/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/visible002(random_dark3)'
target_sequence_dir = '/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/visible001(sequence_dark10)'
stages1 = {
    'ground': {'train': (1, 560), 'validate': (561, 680), 'test': (681, 800)},
    'climb': {'train': (801, 1010), 'validate': (1011, 1055), 'test': (1056, 1100)},
    'high_altitude': {'train': (1101, 1656), 'validate': (1657, 1775), 'test': (1776, 1895)},
}
stages2 = {
    'ground': {'train': (1, 420), 'validate': (421, 510), 'test': (511, 600)},
    'climb': {'train': (601, 880), 'validate': (881, 940), 'test': (941, 1000)},
    'high_altitude': {'train': (1001, 5554), 'validate': (5555, 6464), 'test': (6465, 7520)},
}

# Call the function with the provided directories and stages information
# process_low_light_frames(stages2, source_dir, target_random_dir, select_method='random')
process_low_light_frames(stages1, source_dir, target_sequence_dir, select_method='sequence')



def convert_images_to_jpg(source_dir):
    '''
    将文件夹下的所有图像转换为jpg格式后按序重新命名,例如image000052.bmp, image000054.bmp, image000056.bmp...
    重新命名之后为00000001.jpg, 00000002.jpg, 00000003.jpg...
    bmp图像仍会保留
    '''
    # 获取目录下的所有文件
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    # 过滤出图像文件
    image_files = [f for f in files if f.lower().endswith(('.png', '.bmp', '.jpeg', '.jpg', '.gif'))]

    # 按文件名排序，确保顺序
    image_files.sort()

    # 用于生成新文件名的计数器
    counter = 1

    for image_file in image_files:
        # 构建完整的文件路径
        file_path = os.path.join(source_dir, image_file)
        
        # 打开并转换图像
        with Image.open(file_path) as img:
            # 构建新的文件名
            new_filename = f'{counter:08d}.jpg'
            new_file_path = os.path.join(source_dir, new_filename)
            
            # 转换图像并保存
            img.convert('RGB').save(new_file_path, 'JPEG')
            
            print(f'Converted and saved {image_file} as {new_filename}')
            
            # 更新计数器
            counter += 1

# 调用函数，'source_dir'是你的图像文件所在的目录
# convert_images_to_jpg('/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/visible002jpg(random_dark3+4)')


def delete_bmp_images(source_dir):
    '''
    删除文件夹下所有bmp图像
    '''
    # 获取目录下的所有文件
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    # 过滤出.bmp图像文件
    bmp_files = [f for f in files if f.lower().endswith('.bmp')]

    for bmp_file in bmp_files:
        # 构建完整的文件路径
        file_path = os.path.join(source_dir, bmp_file)
        # 删除文件
        os.remove(file_path)
        print(f'Deleted {bmp_file}')

# delete_bmp_images('/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/infrared001(jpg)')
        


def detect_occlusion(image, bbox):
    # 提取边界框区域
    xmin, ymin, width, height = map(int, bbox)
    xmax, ymax = xmin + width, ymin + height
    roi = image[ymin:ymax, xmin:xmax]

    # 计算标准差
    std_dev = np.std(roi)
    # 计算区域内像素的方差
    variance = np.var(roi)
    # 纹理分析
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    # lbp = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
    laplacian_var = cv2.Laplacian(gray_roi,cv2.CV_64F).var() # laplacian算子计算边缘的清晰度
    # 颜色一致性分析
    # color_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    # color_uniformity = np.std(color_hist)

    # 将标准差转换为0到8的评分
    score_std = min(8, int(std_dev / 7))
    score_var = min(8, int(variance / 500))
    score_lbp = min(8, int(laplacian_var / 10))
    #score_color = min(8, int(color_uniformity / 15))

    score = round((score_std + score_var + score_lbp) / 3)
    return score




def process_sequence(folder_path):
    groundtruth_path = os.path.join(folder_path, 'groundtruth.txt')
    cover_label_path = os.path.join(folder_path, 'cover.label')

    # 读取groundtruth.txt文件
    with open(groundtruth_path, 'r') as f:
        bboxes = [list(map(float, line.strip().split(','))) for line in f.readlines()]

    image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".png"))]
    image_files.sort(key=lambda f: int(f.split('.')[0]))
    
    # 遍历图像计算遮挡程度
    cover_scores = []
    for idx, bbox in enumerate(bboxes):
        image_path = os.path.join(folder_path, f'{image_files[idx]}')  # 图像命名方式可能需要调整
        image = cv2.imread(image_path)
        if image is None:
            continue
        score = detect_occlusion(image, bbox)
        cover_scores.append(score)

    # 将遮挡评分保存到cover.label文件
    with open(cover_label_path, 'w') as file:
        for i, score in enumerate(cover_scores):
            image_file = image_files[i].split('.')[0]
            file.write(f'{image_file}:{score}\n')

def write_inf(folder_path):
    meta = os.path.join(folder_path, "meta_info.ini")

folder_path = '/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/visible001jpg(random_dark3+5)'
# process_sequence(folder_path)

def rearrange_annotations(input_filename, output_filename):
    """
    将标注文档中的图片编号重排。

    参数:
    input_filename (str): 原始标注数据的文件名。
    output_filename (str): 重排后数据保存的文件名。
    """
    # 读取原始标注数据
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    # 新建一个空列表用于存放重排后的数据
    rearranged_lines = []

    # 遍历原始数据，为每个标注分配新的编号
    for i, line in enumerate(lines):
        # 分割行文本以获取坐标数据
        parts = line.split(': ')
        # 生成新的图片编号，格式为image加上六位数的序号
        new_image_number = f"image{str(i+1).zfill(6)}"
        # 重组文本行
        new_line = f"{new_image_number}: {parts[1]}"
        # 将重组后的文本行添加到列表中
        rearranged_lines.append(new_line)

    # 将重排后的数据写入新文件
    with open(output_filename, 'w') as file:
        file.writelines(rearranged_lines)

    print(f"重排后的标注数据已保存到{output_filename}")

# 使用示例
input_filename = "/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/infrared_02.txt"  # 原始标注数据文件名
output_filename = "/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/infrared002(jpg)/groundtruth.txt"  # 输出文件名

# 调用函数
# rearrange_annotations(input_filename, output_filename)

def split_dataset(rearranged_filename, output_prefix, stages):
    """
    根据给定的划分规则将数据集分为训练集、验证集和测试集。

    参数:
    rearranged_filename (str): 重排后的标注数据文件名。
    output_prefix (str): 输出文件的前缀。
    stages (dict): 包含每个阶段的帧划分信息的字典。
    """
    # 读取重排后的标注数据
    with open(rearranged_filename, 'r') as file:
        lines = file.readlines()

    # 创建存储各个数据集的列表
    train_set, val_set, test_set = [], [], []

    # 处理每个阶段的数据划分
    for stage, splits in stages.items():
        for split_type, frame_range in splits.items():
            # 根据范围选取对应的行
            subset = [lines[i-1].split(': ')[1] for i in range(frame_range[0], frame_range[1]+1)]
            # 将选取的行添加到对应的数据集列表中
            if split_type == 'train':
                train_set.extend(subset)
            elif split_type == 'val':
                val_set.extend(subset)
            elif split_type == 'test':
                test_set.extend(subset)

    # 写入数据集到相应的文件中
    for dataset, name in [(train_set, 'train'), (val_set, 'val'), (test_set, 'test')]:
        with open(f"{output_prefix}_{name}.txt", 'w') as file:
            file.writelines(dataset)

# 第一组数据集的划分规则
stages_group_1 = {
    'ground': {'train': (1, 560), 'val': (561, 680), 'test': (681, 800)},
    'climb': {'train': (801, 1010), 'val': (1011, 1055), 'test': (1056, 1100)},
    'high_altitude': {'train': (1101, 1656), 'val': (1657, 1775), 'test': (1776, 1895)}
}
stages2 = {
    'ground': {'train': (1, 420), 'val': (421, 510), 'test': (511, 600)},
    'climb': {'train': (601, 880), 'val': (881, 940), 'test': (941, 1000)},
    'high_altitude': {'train': (1001, 5554), 'val': (5555, 6464), 'test': (6465, 7520)},
}
infrared01_label_path = "/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/infrared001(jpg)/groundtruth.txt"
infrared02_label_path = "/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/infrared002(jpg)/groundtruth.txt"
visible01_label_path = "/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/visible001jpg(random_dark3+5)/groundtruth.txt"
visible02_label_path = "/media/helm/T7/vitb_384_mae_ce_32x4_ep300/processed/visible002jpg(random_dark3+4)/groundtruth.txt"
# 调用函数为第一组数据集生成划分
# split_dataset(visible02_label_path, 'dataset_group_1', stages2)

# 如果需要处理第二组数据集，可以定义stages_group_2，并再次调用split_dataset函数。

