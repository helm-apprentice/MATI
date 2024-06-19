from PIL import Image
import os

from PIL import Image



import matplotlib.pyplot as plt

def onclick(event):
    if event.button == 1:  # 左键点击
        x, y = int(round(event.xdata)), int(round(event.ydata))
        print(f'Clicked at pixel coordinates: ({x}, {y})')

# 读取图像
image_path = '/home/helm/Desktop/figure/plane/stage3/attn_layer2_206_double.png'
img = plt.imread(image_path)

# 创建图像窗口并显示图像
fig, ax = plt.subplots()
ax.imshow(img)

# 添加鼠标点击事件监听器
fig.canvas.mpl_connect('button_press_event', onclick)

# 显示图像窗口，等待用户点击
plt.show()

# 用户手动记录每次点击的坐标，用于后续裁剪操作


import os
from PIL import Image

def _crop_regions(image_path, regions, output_dir):
    # 打开原始图像文件
    with Image.open(image_path) as img:
        width, height = img.size

        for i, region in enumerate(regions):
            left, top, right, bottom = region

            # 验证裁剪区域是否在图像范围内
            if not (0 <= left < width and 0 <= top < height and 0 <= right < width and 0 <= bottom < height):
                raise ValueError(f"Region {i} coordinates are out of bounds.")

            # 裁剪并保存子图
            cropped_img = img.crop((left, top, right, bottom))
            save_path = os.path.join(output_dir, f'region_{i}.png')
            cropped_img.save(save_path)

# 示例用法
regions = [
    (106, 72, 977, 941),  # 第一个方块的左上角和右下角像素坐标
    (1094, 158, 1792, 855),  # 第二个方块的左上角和右下角像素坐标
]


import os
from PIL import Image

def crop_regions(image_path, regions, output_dir):
    # 打开原始图像文件
    with Image.open(image_path) as img:
        width, height = img.size

        base_name = os.path.splitext(os.path.basename(image_path))[0]  # 提取原图像文件名（不含扩展名）

        for i, region in enumerate(regions):
            left, top, right, bottom = region

            # 验证裁剪区域是否在图像范围内
            if not (0 <= left < width and 0 <= top < height and 0 <= right < width and 0 <= bottom < height):
                raise ValueError(f"Region {i} coordinates are out of bounds.")

            # 裁剪并保存子图
            cropped_img = img.crop((left, top, right, bottom))
            save_path = os.path.join(output_dir, f"{base_name}_region_{i}.png")  # 将原图像文件名添加到裁剪后图像的文件名中
            cropped_img.save(save_path)

def process_images(root_dir, regions):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.png'):
                continue

            image_path = os.path.join(dirpath, filename)
            output_dir = dirpath  # 保存裁剪后的图像至当前文件夹
            crop_regions(image_path, regions, output_dir)

# 示例用法
# regions = [
#     (106, 72, 977, 941),  # 第一个方块的左上角和右下角像素坐标
#     (1094, 158, 1792, 855),  # 第二个方块的左上角和右下角像素坐标
# ]
regions = [
    (107, 84, 952, 928),  # 第一个方块的左上角和右下角像素坐标
    (1109, 169, 1785, 841),  # 第二个方块的左上角和右下角像素坐标
]
root_dir = '/home/helm/Desktop/figure/plane'

process_images(root_dir, regions)

# crop_regions('/home/helm/Desktop/figure/bird2/attn_layer4_11_single.png', regions, '/home/helm/Desktop/figure/cropped_regions')