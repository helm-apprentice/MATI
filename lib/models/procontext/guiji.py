import cv2
import numpy as np

# 读取图像
image = cv2.imread('/home/helm/tracker/acsm.png')

# 将图像转换为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化图像
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 寻找轮廓
contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历所有轮廓
for contour in contours:
    # 筛选条件：假设字母的轮廓面积在一个合理的范围内
    if 100 < cv2.contourArea(contour) < 10000:  # 这些值可能需要根据你的图像进行调整
        # 获取轮廓点
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 打印轮廓点的坐标
        for pt in approx:
            x, y = pt[0]
            print(f"Contour point at ({x}, {y})")

        # 如果需要在原图上绘制轮廓点，取消下面两行的注释
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        cv2.imshow('Image with Contour Points', image)

# 显示带有轮廓点的图像（如果需要）
cv2.waitKey(0)
cv2.destroyAllWindows()


# 读取图像
# image = cv2.imread('/home/helm/tracker/acsm.png')

# # 将图像转换为灰度
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # 二值化图像
# _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# # 寻找轮廓
# contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 遍历所有轮廓
# for contour in contours:
#     # 筛选条件：假设字母的轮廓面积在一个合理的范围内
#     if 100 < cv2.contourArea(contour) < 10000:  # 这些值可能需要根据你的图像进行调整
#         # 获取轮廓的中心点（质心）
#         moments = cv2.moments(contour)
#         if moments['m00'] != 0:
#             cx = int(moments['m10']/moments['m00'])
#             cy = int(moments['m01']/moments['m00'])
#             print(f"Centroid of the contour at ({cx}, {cy})")

#             # 在原图上绘制质心点，用于可视化
#             cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

# # 显示带有质心点的图像
# cv2.imshow('Image with Centroid Points', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

