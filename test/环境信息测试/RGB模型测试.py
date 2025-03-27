import numpy as np
import matplotlib.pyplot as plt

# 定义图像尺寸
height, width = 100, 100

# 创建一个空白图像数组，初始值为0（黑色）
image = np.zeros((height, width, 3), dtype=np.uint8)

# 定义矩形的中心坐标（x, y），假设 y 从底部开始
x = np.array([25, 25, 75, 75])
y = np.array([25, 75, 75, 25])

# 定义矩形的宽度和高度
rect_width = np.array([50, 50, 50, 50])
rect_height = np.array([50, 50, 50, 50])

# 定义颜色（蓝色、绿色、红色、黄色）
RGB = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0]])

# 填充矩形
for i in range(4):
    # 计算矩形的左上角和右下角坐标
    left = int(x[i] - rect_width[i] / 2)
    right = int(x[i] + rect_width[i] / 2)
    bottom = int(y[i] - rect_height[i] / 2)
    top = int(y[i] + rect_height[i] / 2)
    
    # 将 y 坐标转换为图像的行索引（y=0 在顶部）
    row_start = height - top
    row_end = height - bottom
    
    # 确保不越界
    row_start = max(0, row_start)
    row_end = min(height, row_end)
    col_start = max(0, left)
    col_end = min(width, right)
    
    # 填充颜色
    image[row_start:row_end, col_start:col_end, :] = RGB[i]

# 显示图像
plt.imshow(image)
plt.title("大正方形和小矩形")
plt.axis("on")  # 显示坐标轴以验证位置
plt.show()


# 将起点为左上角的坐标转换为起点为左下角的坐标
def coordinate_transformation(x,y,W,H):
    x = x
    y = H - y
    return x,y
