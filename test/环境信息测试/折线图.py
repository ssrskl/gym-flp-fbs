import matplotlib.pyplot as plt
import pandas as pd

n = 30
instance = "Du62"
file_path = f"/Users/maoyan/Codes/Python/gym-flp-fbs/Files/ExpResult/{instance}-模拟退火算法1.xlsx"
is_repair = True
# 提取数据
datas = []
df = pd.read_excel(file_path)
x = list(range(1, n + 1))
repair_random = df[df['备注'] == '包含修复动作算子-随机初始解']['适应度值'].tolist()
repair_random.sort()
datas.append(repair_random)
repair_k = df[df['备注'] == '包含修复动作算子-K分初始解']['适应度值'].tolist()
repair_k.sort()
datas.append(repair_k)
unrepair_random = df[df['备注'] == '取消修复动作算子-随机初始解']['适应度值'].tolist()
unrepair_random.sort()
datas.append(unrepair_random)
unrepair_k = df[df['备注'] == '取消修复动作算子-K分初始解']['适应度值'].tolist()
unrepair_k.sort()
datas.append(unrepair_k)
labels = [
    "repair_random",
    "repair_k",
    "unrepair_random",
    "unrepair_k",
]

# 绘制多个数据系列的折线图
if is_repair:
    labels = labels[:2]
    datas = datas[:2]
else:
    labels = labels[2:]
    datas = datas[2:]
for i, y in enumerate(datas):
    plt.plot(x, y, label=labels[i], marker="o")
    for xi, yi in zip(x, y):
        plt.text(xi, yi, str(round(yi, 2)), ha="center", va="bottom")  # 保留两位小数

# 添加标题和标签
plt.title(instance)
plt.xlabel('Iterations')
plt.ylabel('Fitness')
# 显示图例
plt.legend()
# 显示图形
plt.show()