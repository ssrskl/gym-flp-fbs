import matplotlib.pyplot as plt

# 示例数据
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
repair_random = [
    5263.513376,
    4909.767052,
    4685.297214,
    5173.356777,
    4921.167359,
    5461.846145,
    4878.822641,
    5185.552246,
    4442.136344,
    4594.8967,
]
repair_random.sort()
repair_k = [
    5026.3177,
    5013.133566,
    4383.36767,
    4825.63287,
    4588.824729,
    5007.464526,
    4649.068011,
    4811.777192,
    4761.358154,
    5157.960441,
]
repair_k.sort()
unrepair_random = [
    6138.634505,
    51581.16969,
    11813.12912,
    11557.63203,
    15224.8428,
    53691.99208,
    13502.57642,
    12643.92821,
    11265.67194,
    11299.78656,
]
unrepair_random.sort()
unrepair_k = [
    47805.08654,
    5553.349414,
    51246.09288,
    12563.4384,
    13093.13469,
    11958.00086,
    5648.934427,
    44978.07537,
    12080.34537,
    9592.553785,
]
unrepair_k.sort()
datas = [repair_random, repair_k, unrepair_random, unrepair_k]
labels = [
    "repair_random",
    "repair_k",
    "unrepair_random",
    "unrepair_k",
]
# 绘制多个数据系列的折线图
for i, y in enumerate(datas[0:2]):
    plt.plot(x, y, label=labels[i], marker="o")
    for xi, yi in zip(x, y):
        plt.text(xi, yi, str(round(yi, 2)), ha="center", va="bottom")  # 保留两位小数
# 添加标题和标签
plt.title("Du62")
plt.xlabel("Index")
plt.ylabel("Fitness")

# 显示图例
plt.legend()

# 显示图形
plt.show()