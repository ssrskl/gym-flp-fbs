# 用于生成CSV文件，并读取CSV文件
import csv
import ast
import pandas as pd
from FbsEnv.utils import FBSUtil

instance = "Du62"
fitness = "unknown"
array = array = [
    [45, 39, 6],
    [26, 35, 22, 4, 41],
    [33, 30, 60, 50, 36, 23, 13],
    [19, 51, 8, 18, 32, 20, 25, 53],
    [34, 55, 5, 24, 56, 21, 16, 3, 43, 29, 49, 7],
    [31, 47, 12, 59, 38, 61, 48],
    [9, 52, 28, 57, 11, 1, 44, 14],
    [15, 37, 27, 2, 40, 42, 10],
    [46, 58, 54],
    [17, 62],
]

# 生成CSV文件
df = pd.DataFrame({"permutation": [sub_list for sub_list in array]})
path = rf"E:\Codes\pythons\gym-flp-fbs\Files\SolutionSet\{instance}-{fitness}.csv"
df.to_csv(path, index=False)
