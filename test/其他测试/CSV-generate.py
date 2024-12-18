# 用于生成CSV文件，并读取CSV文件
import csv
import ast
import pandas as pd
from FbsEnv.utils import FBSUtil

instance = "Du62"
fitness = "unknown"
array = array = [
    [6, 41, 13, 10, 7],
    [39, 22, 51, 25, 49, 48],
    [4, 36, 20, 42, 53],
    [45, 23, 35, 3, 56, 21, 38, 12, 28, 1, 61, 58, 62],
    [26, 34, 50, 60, 32, 16, 11, 2, 57, 43, 27, 44, 54],
    [33, 8, 30, 18, 5, 59, 24, 52, 29, 47, 14],
    [17, 9, 19, 31, 55, 40, 37, 46, 15],
]

# 生成CSV文件
df = pd.DataFrame({"permutation": [sub_list for sub_list in array]})
path = rf"E:\Codes\pythons\gym-flp-fbs\Files\SolutionSet\{instance}-{fitness}.csv"
df.to_csv(path, index=False)
