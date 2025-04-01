# 用于生成CSV文件，并读取CSV文件
import csv
import ast
import pandas as pd
from FbsEnv.utils import FBSUtil

instance = "VC10"
fitness = "unknown"
array = [
[1],
[7,6],
[4,2],
[8,10,9],
[5,3]
]

# 生成CSV文件
df = pd.DataFrame({"permutation": [sub_list for sub_list in array]})
# path = rf"E:\Codes\pythons\gym-flp-fbs\Files\SolutionSet\{instance}-{fitness}.csv"
path = rf"/Users/maoyan/Codes/Python/gym-flp-fbs/Files/SolutionSet/{instance}-{fitness}.csv"
df.to_csv(path, index=False)
