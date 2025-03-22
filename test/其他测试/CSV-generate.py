# 用于生成CSV文件，并读取CSV文件
import csv
import ast
import pandas as pd
from FbsEnv.utils import FBSUtil

instance = "AB20-ar3"
fitness = "4566.38"
array = [
[18,13,20],
[6,2,1,4,5],
[14,10,8,9,7],
[12,19,15,3],
[16,11,17]
]

# 生成CSV文件
df = pd.DataFrame({"permutation": [sub_list for sub_list in array]})
# path = rf"E:\Codes\pythons\gym-flp-fbs\Files\SolutionSet\{instance}-{fitness}.csv"
path = rf"/Users/maoyan/Codes/Python/gym-flp-fbs/Files/SolutionSet/{instance}-{fitness}.csv"
df.to_csv(path, index=False)
