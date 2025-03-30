# 用于生成CSV文件，并读取CSV文件
import csv
import ast
import pandas as pd
from FbsEnv.utils import FBSUtil

instance = "O7-maoyan"
fitness = "unknown"
array = [
[3,7,5],
[6,2,1,4]
]

# 生成CSV文件
df = pd.DataFrame({"permutation": [sub_list for sub_list in array]})
# path = rf"E:\Codes\pythons\gym-flp-fbs\Files\SolutionSet\{instance}-{fitness}.csv"
path = rf"/Users/maoyan/Codes/Python/gym-flp-fbs/Files/SolutionSet/{instance}-{fitness}.csv"
df.to_csv(path, index=False)
