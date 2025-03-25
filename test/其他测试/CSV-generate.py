# 用于生成CSV文件，并读取CSV文件
import csv
import ast
import pandas as pd
from FbsEnv.utils import FBSUtil

instance = "O9-maoyan"
fitness = "unknown"
array = [
[7,8],
[4,1,2],
[3,6,9,5]
]

# 生成CSV文件
df = pd.DataFrame({"permutation": [sub_list for sub_list in array]})
# path = rf"E:\Codes\pythons\gym-flp-fbs\Files\SolutionSet\{instance}-{fitness}.csv"
path = rf"/Users/maoyan/Codes/Python/gym-flp-fbs/Files/SolutionSet/{instance}-{fitness}.csv"
df.to_csv(path, index=False)
