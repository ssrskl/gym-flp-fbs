# 用于生成CSV文件，并读取CSV文件
import csv
import ast
import pandas as pd
from FbsEnv.utils import FBSUtil

instance = "SC35-maoyan"
fitness = "unknown"
array = [
[27,21,7,8,9,11,31,16],
[24,2,6,5,12,32,14],
[23,1,25,30,10,13],
[19,29,17,15],
[28,33,18],
[35,20,34,22,4,26],
[3]
]

# 生成CSV文件
df = pd.DataFrame({"permutation": [sub_list for sub_list in array]})
# path = rf"E:\Codes\pythons\gym-flp-fbs\Files\SolutionSet\{instance}-{fitness}.csv"
path = rf"/Users/maoyan/Codes/Python/gym-flp-fbs/Files/SolutionSet/{instance}-{fitness}.csv"
df.to_csv(path, index=False)
