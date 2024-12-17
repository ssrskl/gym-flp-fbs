# 用于生成CSV文件，并读取CSV文件
import csv
import ast
import pandas as pd
from FbsEnv.utils import FBSUtil

permutation = [3, 5, 7, 1, 4, 6, 2]
bay = [0, 0, 1, 0, 0, 0, 1]
array = FBSUtil.permutationToArray(permutation, bay)
array = [
    [1, 18, 5],
    [20, 8, 7, 6],
    [2, 4, 19, 3],
    [10, 14, 9, 15],
    [12, 17, 13],
    [16, 11],
]
# print(array)
df = pd.DataFrame({"permutation": [sub_list for sub_list in array]})
df.to_csv(r"E:\Codes\pythons\gym-flp-fbs\Files\SolutionSet\AB20-ar3.csv", index=False)

# 读取数据
df_read = pd.read_csv(r"E:\Codes\pythons\gym-flp-fbs\Files\SolutionSet\AB20-ar3.csv")
result_list = df_read["permutation"].tolist()
new_result_list = [ast.literal_eval(item) for item in result_list]
print(new_result_list)