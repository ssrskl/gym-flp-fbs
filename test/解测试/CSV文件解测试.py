# 用于生成CSV文件，并读取CSV文件
import csv
import ast
import pandas as pd
from FbsEnv.utils import FBSUtil
import FbsEnv
import gym
from FbsEnv.envs.FBSModel import FBSModel

instance_name = "AB20-ar3"
fitness = "3226.22"
path = rf"/Users/maoyan/Codes/Python/gym-flp-fbs/Files/SolutionSet/{instance_name}-{fitness}.csv"


# 读取数据
df_read = pd.read_csv(path)
result_list = df_read["permutation"].tolist()
array = [ast.literal_eval(item) for item in result_list]
# 加载布局

permutation, bay = FBSUtil.arrayToPermutation(array)
env = gym.make("FbsEnv-v0", instance=instance_name)
env.reset(fbs_model=FBSModel(permutation.tolist(), bay.tolist()))
env.render()
