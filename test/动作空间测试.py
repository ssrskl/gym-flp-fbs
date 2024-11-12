# 动作空间测试
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DQN
import FbsEnv.utils.FBSUtil as FBSUtil


instance = "O9-maoyan"
env = gym.make("FbsEnv-v0", instance=instance)
env.reset()
init_permutation = env.fbs_model.permutation
init_bay = env.fbs_model.bay
permutation = init_permutation
bay = init_bay
fac_list_array = FBSUtil.permutationToArray(permutation, bay)
fac_list = [list_item.tolist() for list_item in fac_list_array]
print(f"初始的排列：{init_permutation}，初始的区带：{init_bay}，设施布局为：{fac_list}")
env.render()
# --------------------------------------------执行设施交换--------------------------------------------
print(f"执行设施交换")
permutation, bay = FBSUtil.facility_swap(env.fbs_model.permutation, env.fbs_model.bay)
env.reset(fbs_model=(permutation, bay))
fac_list_array = FBSUtil.permutationToArray(permutation, bay)
fac_list = [list_item.tolist() for list_item in fac_list_array]
env.render()
print(f"变动后的排列：{permutation}，变动后的区带：{bay}，设施布局为：{fac_list}")
env.reset(layout=(init_permutation, init_bay))  # 将布局恢复初始化
# --------------------------------------------执行区带反转--------------------------------------------
print(f"执行区带反转")
bay = FBSUtil.bay_flip(env.bay)
env.reset(fbs_model=(permutation, bay))
fac_list_array = FBSUtil.permutationToArray(permutation, bay)
fac_list = [list_item.tolist() for list_item in fac_list_array]
env.render()
print(f"变动后的排列：{permutation}，变动后的区带：{bay}，设施布局为：{fac_list}")
env.reset(layout=(init_permutation, init_bay))  # 将布局恢复初始化
# --------------------------------------------执行区带交换--------------------------------------------
print(f"执行区带交换")
permutation, bay = FBSUtil.bay_swap(env.fbs_model.permutation, env.fbs_model.bay)
env.reset(fbs_model=(permutation, bay))
fac_list_array = FBSUtil.permutationToArray(permutation, bay)
fac_list = [list_item.tolist() for list_item in fac_list_array]
env.render()
print(f"变动后的排列：{permutation}，变动后的区带：{bay}，设施布局为：{fac_list}")
env.reset(fbs_model=(init_permutation, init_bay))  # 将布局恢复初始化
# --------------------------------------------执行区带交换--------------------------------------------

env.close()
