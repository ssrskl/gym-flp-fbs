# 动作空间测试
"""
测试局部优化方法
"""
import gym
import matplotlib.pyplot as plt
import numpy as np
import FbsEnv.utils.FBSUtil as FBSUtil
import logging
from FbsEnv.envs.FBSModel import FBSModel

instance = "AB20-ar3"
env = gym.make("FbsEnv-v0", instance=instance)
env.reset()
env.render()

# permutation, bay = FBSUtil.SingleBayGradualArrangementOptimization(env)
fac_list = FBSUtil.permutationToArray(env.fbs_model.permutation,
                                      env.fbs_model.bay)
bay_index = np.random.choice(len(fac_list))
child_permutation = fac_list[bay_index]
logging.info(f"子区带{child_permutation}的长度为{len(child_permutation)}")
length_child_permutation = len(child_permutation)
if length_child_permutation > 7:
    logging.info("进行shuffle优化")
    permutation, bay = FBSUtil.shuffleOptimization(env, bay_index)
else:
    logging.info("进行单区带全排列优化")
    permutation, bay = FBSUtil.SingleBayGradualArrangementOptimization(
        env, bay_index)
env.reset(fbs_model=FBSModel(permutation, bay))
env.render()
