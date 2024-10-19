# 基础模型训练测试
"""
使用最简单传统的方式来训练一个模型
"""
import os
import numpy as np
import FbsEnv
import gym
import FbsEnv.utils.FBSUtil as FBSUtil
from stable_baselines3 import DQN

env = gym.make("FbsEnv-v0", instance="O7-maoyan")
model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

current_path = os.path.dirname(os.path.abspath(__file__))
# 将当前路径的上一级路径作为保存路径
save_path = os.path.join(
    current_path, "..", "models", "基础训练模型-DQN-O7-maoyan-10000"
)
model.save(save_path)
