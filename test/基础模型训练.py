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
from loguru import logger

themeName = "基础模型训练"
instance = "O9-maoyan"
total_timesteps = 100_000
env = gym.make("FbsEnv-v0", instance=instance)
env.reset()
env.render()
logger.info(f"当前解：{env.fbs_model.permutationToArray()}")
model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=total_timesteps)

current_path = os.path.dirname(os.path.abspath(__file__))
# 将当前路径的上一级路径作为保存路径
file_name = themeName + "-DQN-" + instance + "-" + str(total_timesteps)
save_path = os.path.join(
    current_path,
    "..",
    "models",
    file_name,
)
model.save(save_path)

logger.info(f"训练完成后的解：{env.fbs_model.permutationToArray()}")
env.render()
