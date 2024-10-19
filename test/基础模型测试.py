# 基础模型测试
"""
使用训练好的模型进行测试
"""
import os
import numpy as np
import FbsEnv
import gym
import FbsEnv.utils.FBSUtil as FBSUtil
import logging
from stable_baselines3 import DQN

env = gym.make("FbsEnv-v0", instance="O7-maoyan")
obs = env.reset()
current_path = os.path.dirname(os.path.abspath(__file__))
# 将当前路径的上一级路径作为保存路径
save_path = os.path.join(
    current_path, "..", "models", "基础训练模型-DQN-O7-maoyan-10000"
)
model = DQN.load(save_path)

max_steps = 10000
current_step = 0
best_fitness = np.inf
best_bay = None
best_permutation = None
while current_step < max_steps:
    current_step += 1
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if env.Fitness < best_fitness:
        best_fitness = env.Fitness.copy()
        best_bay = env.bay.copy()
        best_permutation = env.permutation.copy()
        logging.info(
            f"第{current_step}步，当前最佳适应度为{best_fitness}, 当前bay为{best_bay}, 当前设施排列为{best_permutation}"
        )

logging.info(
    f"Best fitness: {best_fitness}, Best bay: {best_bay}, Best permutation: {best_permutation}"
)
env.reset(layout=(best_permutation, best_bay))
env.render()
