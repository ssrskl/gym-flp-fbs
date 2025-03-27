# 基础模型测试
"""
使用训练好的模型进行测试
"""
import os
import numpy as np
import FbsEnv
import gym
from FbsEnv.envs.FBSModel import FBSModel
import FbsEnv.utils.FBSUtil as FBSUtil
import logging
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from loguru import logger

themeName = "基础模型训练"
instance = "SC35-maoyan"
algorithm = "PPO"
policy = "MlpPolicy"
total_timesteps = 10_000


env = gym.make("FbsEnv-v0", instance=instance)
obs = env.reset()
current_path = os.path.dirname(os.path.abspath(__file__))
# 将当前路径的上一级路径作为保存路径
# model = DQN.load(
#     f"/Users/maoyan/Codes/Python/gym-flp-fbs/models/基础模型训练-DQN-{instance}-{total_timesteps}"
# )
model = PPO.load(
    f"/Users/maoyan/Codes/Python/gym-flp-fbs/models/基础模型训练-{algorithm}-{policy}-{instance}-{total_timesteps}"
)

max_steps = 10000
current_step = 0
best_fitness = np.inf
best_solution = FBSModel([], [])
while current_step < max_steps:
    current_step += 1
    action, _ = model.predict(obs)
    logger.info(f"当前动作: {action}")
    obs, reward, done, info = env.step(action)
    if env.fitness < best_fitness:
        best_fitness = env.fitness
        logging.info(f"当前适应度: {env.fitness}")
        logging.info(f"当前解: {env.fbs_model.permutation}, {env.fbs_model.bay}")
        best_solution = FBSModel(env.fbs_model.permutation,env.fbs_model.bay)
        logging.info(
            f"第{current_step}步，当前最佳适应度为{best_fitness}, 当前解为{best_solution.permutation}, {best_solution.bay}"
        )

logging.info(
    f"Best fitness: {best_fitness}, Best solution: {best_solution.permutation}, {best_solution.bay}"
)
env.reset(fbs_model=best_solution)
env.render()
