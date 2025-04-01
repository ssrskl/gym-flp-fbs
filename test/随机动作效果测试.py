# 随机动作效果测试
import gym
import numpy as np
import FbsEnv
from FbsEnv.envs.FBSModel import FBSModel
from loguru import logger

best_fitness = float("inf")
best_solution = None
max_timesteps = 100000

env = gym.make("FbsEnv-v0", instance="O9-maoyan")
env.reset()

for _ in range(max_timesteps):
    action = np.random.randint(0, 4)
    env.step(action)
    if env.fitness < best_fitness:
        best_fitness = env.fitness
        best_solution = FBSModel(env.fbs_model.permutation, env.fbs_model.bay)
        logger.info(
            f"当前适应度: {best_fitness}, 当前解: {best_solution.permutation}, {best_solution.bay}"
        )

print(f"当前最优解：{best_solution.array_2d},当前最优适应度：{best_fitness}")
env.reset(fbs_model=best_solution)
env.render()
