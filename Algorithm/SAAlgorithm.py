import FbsEnv
import gym
import numpy as np
import os
import logging
import FbsEnv.utils.FBSUtil as FBSUtil
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from FbsEnv.envs.FBSModel import FBSModel
import FbsEnv.utils.ExperimentsUtil as ExperimentsUtil
import datetime
import copy

def simulated_annealing(env, max_iterations=2000, initial_temp=1000.0, alpha=0.99):
    current_temp = initial_temp
    best_solution = None
    best_fitness = np.inf # 初始化最优适应度
    
    env.reset() # 重置环境
    current_solution = copy.deepcopy(env.fbs_model) # 获取当前解（fbs——model是对象，所以需要deepcopy）
    current_fitness = env.fitness # 获取当前适应度
    current_state = env.state # 获取当前状态

    for iteration in range(max_iterations):
        if current_temp <= 0.001: break # 温度过低，结束搜索
        action = np.random.randint(0, 4) # 随机选择动作
        next_state, reward, done, info = env.step(action) # 执行动作
        next_fitness = env.fitness # 获取下一个适应度
        delta_fitness = next_fitness - current_fitness # 计算适应度变化
        if delta_fitness < 0 or np.random.rand() < np.exp(-delta_fitness / current_temp):
            current_solution = copy.deepcopy(env.fbs_model) # 更新当前解
            current_fitness = env.fitness # 更新当前适应度
            if current_fitness < best_fitness:
                best_solution = copy.deepcopy(env.fbs_model) # 更新最优解
                best_fitness = env.fitness # 更新最优适应度
        else:
            env.reset(fbs_model=current_solution) # 重置环境
            current_fitness = env.fitness # 更新当前适应度
        current_temp *= alpha # 降温
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Current Fitness: {current_fitness}, Best Fitness: {best_fitness}")
    return iteration,best_solution, best_fitness # 返回最优解和最优适应度

if __name__ == "__main__":
    for i in range(10):
        exp_start_time = datetime.datetime.now()
        env = gym.make("FbsEnv-v0", instance="AB20-ar3")
        iteration,best_solution, best_fitness = simulated_annealing(env)
        print(f"Best Solution: {best_solution.array_2d}, Best Fitness: {best_fitness}")
        exp_end_time = datetime.datetime.now()
        ExperimentsUtil.save_experiment_result(
            exp_instance="AB20-ar3",
            exp_algorithm="模拟退火算法",
            exp_iterations=iteration,
            exp_solution=best_solution.array_2d,
            exp_fitness=best_fitness,
            exp_start_time=exp_start_time,
            exp_end_time=exp_end_time
        )