# 模拟退火算法
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

def simulated_annealing(env, max_iterations=10000, initial_temp=10000.0, alpha=0.995):
    start_time = datetime.datetime.now()
    current_temp = initial_temp
    best_solution = None
    best_fitness = np.inf # 初始化最优适应度
    
    env.reset() # 重置环境
    current_solution = copy.deepcopy(env.fbs_model) # 获取当前解（fbs——model是对象，所以需要deepcopy）
    current_fitness = env.fitness # 获取当前适应度
    current_state = env.state # 获取当前状态

    for iteration in range(max_iterations):
        if current_temp <= 1e-8: break # 温度过低，结束搜索
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
                fast_time = datetime.datetime.now()
        else:
            env.reset(fbs_model=current_solution) # 重置环境
            current_fitness = env.fitness # 更新当前适应度
        current_temp *= alpha # 降温
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Current Fitness: {current_fitness}, Best Fitness: {best_fitness}")
    end_time = datetime.datetime.now()
    return iteration,best_solution, best_fitness,start_time,end_time,fast_time # 返回最优解和最优适应度

if __name__ == "__main__":
    # 实验参数
    exp_instance = "VC10"
    exp_algorithm = "模拟退火算法"
    exp_remark = "包含修复动作算子-K分初始解"
    exp_number = 30
    is_exp = False
    # 算法参数
    max_iterations = 10000
    initial_temp = 10000.0
    alpha = 0.995
    if is_exp:
        for i in range(exp_number):
            logger.info(f"第{i+1}次实验")
            env = gym.make("FbsEnv-v0", instance=exp_instance)
            iteration,best_solution, best_fitness,exp_start_time,exp_end_time,exp_fast_time = simulated_annealing(env,max_iterations=max_iterations, initial_temp=initial_temp, alpha=alpha)
            print(f"Best Solution: {best_solution.array_2d}, Best Fitness: {best_fitness}")
            ExperimentsUtil.save_experiment_result(
                exp_instance=exp_instance,
                exp_algorithm=exp_algorithm,
                exp_iterations=iteration,
                exp_solution=best_solution.array_2d,
                exp_fitness=best_fitness,
                exp_start_time=exp_start_time,
                exp_fast_time=exp_fast_time,
                exp_end_time=exp_end_time,
                exp_remark=exp_remark
            )
    else:
        env = gym.make("FbsEnv-v0", instance=exp_instance)
        iteration,best_solution, best_fitness,exp_start_time,exp_end_time,exp_fast_time = simulated_annealing(env,max_iterations=max_iterations, initial_temp=initial_temp, alpha=alpha)
        print(f"Best Solution: {best_solution.array_2d}, Best Fitness: {best_fitness}")
        env.reset(fbs_model=best_solution)
        env.render()