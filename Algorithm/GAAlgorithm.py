# 完成遗传算法类以及测试脚本

import copy
import os
import random
import gym
import numpy as np
from FbsEnv.envs.FBSModel import FBSModel
import FbsEnv
from FbsEnv.utils import FBSUtil
from FbsEnv.utils.FBSUtil import FBSUtils
import logging
from stable_baselines3 import DQN
import pathlib
from loguru import logger
import FbsEnv.utils.ExperimentsUtil as ExperimentsUtil


class GAAlgorithm:

    def __init__(
            self,
            population_size=50,  # 种群大小
            crossover_rate=0.8,  # 交叉率
            mutation_rate=0.1,  # 变异率
            max_generations=100,  # 最大迭代次数
            instance=None,  # 案例
    ):
        self.population_size = population_size
        self.best_solutions = []
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.instance = instance
        self.env = gym.make("FbsEnv-v0",
                            instance=instance)  # env只是用于提供环境信息以及计算适应度值
        self.model = DQN("MultiInputPolicy", self.env, verbose=1)
        self.stage_steps = 10000
        self.population = self._initialize_population()
        self.mutate_actions = [
            "facility_swap",
            "bay_flip",
            "bay_swap",
            "bay_shuffle",
            "facility_shuffle",
        ]

    def _initialize_population(self) -> list[FBSModel]:
        """初始化种群，生成初始的FBSModel对象列表"""
        population = []
        for _ in range(self.population_size):
            permutation, bay = FBSUtil.random_solution_generator(self.env.n)
            model = FBSModel(permutation, bay)
            population.append(model)
        return population

    def _evaluate_fitness(self, fbs_model: FBSModel) -> float:
        """计算个体的适应度值"""
        fitness_env = copy.deepcopy(self.env)  # 深拷贝环境，防止环境被修改
        fitness_env.reset(fbs_model=fbs_model)
        return fitness_env.fitness

    def _select_parents(self) -> tuple[FBSModel, FBSModel]:
        """使用轮盘赌选择两个父代个体"""
        fitness_values = np.array(
            [self._evaluate_fitness(ind) for ind in self.population])
        selection_probabilities = fitness_values / fitness_values.sum()
        parents = np.random.choice(self.population,
                                   size=2,
                                   p=selection_probabilities)
        return parents[0], parents[1]

    def _crossover(self, parent1: FBSModel,
                   parent2: FBSModel) -> tuple[FBSModel, FBSModel]:
        """执行交叉操作，生成两个子代个体"""
        if np.random.rand() < self.crossover_rate:
            logging.info("执行交叉操作")
            offspring1, offspring2 = FBSUtils.CrossoverActions.order_crossover(
                parent1, parent2)
            return offspring1, offspring2
        return parent1, parent2

    def _mutate(self, fbs_model: FBSModel):
        """对个体进行变异操作"""
        if np.random.rand() < self.mutation_rate:
            actions = [
                getattr(FBSUtil, func) for func in self.mutate_actions
                if callable(getattr(FBSUtil, func))
            ]
            mutate_action = random.choice(actions)
            logging.info(f"执行的动作为：{mutate_action.__name__}")
            # 执行动作
            new_prem, new_bay = mutate_action(fbs_model.permutation,
                                              fbs_model.bay)
            return FBSModel(new_prem, new_bay)
        return fbs_model

    def _evolve(self):
        """执行单次遗传算法的迭代，更新种群"""
        new_population = []
        while len(new_population) < self.population_size:
            parent1, parent2 = self._select_parents()
            offspring1, offspring2 = self._crossover(parent1, parent2)
            offspring1 = self._mutate(offspring1)
            offspring2 = self._mutate(offspring2)
            new_population.extend([offspring1, offspring2])  # 将子代加入新种群
        new_population.sort(key=lambda x: self._evaluate_fitness(x))
        self.population = new_population[:self.population_size]

    def _train_model(self, fbs_model: FBSModel, current_timesteps: int,
                     train_timesteps: int):  # 没有返回值，所以直接更改env
        """训练强化学习模型"""
        # 初始化模型以及环境
        self.env.reset(fbs_model=fbs_model)
        self.model = DQN("MultiInputPolicy", self.env, verbose=1)
        current_path = os.path.dirname(os.path.abspath(__file__))
        file_name = "GA" + "-DQN-" + self.instance + "-" + str(
            current_timesteps)
        save_path = os.path.join(
            current_path,
            "..",
            "models",
            file_name,
        )
        # 检查是否已经有保存的模型
        if os.path.exists(save_path + ".zip"):
            logger.info(f"加载已有模型：{save_path}")
            self.model = DQN.load(save_path, env=self.env)
            self.model.learn(total_timesteps=train_timesteps)
        else:
            logger.info("未找到已有模型，创建新模型进行训练")
            current_timesteps = 0
            self.model.learn(total_timesteps=train_timesteps)

        file_name = "GA" + "-DQN-" + self.instance + "-" + str(
            current_timesteps + train_timesteps)
        save_path = os.path.join(
            current_path,
            "..",
            "models",
            file_name,
        )
        self.model.save(save_path)

    def _model_project(self, fbs_model: FBSModel, total_timesteps: int):
        """将当前最优解进行强化学习优化"""
        logging.info(f"开始进行强化学习优化")
        obs = self.env.reset(fbs_model=fbs_model)
        # 检查是否已经存在模型
        current_path = os.path.dirname(os.path.abspath(__file__))
        file_name = "GA" + "-DQN-" + self.instance + "-" + str(total_timesteps)
        save_path = os.path.join(
            current_path,
            "..",
            "models",
            file_name,
        )
        self.model = DQN.load(save_path, env=self.env)
        max_steps = 10000
        current_step = 0
        best_fitness = np.inf
        best_solution = FBSModel([], [])
        while current_step < max_steps:
            current_step += 1
            action, _ = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            if self.env.fitness < best_fitness:
                best_fitness = self.env.fitness
                best_solution = FBSModel(self.env.fbs_model.permutation,
                                         self.env.fbs_model.bay)
        # 更新环境
        self.env.reset(fbs_model=best_solution)

    def run(self):
        """运行遗传算法，进行多次迭代，寻找最优解"""
        best_solution = None
        best_fitness = float("inf")
        for generation in range(self.max_generations):
            # 执行一次种群进化
            self._evolve()
            # 对当前种群中的最优解进行强化学习优化
            self._train_model(self.population[0],
                              generation * self.stage_steps, self.stage_steps)
            self._model_project(
                self.population[0],
                100_000)
            # 进行局部优化
            fac_list = FBSUtil.permutationToArray(
                self.env.fbs_model.permutation, self.env.fbs_model.bay)
            bay_index = np.random.choice(len(fac_list))
            if len(fac_list[bay_index]) > 7:
                logging.info("进行shuffle优化")
                permutation, bay = FBSUtil.shuffleOptimization(
                    self.env, bay_index)
            else:
                logging.info("进行单区带全排列优化")
                permutation, bay = FBSUtil.SingleBayGradualArrangementOptimization(
                    self.env, bay_index)
            self.env.reset(fbs_model=FBSModel(permutation, bay))
            # self.env.render()
            self.best_solutions.append(FBSModel(permutation, bay))
        return self.best_solutions


# 测试脚本
if __name__ == "__main__":
    instance = "Du62"  # 假设使用第一个实例
    ga = GAAlgorithm(
        population_size=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        max_generations=10,
        instance=instance,
    )
    best_solutions = ga.run()
    env = gym.make("FbsEnv-v0", instance=instance)
    for solution in best_solutions:
        env.reset(fbs_model=solution)
        env.render()
