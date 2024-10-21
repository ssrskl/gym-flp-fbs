# 强化学习与遗传算法结合训练模型
import gym
import FbsEnv
import numpy as np
import FbsEnv.utils.FBSUtil as FBSUtil
import logging
from stable_baselines3 import DQN


class GAAlgorithm:
    def __init__(
        self,
        instance,
        model_path,
        population_size=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_generations=100,
    ):
        self.instance = instance
        # self.model = DQN.load(model_path)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.population = None
        self.population_fitness = None

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            env = gym.make("FbsEnv-v0", instance=self.instance)
            env.reset()
            population.append(env)
        return population

    def evaluate_population(self):
        population_fitness = []
        for env in self.population:
            fitness = env.Fitness
            population_fitness.append(fitness)
        return population_fitness

    def selection(self):
        sorted_population = sorted(
            zip(self.population, self.population_fitness),
            key=lambda x: x[1],
            reverse=False,
        )  # 按照适应度排序
        return sorted_population[:2]  # 返回适应度最高的两个个体

    def crossover(self, parent1, parent2):
        
        pass

    def mutation(self):
        pass

    def train(self):
        self.population = self.initialize_population()  # 初始化种群
        self.population_fitness = self.evaluate_population()  # 初始化种群适应度
        logging.info(self.population_fitness)
        for _ in range(self.max_generations):
            parent1, parent2 = self.selection()
            self.crossover()
            self.mutation()
            self.evaluate_population()


if __name__ == "__main__":
    ga = GAAlgorithm(instance="AB20-ar3", model_path="DQN_AB20-ar3.zip")
    ga.train()
