import numpy as np
import random
import copy
import datetime
import FbsEnv
import gym
import FbsEnv.utils.FBSUtil as FBSUtil
from FbsEnv.utils.FBSUtil import FBSUtils
from loguru import logger
import FbsEnv.utils.ExperimentsUtil as ExperimentsUtil
import warnings

warnings.filterwarnings("ignore",module="gym") # 忽略gym的警告


class GeneticAlgorithm:
    def __init__(
        self,
        env,
        population_size=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        max_generations=100,
    ):
        """
        初始化遗传算法参数
        :param env: FBS环境对象
        :param population_size: 种群大小
        :param crossover_rate: 交叉概率
        :param mutation_rate: 变异概率
        :param max_generations: 最大迭代次数
        """
        self.env = env
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        初始化种群，生成随机解
        :return: 种群列表，每个个体包含设施排列和条带划分
        """
        population = []
        for _ in range(self.population_size):
            individual = gym.make("FbsEnv-v0", instance=self.env.instance)
            individual.reset()
            population.append(individual)
        return population

    def evaluate_fitness(self, individual):
        """
        评估个体的适应度（MHC）
        :param individual: 个体
        :return: 适应度值（越小越好）
        """
        return individual.fitness

    def select(self):
        """
        选择操作：使用锦标赛选择法
        :return: 选择的个体列表
        """
        tournament_size = 5
        selected = []
        tournament = random.sample(self.population, tournament_size)
        best_in_tournament = min(
            tournament, key=lambda ind: self.evaluate_fitness(ind)
        )
        selected.append(best_in_tournament)
        return selected[0]

    def crossover(self, parent1, parent2):
        """
        交叉操作：对permutation使用OX交叉，对bay使用单点交叉
        :param parent1: 父代个体1
        :param parent2: 父代个体2
        :return: 子代个体1和子代个体2
        """
        if random.random() < self.crossover_rate:
            parent1_fbs_model = parent1.fbs_model
            parent2_fbs_model = parent2.fbs_model
            offspring1_fbs_model,offspring2_fbs_model =  FBSUtils.CrossoverActions.order_crossover(parent1_fbs_model,parent2_fbs_model)
            offspring1 = gym.make("FbsEnv-v0", instance=self.env.instance)
            offspring2 = gym.make("FbsEnv-v0", instance=self.env.instance)
            offspring1.reset(fbs_model=offspring1_fbs_model)
            offspring2.reset(fbs_model=offspring2_fbs_model)
            return offspring1,offspring2
        else:
            return parent1,parent2

    def mutate(self, individual):
        """
        变异操作：对permutation交换两个位置，对bay随机翻转一位
        :param individual: 个体
        :return: 变异后的个体
        """
        if random.random() < self.mutation_rate:
            # 交换变异（permutation）
            action = np.random.randint(0, 4)
            individual.step(action)
        return individual

    def run(self):
        """
        运行遗传算法
        :return: 最优解、最优适应度、开始时间、结束时间、最优解发现时间
        """
        start_time = datetime.datetime.now()
        best_fitness = float("inf")
        best_solution = None

        for generation in range(self.max_generations):
            fitness_values = [
                self.evaluate_fitness(ind) for ind in self.population
            ] # 计算种群适应度值
            elite_indices = np.argsort(fitness_values)[: int(
                self.population_size * 0.1
            )] # 选择精英个体
            elite_population = [
                self.population[idx] for idx in elite_indices
            ]
            new_population = elite_population[:] # 保留精英个体
            while len(new_population) < self.population_size:
                parent1 = self.select()
                parent2 = self.select()
                child1,child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = new_population[: self.population_size] # 更新种群
            # 评估当前种群最佳解
            current_best = min(
                self.population, key=lambda ind: self.evaluate_fitness(ind)
            )
            current_best_fitness = self.evaluate_fitness(current_best)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = copy.deepcopy(current_best)
                fast_time = datetime.datetime.now()

            if generation % 10 == 0:
                logger.info(f"Generation {generation}, Best Fitness: {best_fitness}")
        end_time = datetime.datetime.now()
        return best_solution, best_fitness, start_time, end_time, fast_time


if __name__ == "__main__":
    # 实验参数
    exp_instance = "Du62"
    exp_algorithm = "遗传算法"
    exp_remark = "基本遗传算法实现"
    exp_number = 30  # 运行次数
    is_exp = False  # 是否进行实验
    # 算法参数
    population_size = 50
    crossover_rate = 0.8
    mutation_rate = 0.1
    max_generations = 62*10 # 最大迭代次数
    env = gym.make("FbsEnv-v0", instance=exp_instance)
    if is_exp:
        pass
    else:    
        ga = GeneticAlgorithm(
            env=env,
            population_size=50,
            crossover_rate=0.8,
            mutation_rate=0.1,
            max_generations=100,
        )
        best_solution, best_fitness, exp_start_time, exp_end_time, exp_fast_time = ga.run()
        logger.info(f"Best Solution: {best_solution.fbs_model.array_2d}, Best Fitness: {best_fitness}")
        env.reset(fbs_model=best_solution.fbs_model)
        env.render()
