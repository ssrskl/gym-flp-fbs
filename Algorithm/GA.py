import torch
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
from stable_baselines3 import DQN
import torch.nn as nn

warnings.filterwarnings("ignore", module="gym")  # 忽略gym的警告

class GeneticAlgorithm:
    def __init__(
        self,
        env,
        population_size=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        max_generations=100,
        dqn_train_freq=4,
        dqn_learning_starts=1000
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
        self.dqn_model = DQN(
            "CnnPolicy",  # 使用CNN策略
            env,
            policy_kwargs=dict(
                net_arch=[256, 256, 128],  # 更深层网络
                activation_fn=nn.ReLU
            ),
            verbose=0,
            learning_rate=1e-3,
            buffer_size=50000,  # 增大缓冲区
            learning_starts=dqn_learning_starts,
            batch_size=64,  # 增大批次大小
            train_freq=dqn_train_freq,
            gradient_steps=8,  # 增加梯度步数
            target_update_interval=1000,  # 调整目标网络更新频率
            exploration_fraction=0.2  # 控制探索率衰减
        )
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
        :return: 选择的个体
        """
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        best_in_tournament = min(tournament, key=lambda ind: self.evaluate_fitness(ind))
        return best_in_tournament

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
            offspring1_fbs_model, offspring2_fbs_model = FBSUtils.CrossoverActions.order_crossover(
                parent1_fbs_model, parent2_fbs_model
            )
            offspring1 = gym.make("FbsEnv-v0", instance=self.env.instance)
            offspring2 = gym.make("FbsEnv-v0", instance=self.env.instance)
            offspring1.reset(fbs_model=offspring1_fbs_model)
            offspring2.reset(fbs_model=offspring2_fbs_model)
            return offspring1, offspring2
        else:
            return parent1, parent2
    def rl_mutate(self, individual):
        if random.random() < self.mutation_rate:
            action,_ = self.dqn_model.predict(individual.state, deterministic=False)  # 训练DQN模型
            original_fitness = individual.fitness  # 记录原始适应度
            original_state = individual.state  # 记录原始状态
            new_state, reward, done, info = individual.step(action)  # 执行变异操作
            self.dqn_model.replay_buffer.add(
                obs=original_state,
                next_obs=new_state,
                action=action,
                reward=reward,
                done=done,
                infos=[info]
            )  # 将新状态添加到回放缓冲区
            if self.dqn_model.num_timesteps > self.dqn_model.learning_starts and self.dqn_model.num_timesteps % 100 == 0:
               self.dqn_model.learn(total_timesteps=200)  # 训练DQN模型
        return individual
    def mutate(self, individual):
        """
        变异操作：
        :param individual: 个体
        :return: 变异后的个体
        """
        if random.random() < self.mutation_rate:
            action  = np.random.randint(0, 3)  # 随机选择变异类型
            individual.step(action)  # 执行变异操作
            return individual
        else:
            return individual
        

    # TODO 修复局部优化问题
    def local_optimize(self, individual):
        """
        对个体进行局部优化
        :param individual: 需要优化的个体
        :return: 优化后的个体
        """
        env = gym.make("FbsEnv-v0", instance=self.env.instance)
        env.reset(fbs_model=individual.fbs_model)
        fac_list = FBSUtil.permutationToArray(env.fbs_model.permutation, env.fbs_model.bay)
        bay_index = np.random.choice(len(fac_list))
        logger.info("进行shuffle优化")
        permutation, bay = FBSUtil.shuffleOptimization(env, bay_index)
        optimized_fbs_model = FBSUtil.FBSModel(permutation, bay)
        optimized_individual = gym.make("FbsEnv-v0", instance=self.env.instance)
        optimized_individual.reset(fbs_model=optimized_fbs_model)
        return optimized_individual

    def run(self):
        """
        运行遗传算法，添加对优秀解的局部优化
        :return: 最优解、最优适应度、开始时间、结束时间、最优解发现时间
        """
        start_time = datetime.datetime.now()
        best_fitness = float("inf")
        best_solution = None

        for generation in range(self.max_generations):
            fitness_values = [self.evaluate_fitness(ind) for ind in self.population]  # 计算种群适应度值
            elite_indices = np.argsort(fitness_values)[: int(self.population_size * 0.1)]  # 选择精英个体
            elite_population = [self.population[idx] for idx in elite_indices]
            new_population = elite_population[:]  # 保留精英个体

            while len(new_population) < self.population_size:
                parent1 = self.select()
                parent2 = self.select()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.rl_mutate(child1)
                child2 = self.rl_mutate(child2)
                new_population.extend([child1, child2])
            self.population = new_population[: self.population_size]  # 更新种群

            # 对最佳个体进行局部优化
            best_individual = min(self.population, key=lambda ind: self.evaluate_fitness(ind))
            # optimized_individual = self.local_optimize(best_individual)
            # # 替换种群中最差个体
            # worst_index = np.argmax(fitness_values)
            # self.population[worst_index] = optimized_individual

            # 评估当前种群最佳解
            current_best = min(self.population, key=lambda ind: self.evaluate_fitness(ind))
            current_best_fitness = self.evaluate_fitness(current_best)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = copy.deepcopy(current_best.fbs_model)
                fast_time = datetime.datetime.now()
            if generation % 10 == 0:
                logger.info(f"Generation {generation}, Best Fitness: {best_fitness}")

        end_time = datetime.datetime.now()
        return best_solution, best_fitness, start_time, end_time, fast_time


if __name__ == "__main__":
    device = (
    "mps" if torch.backends.mps.is_available() else "cpu"
    )  # 检查是否有可用的MPS设备
    logger.info(f"使用设备: {device}")
    # 实验参数
    exp_instance = "Du62"
    exp_algorithm = "RL+遗传算法"
    exp_remark = "包含修复动作算子-随机初始解"
    exp_number = 30  # 运行次数
    is_exp = False  # 是否进行实验
    # 算法参数
    population_size = 50
    crossover_rate = 0.8
    mutation_rate = 0.1
    max_generations = 62 * 10  # 最大迭代次数
    env = gym.make("FbsEnv-v0", instance=exp_instance)
    ga = GeneticAlgorithm(
        env=env,
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        max_generations=max_generations,
    )
    if is_exp:
        for i in range(exp_number):
            logger.info(f"第{i+1}次实验")
            env = gym.make("FbsEnv-v0", instance=exp_instance)
            ga = GeneticAlgorithm(
                env=env,
                population_size=population_size,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                max_generations=max_generations,                  
            )
            best_solution, best_fitness, exp_start_time, exp_end_time, exp_fast_time = ga.run()
            print(f"Best Solution: {best_solution.array_2d}, Best Fitness: {best_fitness}")
            ExperimentsUtil.save_experiment_result(
                exp_instance=exp_instance,
                exp_algorithm=exp_algorithm,
                exp_iterations=max_generations,
                exp_solution=best_solution.array_2d,
                exp_fitness=best_fitness,
                exp_start_time=exp_start_time,
                exp_fast_time=exp_fast_time,
                exp_end_time=exp_end_time,
                exp_remark=exp_remark
            )
    else:
        best_solution, best_fitness, exp_start_time, exp_end_time, exp_fast_time = ga.run()
        logger.info(f"Best Solution: {best_solution.array_2d}, Best Fitness: {best_fitness}")
        env.reset(fbs_model=best_solution)
        env.render()