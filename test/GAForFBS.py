# 强化学习与遗传算法结合训练模型
import gym
import FbsEnv
import numpy as np
from FbsEnv.envs.FBSModel import FBSModel
import FbsEnv.utils.FBSUtil as FBSUtil
import logging
from stable_baselines3 import DQN


class GAAlgorithm:
    def __init__(
        self,
        instance,
        population_size=50,
        mutation_rate=0.1,
        crossover_rate=0.8,
        max_generations=100,
    ):
        self.instance = instance
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
            fitness = env.fitness
            population_fitness.append(fitness)
        return population_fitness

    def selection(self):
        # 使用轮盘赌选择
        total_fitness = sum(self.population_fitness)
        selection_probs = [
            fitness / total_fitness for fitness in self.population_fitness
        ]
        selected_indices = np.random.choice(
            range(self.population_size), size=2, p=selection_probs
        )
        return (
            self.population[selected_indices[0]],
            self.population[selected_indices[1]],
        )

    def crossover(self, parent1, parent2):
        (
            offspring1_permutation,
            offspring1_bay,
            offspring2_permutation,
            offspring2_bay,
        ) = FBSUtil.orderCrossover(parent1.fbs_model, parent2.fbs_model)
        offspring1 = gym.make("FbsEnv-v0", instance=self.instance)
        offspring1_fbs_model = FBSModel(
            permutation=offspring1_permutation, bay=offspring1_bay
        )
        offspring1.reset(fbs_model=offspring1_fbs_model)
        offspring2 = gym.make("FbsEnv-v0", instance=self.instance)
        offspring2_fbs_model = FBSModel(
            permutation=offspring2_permutation, bay=offspring2_bay
        )
        offspring2.reset(fbs_model=offspring2_fbs_model)
        return offspring1, offspring2

    def mutation(self, individual):
        old_state = individual.state.copy()
        action, _ = self.model.predict(old_state)
        new_state, reward, done, info = individual.step(action)
        self.model.replay_buffer.add(
            old_state, new_state, action, reward, done, [info]
        )  # 将旧状态、新状态、动作、奖励、是否结束、信息添加到经验回放池中
        return individual

    def train_model(self):  # 训练模型
        if self.model.replay_buffer.size() > self.model.batch_size:
            logging.info(f"开始训练模型")
            self.model.train(batch_size=self.model.batch_size, gradient_steps=1)
            self.model.save(
                f"../models/GA/DQN-FBS_GA-{self.instance}-{self.generation}"
            )

    def train(self):
        self.population = self.initialize_population()  # 初始化种群
        self.population_fitness = self.evaluate_population()  # 初始化种群适应度
        self.model.learn(total_timesteps=10000, reset_num_timesteps=False)  # 初始化训练
        logging.info(self.population_fitness)
        no_improvement_generations = 0
        best_fitness = -np.inf
        for generation in range(self.max_generations):
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.selection()  # 选择两个个体
                offspring1, offspring2 = self.crossover(parent1, parent2)  # 交叉
                offspring1 = self.mutation(offspring1)
                offspring2 = self.mutation(offspring2)
                new_population.extend([offspring1, offspring2])
            self.population = new_population  # 更新种群
            self.population_fitness = self.evaluate_population()  # 重新评估适应度
            current_best_fitness = max(self.population_fitness)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                no_improvement_generations = 0
            else:
                no_improvement_generations += 1
            if no_improvement_generations >= 20:  # 早停机制
                print(f"早停于第{generation}代")
                break
            print(f"第{generation}代最优个体适应度为{best_fitness}")
        best_individual = max(self.population, key=lambda x: x.fitness)
        print(f"最终最优个体适应度为{best_individual.fitness}")


if __name__ == "__main__":
    ga = GAAlgorithm(instance="O7-maoyan")
    ga.train()
