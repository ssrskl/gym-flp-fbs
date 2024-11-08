# 完成遗传算法类以及测试脚本

import gym
import numpy as np
from FbsEnv.envs.FBSModel import FBSModel
import FbsEnv
from FbsEnv.utils import FBSUtil


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
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.instance = instance
        self.env = gym.make("FbsEnv-v0", instance=instance)
        self.population = self._initialize_population()

    def _initialize_population(self) -> list[FBSModel]:
        """初始化种群，生成初始的FBSModel对象列表"""
        population = []
        for _ in range(self.population_size):
            permutation, bay = FBSUtil.random_solution_generator(self.env.n)
            model = FBSModel(permutation, bay)
            population.append(model)
        return population

    def _evaluate_fitness(self, model):
        """计算个体的适应度值"""
        self.env.reset(fbs_model=model)
        return self.env.fitness

    def _select_parents(self):
        """使用轮盘赌选择两个父代个体"""
        fitness_values = np.array(
            [self._evaluate_fitness(ind) for ind in self.population]
        )
        selection_probabilities = fitness_values / fitness_values.sum()
        parents = np.random.choice(self.population, size=2, p=selection_probabilities)
        return parents[0], parents[1]

    def _crossover(self, parent1: FBSModel, parent2: FBSModel):
        """执行交叉操作，生成两个子代个体"""
        if np.random.rand() < self.crossover_rate:
            offspring1_perm, offspring1_bay, offspring2_perm, offspring2_bay = (
                FBSUtil.orderCrossover(parent1, parent2)
            )
            offspring1 = FBSModel(offspring1_perm, offspring1_bay)
            offspring2 = FBSModel(offspring2_perm, offspring2_bay)
            return offspring1, offspring2
        return parent1, parent2

    def _mutate(self, fbs_model: FBSModel):
        """对个体进行变异操作"""
        if np.random.rand() < self.mutation_rate:
            fbs_model.permutation, fbs_model.bay = FBSUtil.facility_swap(
                fbs_model.permutation, fbs_model.bay
            )
        return fbs_model

    def _evolve(self):
        """执行单次遗传算法的迭代，更新种群"""
        new_population = []
        while len(new_population) < self.population_size:
            # 选择父代
            parent1, parent2 = self._select_parents()
            # 交叉生成子代
            offspring1, offspring2 = self._crossover(parent1, parent2)
            # 变异操作
            offspring1 = self._mutate(offspring1)
            offspring2 = self._mutate(offspring2)
            # 将子代加入新种群
            new_population.extend([offspring1, offspring2])
        # 更新种群
        self.population = new_population[: self.population_size]

    def run(self):
        """运行遗传算法，进行多次迭代，寻找最优解"""
        best_solution = None
        best_fitness = float("inf")
        for generation in range(self.max_generations):
            # 执行一次种群进化
            self._evolve()
            # 评估当前种群中的最优解
            for model in self.population:
                fitness = self._evaluate_fitness(model)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = model
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
        return best_solution


# 测试脚本
if __name__ == "__main__":
    instance = 0  # 假设使用第一个实例
    ga = GAAlgorithm(
        population_size=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        max_generations=100,
        instance=instance,
    )
    best_solution = ga.run()
    print("Best Solution:")
    print(f"Permutation: {best_solution.permutation}")
    print(f"Bay: {best_solution.bay}")
