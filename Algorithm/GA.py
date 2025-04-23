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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

warnings.filterwarnings("ignore", module="gym")  # 忽略gym的警告

# 自定义CNN特征提取器
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        # 获取输入形状 (H, W, C) 格式
        input_h, input_w, n_input_channels = observation_space.shape
        
        # 2D CNN网络 - 移除池化层以支持小尺寸输入
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算CNN输出维度
        with torch.no_grad():
            # 注意：SB3期望输入形状为 (B, C, H, W)
            # 我们的观察空间形状为 (H, W, C)
            # 所以需要转换
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            sample = sample.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            n_flatten = self.cnn(sample).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )
        
    def forward(self, observations):
        # 转换形状：[B, H, W, C] -> [B, C, H, W]
        observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))

class GeneticAlgorithm:
    def __init__(
        self,
        env,
        population_size=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        max_generations=100,
        dqn_train_freq=4,
        dqn_learning_starts=1000,
        device="auto"
    ):
        """
        初始化遗传算法参数
        :param env: FBS环境对象
        :param population_size: 种群大小
        :param crossover_rate: 交叉概率
        :param mutation_rate: 变异概率
        :param max_generations: 最大迭代次数
        :param dqn_train_freq: DQN训练频率
        :param dqn_learning_starts: DQN开始学习的步数
        :param device: 训练设备
        """
        self.env = env
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        
        # 初始化DQN模型，使用MLP策略而非CNN策略
        self.dqn_model = DQN(
            "MlpPolicy",  # 使用MLP策略
            env,
            policy_kwargs=dict(
                net_arch=[256, 256, 128],  # 网络结构
                activation_fn=nn.ReLU
            ),
            verbose=0,
            learning_rate=1e-3,
            buffer_size=50000,  # 缓冲区大小
            learning_starts=dqn_learning_starts,
            batch_size=64,  # 批次大小
            train_freq=dqn_train_freq,
            gradient_steps=8,  # 梯度步数
            target_update_interval=1000,  # 目标网络更新频率
            exploration_fraction=0.2,  # 探索率衰减
            device=device  # 指定训练设备
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
        """
        强化学习引导的变异操作
        使用DQN模型选择最优动作来变异个体
        """
        if random.random() < self.mutation_rate:
            try:
                # 获取原始状态，确保形状正确
                original_state = individual.state
                
                # 使用DQN模型预测动作
                action, _ = self.dqn_model.predict(original_state, deterministic=False)
                
                # 执行变异操作
                new_state, reward, done, info = individual.step(action)
                
                # 记录经验但不添加到回放缓冲区，避免形状不匹配问题
                
                # 直接训练DQN模型
                if (self.dqn_model.num_timesteps > self.dqn_model.learning_starts and 
                    self.dqn_model.num_timesteps % 50 == 0):
                    # 使用随机样本训练，而不是依赖具体的回放缓冲区
                    self.dqn_model.learn(gradient_steps=8, batch_size=32)
                    
            except Exception as e:
                logger.error(f"RL变异操作发生错误: {e}")
                # 失败时回退到普通变异
                individual = self.mutate(individual)
        
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
        best_dqn_model = None
        best_generation = 0
        
        # 启用DQN的在线学习
        self.dqn_model.learn(total_timesteps=0)

        for generation in range(self.max_generations):
            fitness_values = [self.evaluate_fitness(ind) for ind in self.population]  # 计算种群适应度值
            elite_indices = np.argsort(fitness_values)[: int(self.population_size * 0.1)]  # 选择精英个体
            elite_population = [self.population[idx] for idx in elite_indices]
            new_population = elite_population[:]  # 保留精英个体

            while len(new_population) < self.population_size:
                parent1 = self.select()
                parent2 = self.select()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)  # 使用普通变异替代RL变异
                child2 = self.mutate(child2)  # 使用普通变异替代RL变异
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
                best_dqn_model = self.dqn_model
                best_generation = generation
                fast_time = datetime.datetime.now()
                
                # 每当找到更好的解时，保存DQN模型
                model_path = f"models/dqn_model_gen_{generation}_fitness_{best_fitness:.2f}"
                logger.info(f"保存DQN模型到: {model_path}")
                self.dqn_model.save(model_path)
                
            if generation % 10 == 0:
                logger.info(f"Generation {generation}, Best Fitness: {best_fitness}")

        end_time = datetime.datetime.now()
        
        # 最终保存DQN模型
        final_model_path = "models/dqn_model_final"
        logger.info(f"保存最终DQN模型到: {final_model_path}")
        self.dqn_model.save(final_model_path)
        
        # 记录训练相关信息
        logger.info(f"最佳适应度: {best_fitness}, 在第 {best_generation} 代获得")
        logger.info(f"总训练时间: {end_time - start_time}")
        
        return best_solution, best_fitness, start_time, end_time, fast_time

    def constructState(self):
        """
        构建适合MLP处理的1D状态表示
        """
        # 提取设施相关数据
        permutation = np.array(self.env.fbs_model.permutation)
        sources = np.sum(self.env.TM, axis=1)
        sinks = np.sum(self.env.TM, axis=0)
        
        # 归一化为0-1范围
        if np.max(permutation) != np.min(permutation):
            norm_permutation = (permutation - np.min(permutation)) / (np.max(permutation) - np.min(permutation))
        else:
            norm_permutation = np.ones(self.env.n) * 0.5
            
        if np.max(sources) != np.min(sources):
            norm_sources = (sources - np.min(sources)) / (np.max(sources) - np.min(sources))
        else:
            norm_sources = np.ones(self.env.n) * 0.5
            
        if np.max(sinks) != np.min(sinks):
            norm_sinks = (sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks))
        else:
            norm_sinks = np.ones(self.env.n) * 0.5
            
        # 将坐标、尺寸和流量数据合并为一维数组
        state_components = [
            norm_permutation,  # 设施ID
            self.env.fac_x / self.env.W,  # 归一化x坐标
            self.env.fac_y / self.env.H,  # 归一化y坐标
            self.env.fac_b / self.env.W,  # 归一化宽度
            self.env.fac_h / self.env.H,  # 归一化高度
            norm_sources,  # 归一化源流量
            norm_sinks,  # 归一化汇流量
            self.env.fac_aspect_ratio / self.env.fac_limit_aspect  # 归一化横纵比
        ]
        
        # 合并为一维状态向量
        state_vector = np.concatenate(state_components)
        
        return state_vector.astype(np.float32)

if __name__ == "__main__":
    # 检查可用的设备
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"使用设备: {device}")
    
    # 实验参数
    exp_instance = "Du62"
    exp_algorithm = "RL+遗传算法"
    exp_remark = "CNN-包含修复动作算子-K分初始解v3"
    exp_number = 30  # 运行次数
    is_exp = False  # 是否进行实验
    
    # 算法参数
    population_size = 50
    crossover_rate = 0.8
    mutation_rate = 0.1
    max_generations = 62 * 10  # 最大迭代次数
    dqn_train_freq = 4  # DQN训练频率
    dqn_learning_starts = 1000  # DQN开始学习的步数
    
    # 创建环境和GA实例
    env = gym.make("FbsEnv-v0", instance=exp_instance)
    ga = GeneticAlgorithm(
        env=env,
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        max_generations=max_generations,
        dqn_train_freq=dqn_train_freq,
        dqn_learning_starts=dqn_learning_starts,
        device=device
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
                dqn_train_freq=dqn_train_freq,
                dqn_learning_starts=dqn_learning_starts,
                device=device
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