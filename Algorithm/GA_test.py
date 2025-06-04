import torch
import numpy as np
import gym
import FbsEnv
import FbsEnv.utils.FBSUtil as FBSUtil
from FbsEnv.utils.FBSUtil import FBSUtils
from loguru import logger
import datetime
import copy
from stable_baselines3 import DQN
import os
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings("ignore", module="gym")  # 忽略gym的警告
# /Users/maoyan/miniconda3/envs/gym-fbs/bin/python /Users/maoyan/Codes/Python/gym-flp-fbs/Algorithm/GA_test.py --instance O9-maoyan --model_path models/dqn_model_final --population_size 50 --max_generations 100 --crossover_rate 0.8 --mutation_rate 0.2 --rl_guidance True --visualize True --save_result True
def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试RL+GA模型')
    parser.add_argument('--instance', type=str, default='Du62', help='测试实例')
    parser.add_argument('--model_path', type=str, default='models/dqn_model_final', help='DQN模型路径')
    parser.add_argument('--population_size', type=int, default=50, help='种群大小')
    parser.add_argument('--max_generations', type=int, default=100, help='最大迭代次数')
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='交叉概率')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='变异概率')
    parser.add_argument('--rl_guidance', type=bool, default=True, help='是否使用RL指导')
    parser.add_argument('--visualize', type=bool, default=True, help='是否可视化结果')
    parser.add_argument('--save_result', type=bool, default=True, help='是否保存结果')
    
    return parser.parse_args()

class GATest:
    def __init__(
        self,
        env,
        model_path,
        population_size=30,
        crossover_rate=0.8,
        mutation_rate=0.2,
        max_generations=100,
        rl_guidance=True,
        device="auto"
    ):
        """
        初始化测试环境
        :param env: FBS环境对象
        :param model_path: DQN模型路径
        :param population_size: 种群大小
        :param crossover_rate: 交叉概率
        :param mutation_rate: 变异概率
        :param max_generations: 最大迭代次数
        :param rl_guidance: 是否使用RL指导变异
        :param device: 测试设备
        """
        self.env = env
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.rl_guidance = rl_guidance
        self.device = device
        
        # 加载预训练的DQN模型
        if os.path.exists(model_path):
            logger.info(f"加载DQN模型从: {model_path}")
            self.dqn_model = DQN.load(model_path, env=env)
            logger.info("DQN模型加载成功")
        else:
            logger.error(f"模型路径不存在: {model_path}")
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        # 初始化种群
        self.population = self.initialize_population()
        
        # 记录最佳适应度历史
        self.fitness_history = []
        self.best_fitness_history = []
        
    def initialize_population(self):
        """初始化种群，生成随机解"""
        population = []
        for _ in range(self.population_size):
            individual = gym.make("FbsEnv-v0", instance=self.env.instance)
            individual.reset()
            population.append(individual)
        return population
    
    def evaluate_fitness(self, individual):
        """评估个体的适应度（MHC）"""
        return individual.fitness
    
    def select(self):
        """选择操作：使用锦标赛选择法"""
        tournament_size = 5
        tournament = np.random.choice(self.population, tournament_size, replace=False)
        best_in_tournament = min(tournament, key=lambda ind: self.evaluate_fitness(ind))
        return best_in_tournament
    
    def crossover(self, parent1, parent2):
        """交叉操作：对permutation使用OX交叉，对bay使用单点交叉"""
        if np.random.random() < self.crossover_rate:
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
        """使用预训练的RL模型指导变异"""
        if np.random.random() < self.mutation_rate:
            try:
                # 获取原始状态
                original_state = individual.state
                
                # 使用DQN模型预测最佳动作
                action, _ = self.dqn_model.predict(original_state, deterministic=True)
                
                # 执行变异操作
                new_state, reward, done, info = individual.step(action)
                
                # 如果动作后适应度变差，尝试其他动作
                if reward < 0 and not done:
                    # 尝试随机动作
                    action = np.random.randint(0, 3)
                    individual.step(action)
                
            except Exception as e:
                logger.error(f"RL变异操作发生错误: {e}")
                # 失败时回退到普通变异
                individual = self.random_mutate(individual)
        
        return individual
    
    def random_mutate(self, individual):
        """随机变异操作"""
        if np.random.random() < self.mutation_rate:
            action = np.random.randint(0, 3)  # 随机选择变异类型
            individual.step(action)  # 执行变异操作
        return individual
    
    def mutate(self, individual):
        """根据配置选择变异方法"""
        if self.rl_guidance:
            return self.rl_mutate(individual)
        else:
            return self.random_mutate(individual)
    
    def local_optimize(self, individual):
        """对个体进行局部优化"""
        env = gym.make("FbsEnv-v0", instance=self.env.instance)
        env.reset(fbs_model=individual.fbs_model)
        fac_list = FBSUtil.permutationToArray(env.fbs_model.permutation, env.fbs_model.bay)
        bay_index = np.random.choice(len(fac_list))
        logger.debug("进行shuffle优化")
        permutation, bay = FBSUtil.shuffleOptimization(env, bay_index)
        optimized_fbs_model = FBSUtil.FBSModel(permutation, bay)
        optimized_individual = gym.make("FbsEnv-v0", instance=self.env.instance)
        optimized_individual.reset(fbs_model=optimized_fbs_model)
        return optimized_individual

    def run(self):
        """运行测试算法"""
        start_time = datetime.datetime.now()
        best_fitness = float("inf")
        best_solution = None
        best_generation = 0
        best_time = None
        
        logger.info(f"开始测试GA+RL算法, 使用{'RL指导' if self.rl_guidance else '随机'}变异")
        
        for generation in range(self.max_generations):
            # 计算并记录当前种群的适应度
            fitness_values = [self.evaluate_fitness(ind) for ind in self.population]
            avg_fitness = np.mean(fitness_values)
            self.fitness_history.append(avg_fitness)
            
            # 选择精英个体
            elite_indices = np.argsort(fitness_values)[:int(self.population_size * 0.1)]
            elite_population = [self.population[idx] for idx in elite_indices]
            new_population = elite_population[:]  # 保留精英个体
            
            # 产生新一代种群
            while len(new_population) < self.population_size:
                parent1 = self.select()
                parent2 = self.select()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            # 更新种群
            self.population = new_population[:self.population_size]
            
            # 对最佳个体进行局部优化（每10代执行一次）
            if generation % 10 == 0:
                best_individual = min(self.population, key=lambda ind: self.evaluate_fitness(ind))
                optimized_individual = self.local_optimize(best_individual)
                
                # 如果优化后更好，替换原来的个体
                if self.evaluate_fitness(optimized_individual) < self.evaluate_fitness(best_individual):
                    worst_index = np.argmax(fitness_values)
                    self.population[worst_index] = optimized_individual
            
            # 评估当前种群最佳解
            current_best = min(self.population, key=lambda ind: self.evaluate_fitness(ind))
            current_best_fitness = self.evaluate_fitness(current_best)
            self.best_fitness_history.append(current_best_fitness)
            
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = copy.deepcopy(current_best.fbs_model)
                best_generation = generation
                best_time = (datetime.datetime.now() - start_time).total_seconds()
            
            if generation % 10 == 0:
                logger.info(f"Generation {generation}, Best Fitness: {best_fitness}, Avg Fitness: {avg_fitness:.2f}")
        
        end_time = datetime.datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        logger.info(f"测试完成, 最佳适应度: {best_fitness}, 在第 {best_generation} 代获得")
        logger.info(f"总测试时间: {total_time:.2f} 秒, 获得最佳解用时: {best_time:.2f} 秒")
        
        test_results = {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'best_generation': best_generation,
            'start_time': start_time,
            'end_time': end_time,
            'best_time': best_time,
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history
        }
        
        return test_results
    
    def visualize_results(self, results):
        """可视化测试结果"""
        plt.figure(figsize=(12, 6))
        
        # 绘制适应度历史曲线
        plt.subplot(1, 2, 1)
        plt.plot(results['fitness_history'], label='平均适应度')
        plt.plot(results['best_fitness_history'], label='最佳适应度')
        plt.xlabel('代数')
        plt.ylabel('适应度值 (MHC)')
        plt.title('适应度进化曲线')
        plt.legend()
        plt.grid(True)
        
        # 绘制最终布局
        plt.subplot(1, 2, 2)
        env_vis = gym.make("FbsEnv-v0", instance=self.env.instance)
        env_vis.reset(fbs_model=results['best_solution'])
        env_vis.render(mode='rgb_array', ax=plt.gca())
        plt.title(f'最佳布局 (适应度: {results["best_fitness"]:.2f})')
        
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f'results/GA_RL_test_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
        plt.show()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 检查设备
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"使用设备: {device}")
    
    # 创建测试环境
    env = gym.make("FbsEnv-v0", instance=args.instance)
    
    # 创建并运行测试
    ga_test = GATest(
        env=env,
        model_path=args.model_path,
        population_size=args.population_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        max_generations=args.max_generations,
        rl_guidance=args.rl_guidance,
        device=device
    )
    
    # 运行测试
    results = ga_test.run()
    
    # 可视化结果
    if args.visualize:
        ga_test.visualize_results(results)
    
    # 保存最佳结果
    if args.save_result:
        # 确保目录存在
        os.makedirs('results', exist_ok=True)
        
        # 保存最佳解的环境渲染
        env.reset(fbs_model=results['best_solution'])
        env.render(mode='save', filename=f'results/best_solution_{args.instance}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
        # 打印最佳解的详细信息
        logger.info(f"最佳解布局: {results['best_solution'].array_2d}")
        logger.info(f"最佳适应度: {results['best_fitness']}")

if __name__ == "__main__":
    main()