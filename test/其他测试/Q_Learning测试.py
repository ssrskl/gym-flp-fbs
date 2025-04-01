
import FbsEnv
import gym
import FbsEnv.utils.FBSUtil as FBSUtil
from Algorithm.RL.Q_Learning import QLearningAgent, evaluate_policy
from loguru import logger

Max_train_steps = 20000
instance = "O7-maoyan"
env = gym.make("FbsEnv-v0", instance=instance)
env.reset()
agent = QLearningAgent(
    s_dim=env.observation_space.shape[0],
    a_dim=env.action_space.n,
    lr=0.2,
    gamma=0.9,
    exp_noise=0.1
)
total_steps = 0
best_fitness = float("inf")
while total_steps < Max_train_steps:
    state = env.reset()
    done = False
    action = agent.select_action(state,deterministic=False)
    next_state, reward, done, _ = env.step(action)
    agent.train(state, action, reward, next_state, done)
    state = next_state
    total_steps += 1
    if env.fitness < best_fitness:
        best_fitness = env.fitness
        logger.info(
            f"当前适应度: {best_fitness}, 当前解: {env.fbs_model.permutation}, {env.fbs_model.bay}"
        )
logger.info(f"训练结束，总步数: {total_steps}")
logger.info(f"当前最优解：{env.fbs_model.array_2d},当前最优适应度：{best_fitness}")