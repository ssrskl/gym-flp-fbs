# 在已有的模型的基础上训练
import os
import numpy as np
import FbsEnv
import gym
import FbsEnv.utils.FBSUtil as FBSUtil
from stable_baselines3 import DQN
from loguru import logger

themeName = "基础模型训练"
instance = "O7-maoyan"
current_timesteps = 20000
train_timesteps = 100000
env = gym.make("FbsEnv-v0", instance=instance)
env.reset()

# 如果模型存在，加载已有模型，否则创建新模型
current_path = os.path.dirname(os.path.abspath(__file__))
file_name = themeName + "-DQN-" + instance + "-" + str(current_timesteps)
save_path = os.path.join(
    current_path,
    "..",
    "models",
    file_name,
)

# 检查是否已经有保存的模型
if os.path.exists(save_path + ".zip"):  # 如果模型文件存在
    logger.info(f"加载已有模型：{save_path}")
    model = DQN.load(save_path, env=env)  # 加载已有的模型
    model.learn(total_timesteps=train_timesteps)
    file_name = themeName + "-DQN-" + instance + "-" + str(current_timesteps +
                                                           train_timesteps)
    save_path = os.path.join(
        current_path,
        "..",
        "models",
        file_name,
    )
    model.save(save_path)
else:
    current_timesteps = 0
    logger.info("未找到已有模型，创建新模型进行训练")
    model = DQN("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=train_timesteps)
    file_name = themeName + "-DQN-" + instance + "-" + str(current_timesteps +
                                                           train_timesteps)
    save_path = os.path.join(
        current_path,
        "..",
        "models",
        file_name,
    )
    model.save(save_path)
