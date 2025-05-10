# 输出模型的相关信息

from FbsEnv.utils import FBSUtil
import FbsEnv
import gym
from FbsEnv.envs.FBSModel import FBSModel
from loguru import logger

instance_name = "SC35-maoyan"
env = gym.make("FbsEnv-v0", instance=instance_name)
env.reset()
logger.info(f"模型信息: {env.fac_limit_aspect}")
logger.info(f"模型信息: {env.fbs_model.array_2d}")
logger.info(f"设施总面积: {sum(env.areas)}")
logger.info(f"模型信息: {env.H}")
logger.info(f"模型信息: {env.W}")

env.render()
