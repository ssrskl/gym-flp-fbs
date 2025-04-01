# 输出模型的相关信息

from FbsEnv.utils import FBSUtil
import FbsEnv
import gym
from FbsEnv.envs.FBSModel import FBSModel
from loguru import logger

instance_name = "VC10"
env = gym.make("FbsEnv-v0", instance=instance_name)
env.reset()
logger.info(f"模型信息: {env.fac_limit_aspect}")
logger.info(f"模型信息: {env.fbs_model.array_2d}")

env.render()
