# 环境注册测试
import numpy as np
import FbsEnv
import gym
from FbsEnv.envs.FBSModel import FBSModel
from FbsEnv.utils.FBSUtil import FBSUtils
from loguru import logger

env = gym.make("FbsEnv-v0", instance="O7-maoyan")

permutation = [3, 5, 7, 1, 4, 6, 2]
bay = [0, 0, 1, 0, 0, 0, 1]
fbs_model = FBSModel(permutation, bay)

FBSUtils.MutateActions.facility_swap(fbs_model)

env.reset(fbs_model=fbs_model)

logger.info(f"当前环境：{env}")
logger.info(f"当前环境fac_b：\n{env.fac_b}")
logger.info(f"当前环境fac_h：\n{env.fac_h}")
logger.info(f"当前环境fac_aspect_ratio：{env.fac_b / env.fac_h}")
env.render()
