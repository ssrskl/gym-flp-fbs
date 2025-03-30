# 群体动作空间测试
import gym
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from FbsEnv.utils.FBSUtil import FBSUtils
from FbsEnv.envs.FBSModel import FBSModel

instance = "O9-maoyan"

parent1 = FBSModel([1, 2, 3, 4, 5, 6], [0, 0, 0, 1, 0, 1])
parent2 = FBSModel([6, 5, 4, 3, 2, 1], [0, 0, 1, 0,0, 1])

offspring1, offspring2 = FBSUtils.CrossoverActions.order_crossover(
    parent1, parent2)

logger.info(offspring1.permutation)
logger.info(offspring1.bay)
logger.info(offspring2.permutation)
logger.info(offspring2.bay)
