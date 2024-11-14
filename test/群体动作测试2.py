# 群体动作空间测试
import gym
import matplotlib.pyplot as plt
import numpy as np
import logging
from FbsEnv.utils.FBSUtil import FBSUtils
from FbsEnv.envs.FBSModel import FBSModel

instance = "O9-maoyan"

parent1 = FBSModel([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6])
parent2 = FBSModel([6, 5, 4, 3, 2, 1], [6, 5, 4, 3, 2, 1])

offspring1, offspring2 = FBSUtils.CrossoverActions.order_crossover(
    parent1, parent2)

logging.info(offspring1.permutation)
logging.info(offspring1.bay)
logging.info(offspring2.permutation)
logging.info(offspring2.bay)
