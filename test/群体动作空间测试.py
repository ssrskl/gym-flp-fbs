# 群体动作空间测试
import gym
import matplotlib.pyplot as plt
import numpy as np
import logging
import FbsEnv.utils.FBSUtil as FBSUtil

instance = "O9-maoyan"
parent1 = gym.make("FbsEnv-v0", instance=instance)
parent1.reset()

parent2 = gym.make("FbsEnv-v0", instance=instance)
parent2.reset()

offspring1_permutation, offspring1_bay, offspring2_permutation, offspring2_bay = (
    FBSUtil.orderCrossover(parent1.fbs_model, parent2.fbs_model)
)

logging.info(offspring1_permutation)
logging.info(offspring1_bay)
logging.info(offspring2_permutation)
logging.info(offspring2_bay)
