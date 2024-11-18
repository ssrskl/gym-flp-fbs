import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DQN
from FbsEnv.envs.FBSModel import FBSModel
import FbsEnv.utils.FBSUtil as FBSUtil

instance = "AB20-ar3"
env = gym.make("FbsEnv-v0", instance=instance)
#[ 6 17  5 18 12 19 15 16 20  1 10 13 11  2  4  7  3 14  8  9], [0 0 1 0 0 1 0 1 0 0 0 1 0 0 1 0 0 0 1 1]
# fbs_model = FBSModel(
# [6, 17, 5, 18, 12, 19, 15, 16, 20, 1, 10, 13, 11, 2, 4, 7, 3, 14, 8, 9],
# [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1])
# [ 6 13  3  7  5 20 10  9 18 19 16  4 12 15 17 11  8 14  1  2]，[0 0 0 1 0 1 1 0 0 0 1 0 1 0 0 0 1 0 0 1]
fbs_model = FBSModel(
    [6, 13, 3, 7, 5, 20, 10, 9, 18, 19, 16, 4, 12, 15, 17, 11, 8, 14, 1, 2],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1])
env.reset(fbs_model=fbs_model)
env.render()
env.step(3)
env.render()
env.step(3)
env.render()
env.step(3)
env.render()
env.step(3)
env.render()
env.step(3)
env.render()
env.step(3)
env.render()
env.step(3)
env.render()
env.step(3)
env.render()
env.step(3)
env.render()
env.step(3)
env.render()
