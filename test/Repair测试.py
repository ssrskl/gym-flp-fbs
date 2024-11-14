import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import DQN
from FbsEnv.envs.FBSModel import FBSModel
import FbsEnv.utils.FBSUtil as FBSUtil

instance = "O9-maoyan"
env = gym.make("FbsEnv-v0", instance=instance)
fbs_model = FBSModel([9, 8, 7, 6, 3, 1, 2, 4, 5], [0, 0, 0, 1, 0, 0, 0, 0, 1])
env.reset(fbs_model=fbs_model)
env.render()
env.step(8)
env.render()

