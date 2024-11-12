import gym
from FbsEnv.envs.FBSModel import FBSModel
import logging

instance = "O7-maoyan"
env = gym.make("FbsEnv-v0", instance=instance)
env.reset(fbs_model=FBSModel([3, 5, 7, 1, 4, 6, 2], [0, 0, 1, 0, 0, 0, 1]))
env.render()
