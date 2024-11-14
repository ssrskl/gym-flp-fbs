import gym
from FbsEnv.envs.FBSModel import FBSModel
import logging

instance = "O9-maoyan"
env = gym.make("FbsEnv-v0", instance=instance)
env.reset(fbs_model=FBSModel([5, 9, 6, 2, 3, 8, 1, 4, 7],
                             [0, 0, 0, 0, 1, 0, 0, 0, 1]))
env.render()
