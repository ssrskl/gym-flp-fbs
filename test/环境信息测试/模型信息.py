# 输出模型的相关信息

from FbsEnv.utils import FBSUtil
import FbsEnv
import gym
from FbsEnv.envs.FBSModel import FBSModel


instance_name = "Du62"
env = gym.make("FbsEnv-v0", instance=instance_name)
env.reset()
env.render()
