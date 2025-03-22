
import FbsEnv
import gym
import FbsEnv.utils.FBSUtil as FBSUtil

instance = "O7-maoyan"
env = gym.make("FbsEnv-v0", instance=instance)
env.reset()
env.render()
