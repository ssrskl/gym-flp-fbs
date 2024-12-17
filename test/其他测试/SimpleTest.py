
import FbsEnv
import gym
import FbsEnv.utils.FBSUtil as FBSUtil


instance = "Du62"
env = gym.make("FbsEnv-v0", instance=instance)
env.reset()
env.render()
