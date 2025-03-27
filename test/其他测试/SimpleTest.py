
import FbsEnv
import gym
import FbsEnv.utils.FBSUtil as FBSUtil

instance = "SC35-maoyan"
env = gym.make("FbsEnv-v0", instance=instance)
env.reset()
print(env.state)
env.render()
