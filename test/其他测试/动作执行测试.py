
import FbsEnv
import gym
import FbsEnv.utils.FBSUtil as FBSUtil

instance = "SC35-maoyan"
env = gym.make("FbsEnv-v0", instance=instance)
env.reset()
original_state = env.state
print(original_state)
new_state, reward, done, info = env.step(1)
print(env.state) # env已经执行了step，所以已经发生了变化
print(new_state)
print(original_state) # 原始状态没有发生变化
