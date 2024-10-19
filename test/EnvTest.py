# 环境注册测试
import FbsEnv
import gym

env = gym.make("FbsEnv-v0")
env.reset()
env.step(0)
env.render()
env.close()

