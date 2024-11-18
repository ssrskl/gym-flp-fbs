import gym
from FbsEnv.envs.FBSModel import FBSModel
import logging

from FbsEnv.utils import FBSUtil

# AB20_ar3_permutation, AB20_ar3_bay = FBSUtil.arrayToPermutation([
#     [16, 11],
#     [17, 13, 15],
#     [12, 9, 10, 14],
#     [1, 19, 3],
#     [6, 4, 2, 7, 8, 5],
#     [18, 20],
# ])

AB20_ar3_permutation, AB20_ar3_bay = FBSUtil.arrayToPermutation([
    [1, 18, 5],
    [20, 8, 7, 6],
    [2, 4, 19, 3],
    [10, 14, 9, 15],
    [12, 17, 13],
    [16, 11],
])

instance = {
    "AB20-ar3": FBSModel(AB20_ar3_permutation.tolist(), AB20_ar3_bay.tolist()),
}
env = gym.make("FbsEnv-v0", instance="AB20-ar3")
env.reset(fbs_model=instance["AB20-ar3"])
env.render()
