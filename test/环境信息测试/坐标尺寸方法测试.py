from FbsEnv.utils import FBSUtil
import FbsEnv
import gym
from FbsEnv.envs.FBSModel import FBSModel
from loguru import logger

instance_name = "O7-maoyan"
permutation = [3, 5, 7, 1, 4, 6, 2]
bay = [0, 0, 1, 0, 0, 0, 1]
fbsModel = FBSModel(permutation, bay)
env = gym.make("FbsEnv-v0", instance=instance_name)
env.reset(fbs_model=fbsModel)
logger.info(env.areas)


logger.info(FBSUtil.getCoordinates_mao(fbsModel,env.areas,13.0))
env.render()