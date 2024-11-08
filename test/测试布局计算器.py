# 测试布局计算器
from FbsEnv.envs.FBSModel import FBSModel
from FbsEnv.utils import FBSUtil
import numpy as np
import logging


fbs_model = FBSModel([3, 5, 7, 1, 4, 6, 2], [0, 0, 1, 0, 0, 0, 1])
area = np.array([16, 16, 16, 36, 9, 9, 9], dtype=float)
W = 13
fac_x, fac_y, fac_b, fac_h = FBSUtil.getCoordinates_mao(fbs_model, area, W)
logging.info(f"fac_x: {fac_x}")
logging.info(f"fac_y: {fac_y}")
logging.info(f"fac_b: {fac_b}")
logging.info(f"fac_h: {fac_h}")
