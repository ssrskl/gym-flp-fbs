import numpy as np
import FbsEnv.utils.FBSUtil as FBSUtil

permutation = np.array([4, 5, 3, 6, 7, 1, 2])
bay = np.array([0, 1, 0, 1, 1, 1, 0])
fac_list = FBSUtil.permutationToArray(permutation, bay)
print(fac_list)

