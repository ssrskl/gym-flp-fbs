import random

import numpy as np
import FbsEnv.utils.FBSUtil as FBSUtil


chosen_functions = ["facility_swap", "bay_flip"]
# 获取这些函数对象
functions = [
    getattr(FBSUtil, func)
    for func in chosen_functions
    if callable(getattr(FBSUtil, func))
]

random_func = random.choice(functions)

permutation = np.array([3, 5, 7, 1, 4, 6, 2])
bay = np.array([0, 0, 1, 0, 0, 0, 1])

print(f"执行的方法为：{random_func.__name__}")
new_permutation, new_bay = random_func(permutation, bay)
print(new_permutation)
print(new_bay)
