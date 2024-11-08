import numpy as np


class FBSModel:
    def __init__(self, permutation: list[int], bay: list[int]):
        self._permutation = permutation
        self._bay = bay
    @property
    def permutation(self) -> list[int]:
        return self._permutation.copy()

    @property
    def bay(self) -> list[int]:
        return self._bay.copy()

    @permutation.setter
    def permutation(self, permutation: list[int]):
        self._permutation = permutation

    @bay.setter
    def bay(self, bay: list[int]):
        if len(bay) != len(self._permutation):
            raise ValueError("bay的长度必须与permutation的长度相同")
        self._bay = bay

    def permutationToArray(self) -> np.ndarray:
        """将排列转换为二维数组"""
        self._bay[-1] = 1  # 将bay的最后一个元素设置为1
        array = []
        start = 0
        for i, val in enumerate(self._bay):
            if val == 1:
                array.append(self._permutation[start : i + 1])
                start = i + 1
        return array