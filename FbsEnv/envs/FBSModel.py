import numpy as np


class FBSModel:

    def __init__(self, permutation: list[int], bay: list[int]):
        if len(permutation) != len(bay):
            raise ValueError("permutation和bay的长度必须相同")
        self._permutation = permutation
        self._bay = bay

    @property
    def permutation(self) -> list[int]:
        return self._permutation.copy()  # 使用浅拷贝，防止外部修改

    @property
    def bay(self) -> list[int]:
        return self._bay.copy()  # 使用浅拷贝，防止外部修改

    @permutation.setter
    def permutation(self, permutation: list[int]):
        self._permutation = permutation

    @bay.setter
    def bay(self, bay: list[int]):
        if len(bay) != len(self._permutation):
            raise ValueError("bay的长度必须与permutation的长度相同")
        self._bay = bay

    def permutationToArray(self) -> list:
        """将排列转换为二维数组，并且不改变原始bay数据"""
        bay_copy = self._bay.copy()
        bay_copy[-1] = 1  # 将bay的最后一个元素设置为1
        array = []
        start = 0
        for i, val in enumerate(bay_copy):
            if val == 1:
                array.append(self._permutation[start:i + 1])
                start = i + 1
        return array
