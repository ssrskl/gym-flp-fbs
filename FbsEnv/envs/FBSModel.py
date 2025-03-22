class FBSModel:
    def __init__(self, permutation: list[int], bay: list[int]):
        """初始化FBSModel类，传入permutation和bay"""
        if len(permutation) != len(bay):
            raise ValueError("permutation和bay的长度必须相同")
        self._permutation = permutation.copy()
        self._bay = bay.copy()

    @property
    def permutation(self) -> list[int]:
        """获取permutation的副本"""
        return self._permutation.copy()

    @permutation.setter
    def permutation(self, value: list[int]):
        """设置permutation，并检查长度一致性"""
        if len(value) != len(self._bay):
            raise ValueError("permutation的长度必须与bay的长度相同")
        self._permutation = value.copy()

    @property
    def bay(self) -> list[int]:
        """获取bay的副本"""
        return self._bay.copy()

    @bay.setter
    def bay(self, value: list[int]):
        """设置bay，并检查长度一致性"""
        if len(value) != len(self._permutation):
            raise ValueError("bay的长度必须与permutation的长度相同")
        self._bay = value.copy()

    @property
    def array_2d(self) -> list[list[int]]:
        """根据permutation和bay动态计算二维数组"""
        bay_copy = self._bay.copy()
        # 假设最后一个元素必须是1，如果不是则强制设为1
        if bay_copy and bay_copy[-1] != 1:
            bay_copy[-1] = 1
        array = []
        start = 0
        for i, val in enumerate(bay_copy):
            if val == 1:
                array.append(self._permutation[start:i + 1])
                start = i + 1
        return array

    @array_2d.setter
    def array_2d(self, value: list[list[int]]):
        """根据二维数组更新permutation和bay"""
        # 从二维数组反推出permutation
        new_permutation = []
        for subarray in value:
            new_permutation.extend(subarray)

        # 从二维数组反推出bay
        new_bay = []
        total_len = 0
        for subarray in value[:-1]:  # 除了最后一个子数组
            total_len += len(subarray)
            new_bay.extend([0] * (len(subarray) - 1) + [1])
        # 处理最后一个子数组
        if value:
            last_subarray_len = len(value[-1])
            new_bay.extend([0] * (last_subarray_len - 1) + [1])

        # 检查长度是否一致
        if len(new_permutation) != len(new_bay):
            raise ValueError("从array_2d反推出的permutation和bay长度不一致")
        
        # 更新内部数据
        self._permutation = new_permutation
        self._bay = new_bay

# # 示例用法
# if __name__ == "__main__":
#     # 初始化
#     model = FBSModel([1, 2, 3, 4], [0, 1, 0, 1])
#     print("初始array_2d:", model.array_2d)  # [[1, 2], [3, 4]]

#     # 修改permutation
#     model.permutation = [4, 3, 2, 1]
#     print("修改permutation后的array_2d:", model.array_2d)  # [[4, 3], [2, 1]]

#     # 修改bay
#     model.bay = [1, 0, 0, 1]
#     print("修改bay后的array_2d:", model.array_2d)  # [[4], [3, 2, 1]]

#     # 修改array_2d
#     model.array_2d = [[1, 2, 3], [4]]
#     print("修改array_2d后的permutation:", model.permutation)  # [1, 2, 3, 4]
#     print("修改array_2d后的bay:", model.bay)  # [0, 0, 1, 1]