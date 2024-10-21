import itertools
import math
import random
import gym
import numpy as np
import re
from itertools import permutations, product
import logging
import colorlog


# 物流强度矩阵转换
def transfer_matrix(matrix: np.ndarray):
    """
    转置矩阵
    :param matrix: 矩阵
    :return: 转置后的矩阵
    """
    print("转换前: ", matrix)
    LowerTriangular = np.tril(matrix, -1).T
    resultMatrix = LowerTriangular + matrix
    resultMatrix = np.triu(resultMatrix)
    print("转换后: ", resultMatrix)
    return resultMatrix


# 获取面积数据
def getAreaData(df):
    # 检查Area数据是否存在，存在则转换为numpy数组，否则为None
    if np.any(df.columns.str.contains("Area", na=False, case=False)):
        a = df.filter(regex=re.compile("Area", re.IGNORECASE)).to_numpy()
    else:
        a = None

    if np.any(df.columns.str.contains("Length", na=False, case=False)):
        l = df.filter(regex=re.compile("Length", re.IGNORECASE)).to_numpy()
        l = np.reshape(l, (l.shape[0],))
    else:
        l = None

    if np.any(df.columns.str.contains("Width", na=False, case=False)):
        w = df.filter(regex=re.compile("Width", re.IGNORECASE)).to_numpy()
        w = np.reshape(w, (w.shape[0],))
    else:
        w = None
    # 横纵比
    if np.any(df.columns.str.contains("Aspect", na=False, case=False)):
        ar = df.filter(regex=re.compile("Aspect", re.IGNORECASE)).to_numpy()
        # print("横纵比数据: ", ar)
    else:
        ar = None

    l_min = 1  # 最小长度
    # 面积数据不存在，则根据长度和宽度计算面积
    if a is None:
        if not l is None and not w is None:
            a = l * w
        elif not l is None:
            a = l * max(l_min, max(l))
        else:
            a = w * max(l_min, max(w))

    # 如果横纵比存在上下限则不变，否则下限设置为1
    if not ar is None and ar.ndim > 1:
        if ar.shape[1] == 1:
            ar = np.hstack((np.ones((ar.shape[0], 1)), ar))
        else:
            pass
    if not a is None and a.ndim > 1:
        # a = a[np.where(np.max(np.sum(a, axis = 0))),:]
        a = a[:, 0]
    a = np.reshape(a, (a.shape[0],))
    return ar, l, w, a, l_min


# k分初始解生成器(输入：面积数据a，设施数n，横纵比限制beta，厂房x轴长度L)
def binary_solution_generator(area, n, beta, L):
    # 存储可行的k分解
    bay_list = []
    # 分界参数
    k = 2
    # 计算面积之和
    total_area = np.sum(area)
    print("总面积: ", total_area)
    # 生成一个设施默认的编号序列
    permutation = np.arange(1, n + 1)
    # 根据area对序列进行排序
    permutation = permutation[np.argsort(area[permutation - 1])]
    # 对a也进行排序
    area = np.sort(area)
    # 对beta也按照a的顺序进行排序
    if beta is not None:
        beta = np.array([beta[i - 1] for i in permutation])
    while k <= n:
        # 计算W的k分
        l = L / k
        w = area / l  # 每个设施的宽度
        aspect_ratio = np.maximum(w, l) / np.minimum(w, l)
        # 验证k分是否可行
        # print("a/l", a / l)
        # 合格个数
        if beta is not None:
            qualified_number = np.sum(
                (aspect_ratio >= beta[:, 0]) & (aspect_ratio <= beta[:, 1])
            )
        else:
            qualified_number = np.sum((w > 1) & (l > 1))
        # 如果合格个数大于等于3/4*n，即此k值可行
        bay = np.zeros(n)
        if qualified_number >= n * 3 / 4:
            # print("可行的k: ", k)
            # print("符合的个数: ", qualified_number)
            # 根据面积和找到k分界点
            best_partition, partitions = _find_best_partition(area, k)
            # print("序列分界点: ", best_partition)
            # 将k分界点转换为bay
            for i, p in enumerate(best_partition):
                bay[p - 1] = 1
            # 将最后一个分界点设为1
            bay[n - 1] = 1
            bay_list.append(bay)
        k += 1
    # print("可行的bay: ", bay_list)
    # 从可行的bay中随机选择一个
    if len(bay_list) > 0:
        bay = random.choice(bay_list)
    #  TODO 对permutation使用bay进行划分，并对每个bay中的设施进行随机排列
    j = 0
    for i in range(len(bay)):
        if bay[i] == 1:
            np.random.shuffle(permutation[j:i])
            j = i + 1
    return (permutation, bay)


# k分划分法的动态规划版（输入：排列序列a，划分数k）
def _find_best_partition(arr, k):
    print(f"k分划分法-->k = {k}")
    n = len(arr)
    target_sum = np.sum(arr) // k

    # dp[i][j] 表示前i个设施被划分为j个组的最小差异和
    dp = np.full((n + 1, k + 1), float("inf"))
    dp[0][0] = 0

    # sum[i] 表示arr[0:i]的累积和
    cum_sum = np.cumsum(arr)

    partition_idx = [[[] for _ in range(k + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, k + 1):
            for m in range(i):
                current_sum = cum_sum[i - 1] - (cum_sum[m - 1] if m > 0 else 0)
                current_diff = abs(target_sum - current_sum)
                total_diff = dp[m][j - 1] + current_diff

                if total_diff < dp[i][j]:
                    dp[i][j] = total_diff
                    partition_idx[i][j] = partition_idx[m][j - 1] + [i]

    best_partition = partition_idx[-1][-1][:-1]  # 排除最后一个分界点
    return best_partition, np.split(arr, best_partition)


# 计算设施坐标和尺寸
def getCoordinates_mao(permutation, bay, area, W):
    # 将排列按照划分点分割成多个子数组，每个子数组代表一个区段的排列
    bays = np.split(permutation, indices_or_sections=np.where(bay == 1)[0][:-1] + 1)

    # 初始化长度、宽度和坐标数组
    lengths = np.zeros(len(permutation))
    widths = np.zeros(len(permutation))
    fac_x = np.zeros(len(permutation))
    fac_y = np.zeros(len(permutation))

    x = 0
    start = 0
    # 从上向下排列
    for b in bays:
        areas = area[b - 1]
        end = start + len(areas)

        # 计算每个设施的长度和宽度
        lengths[start:end] = np.sum(areas) / W
        widths[start:end] = areas / lengths[start:end]

        # 计算设施的x坐标
        fac_x[start:end] = lengths[start:end] * 0.5 + x
        x += np.sum(areas) / W

        # 计算设施的y坐标
        y = np.cumsum(widths[start:end]) - widths[start:end] * 0.5
        fac_y[start:end] = y
        start = end
    # 顺序恢复
    order = np.argsort(permutation)
    fac_x = fac_x[order]
    fac_y = fac_y[order]
    lengths = lengths[order]
    widths = widths[order]
    return fac_x, fac_y, lengths, widths


# 计算欧几里得距离矩阵
def getEuclideanDistances(x, y):
    """计算欧几里得距离矩阵
    Args:
        x (np.ndarray): 设施x坐标
        y (np.ndarray): 设施y坐标
    Returns:
        np.ndarray: 距离矩阵
    """
    return np.sqrt(
        np.array(
            [
                [(x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 for j in range(len(x))]
                for i in range(len(x))
            ]
        )
    )


# 计算曼哈顿距离矩阵
def getManhattanDistances(x, y):
    """计算曼哈顿距离矩阵
    Args:
        x (np.ndarray): 设施x坐标
        y (np.ndarray): 设施y坐标
    """
    return np.array(
        [
            [abs(x[i] - x[j]) + abs(y[i] - y[j]) for j in range(len(x))]
            for i in range(len(x))
        ],
        dtype=float,
    )


def permutationMatrix(a):
    P = np.zeros((len(a), len(a)))
    for idx, val in enumerate(a):
        logging.debug(f"idx: {idx}, val: {val}")
        P[idx][val - 1] = 1
    return P


def getTransportIntensity(D, F, s):
    P = permutationMatrix(s)
    return np.dot(np.dot(D, P), np.dot(F, P.T))


# 计算MHC
def getMHC(D, F, permutation):
    P = permutationMatrix(permutation)
    # MHC = np.sum(np.tril(np.dot(P.T, np.dot(D, P))) * (F.T))
    # MHC = np.sum(np.triu(D) * (F))
    MHC = np.sum(D * F)
    return MHC


# 计算适应度
def getFitness(mhc, fac_b, fac_h, fac_limit_aspect):
    aspect_ratio_list = []
    k = 1
    non_feasible_counter = 0
    MHC = mhc

    if fac_limit_aspect is None:
        for i, (b, h) in enumerate(zip(fac_b, fac_h)):
            if b < 1 or h < 1:
                non_feasible_counter += 1
    else:
        for i, (b, h) in enumerate(zip(fac_b, fac_h)):
            facility_aspect_ratio = max(b, h) / min(b, h)
            aspect_ratio_list.append(facility_aspect_ratio)
            if not (
                min(fac_limit_aspect[i])
                <= facility_aspect_ratio
                <= max(fac_limit_aspect[i])
            ):
                non_feasible_counter += 1
    aspect_ratio = np.array(aspect_ratio_list)
    fitness = MHC + MHC * (non_feasible_counter**k)
    return fitness


def StatusUpdatingDevice(permutation, bay, a, W, F, fac_limit_aspect_ratio):
    fac_x, fac_y, fac_b, fac_h = getCoordinates_mao(permutation, bay, a, W)
    fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
    D = getManhattanDistances(fac_x, fac_y)
    TM = getTransportIntensity(D, F, permutation)
    mhc = getMHC(D, F, permutation)
    fitness = getFitness(mhc, fac_b, fac_h, fac_limit_aspect_ratio)
    return (fac_x, fac_y, fac_b, fac_h, fac_aspect_ratio, D, TM, mhc, fitness)


# ---------------------------------------------------FBS局部优化开始---------------------------------------------------
# Shuffle单区带优化
def shuffleOptimization(env, bay_index):
    fac_list = permutationToArray(env.permutation, env.bay)
    child_permutation = fac_list[bay_index]
    # 对child_permutation进行shuffle n*n 次
    n = env.n
    max_not_improve_steps = n * 100  # 最大不改进步数
    best_fitness = env.Fitness.copy()
    best_permutation = env.permutation.copy()
    not_improve_steps = 0
    for _ in range(n * n):
        np.random.shuffle(child_permutation)
        fac_list[bay_index] = child_permutation  # 更新fac_list
        permutation, bay = arrayToPermutation(fac_list)
        env.reset(layout=(permutation, bay))
        if env.Fitness < best_fitness:
            best_fitness = env.Fitness
            best_permutation = env.permutation
            not_improve_steps = 0
        else:
            not_improve_steps += 1
        if not_improve_steps > max_not_improve_steps:
            break
    return best_permutation, env.bay


# 全排列局部优化
def fullPermutationOptimization(permutation, bay, a, W, D, F, fac_limit_aspect):
    # 对当前的状态进行局部搜索，返回新的状态和适应度函数值
    # print("开始局部搜索优化")
    # 局部搜索优化，全排列每一个bay中的设施，并计算适应度函数值，选择最优的排列
    best_perm = np.array(permutation)
    best_fitness = float("inf")
    split_indices = np.where(bay == 1)[0] + 1
    split_indices = split_indices[split_indices < len(permutation)]
    bays = np.split(permutation, split_indices)
    # print("bays:", bays)
    perms = [list(permutations(bay)) for bay in bays]  # 对每个bay中的设施进行全排列
    # 对排列后的结果进行笛卡尔积进行组合
    combinations = list(product(*perms))
    combined_permutations = [list(comb) for comb in combinations]
    for perm in combined_permutations:
        convert_perm = np.concatenate(perm)
        print("convert_perm:", convert_perm)
        # 计算当前排列下的设施参数信息
        facx, facy, facb, fach = getCoordinates_mao(convert_perm, bay, a, W)
        MHC = getMHC(D, F, convert_perm)
        # 计算适应度函数值
        fitness = getFitness(MHC, facb, fach, fac_limit_aspect)
        # print("当前排列下的设施参数信息: ", facx, facy, facb, fach)
        # print("当前排列下的适应度函数值: ", fitness)
        if fitness < best_fitness:
            best_fitness = fitness
            best_perm = convert_perm
    # print("局部搜索优化后的最优排列: ", best_perm)
    # print("局部搜索优化后的最优适应度函数值: ", best_fitness)
    return np.array(best_perm)


# 单区带全排列优化
def SingleBayGradualArrangementOptimization(env, bay_index):
    fac_list = permutationToArray(env.permutation, env.bay)
    best_fitness = env.Fitness.copy()
    best_permutation = env.permutation.copy()
    best_bay = env.bay.copy()
    child_permutation = fac_list[bay_index]
    child_permutations = itertools.permutations(child_permutation)
    for child_perm in child_permutations:  # 从单区带的全排列中选择最优的排列
        fac_list[bay_index] = child_perm  # 合并到fac_list
        permutation, bay = arrayToPermutation(fac_list)
        env.reset(layout=(permutation, bay))
        if env.Fitness < best_fitness:
            best_fitness = env.Fitness
            best_permutation = permutation
            best_bay = bay
    return best_permutation, best_bay


# 交换局部优化算法
def exchangeOptimization(
    permutation: np.ndarray,
    bay: np.ndarray,
    a,
    W,
    D,
    F,
    fac_limit_aspect,
):
    best_perm = permutation.copy()  # 最佳排列
    best_fitness = float("inf")  # 最佳适应度函数值
    improved = True  # 标记是否有改进
    while improved:
        improved = False
        for i in range(len(permutation) - 1):
            new_perm = permutation.copy()
            new_perm[i], new_perm[i + 1] = new_perm[i + 1], new_perm[i]
            # 计算当下排列的适应度函数值
            mhc = getMHC(D, F, new_perm)
            fac_x, fac_y, fac_b, fac_h = getCoordinates_mao(new_perm, bay, a, W)
            fitness = getFitness(mhc, fac_b, fac_h, fac_limit_aspect)
            if fitness < best_fitness:
                best_fitness = fitness
                best_perm = new_perm
                improved = True
        permutation = best_perm.copy()
    return best_perm


# 排列优化算法
def arrangementOptimization(
    permutation: np.ndarray, bay: np.ndarray, instance: str
):  # -> tuple[ndarray, ndarray]:
    # 创建env对象
    env = gym.make("fbs-v0", instance=instance)
    env.reset(layout=(permutation, bay))

    # 初始化最佳解
    best_permutation = permutation.copy()
    best_bay = bay.copy()
    best_fitness = env.Fitness

    # 将排列和分区转换成二维数组
    array = permutationToArray(permutation, bay)

    # 遍历每个子数组
    for i in range(len(array)):
        best_sub_perm = array[i].copy()  # 当前子数组的最佳排列
        for perm in itertools.permutations(array[i]):  # 遍历当前子数组的所有排列
            array[i] = perm
            # 将二维数组转换回排列和分区
            permutation, bay = arrayToPermutation(array)
            env.reset(layout=(permutation, bay))
            fitness = env.Fitness
            # 如果找到更优的解，则更新
            if fitness < best_fitness:
                best_fitness = fitness
                best_sub_perm = perm  # 更新子数组最佳排列
                best_permutation = permutation.copy()  # 更新整体排列
                best_bay = bay.copy()

        # 固定当前子数组的最佳排列
        array[i] = best_sub_perm
        print(f"阶段: {i} 最佳排列: {array}")

    # 输出最终的最佳适应度
    print(f"best_fitness: {best_fitness}")
    return best_permutation, best_bay


# ---------------------------------------------------FBS局部优化结束---------------------------------------------------


# ---------------------------------------------------FBS动作空间开始---------------------------------------------------
# 返回的类型为：(np.ndarray, np.ndarray)
# 动作装饰器
def log_action(func):
    def wrapper(*args, **kwargs):
        # 输出方法名
        logging.debug(f"方法名：{func.__name__}")
        logging.debug(
            f"变换前的排列：{args[0]}，变换前的区带：{args[1]}, 设施布局为：{permutationToArray(args[0], args[1])}"
        )
        result = func(*args, **kwargs)
        logging.debug(
            f"变换后的排列：{result[0]}，变换后的区带：{result[1]}, 设施布局为：{permutationToArray(result[0], result[1])}"
        )
        return result

    return wrapper


# -------------------------------------------------单区带动作开始-------------------------------------------------
# 交换同一bay中的两个设施, single表示在同一个bay中交换两个设施
@log_action
def facility_swap_single(permutation: np.ndarray, bay: np.ndarray):
    """交换同一bay中的两个设施"""
    # 选择一个bay
    bay_index = np.where(bay == 1)[0]
    if len(bay_index) < 2:
        return permutation, bay
    # 随机选择两个设施
    i, j = np.random.choice(bay_index, 2, replace=False)
    # 交换设施
    permutation[i], permutation[j] = permutation[j], permutation[i]
    return permutation, bay


# 单一区代Shuffle
@log_action
def shuffle_single(permutation: np.ndarray, bay: np.ndarray):
    """单一区代Shuffle"""
    fac_list = permutationToArray(permutation, bay)
    bay_index = np.random.choice(len(fac_list))  # 随机选择一个bay
    sub_permutation = fac_list[bay_index]  # 得到bay_index对应的子排列
    np.random.shuffle(sub_permutation)  # 将sub_permutation进行shuffle
    fac_list[bay_index] = sub_permutation  # 将sub_permutation放回fac_list
    permutation, bay = arrayToPermutation(fac_list)  # 将fac_list转换为permutation和bay
    return permutation, bay


# -------------------------------------------------单区带动作结束-------------------------------------------------
# -------------------------------------------------个体动作开始-------------------------------------------------
# 交换两个设施
@log_action
def facility_swap(permutation: np.ndarray, bay: np.ndarray):
    """交换两个设施"""
    i, j = np.random.choice(len(permutation), 2, replace=False)  # 随机选择两个设施
    permutation[i], permutation[j] = permutation[j], permutation[i]  # 交换设施
    return permutation, bay


# 将bay的值转换
@log_action
def bay_flip(permutation: np.ndarray, bay: np.ndarray):
    """将bay的值转换"""
    index = np.random.choice(len(bay))
    bay[index] = 1 - bay[index]
    return permutation, bay


# 交换两个bay
@log_action
def bay_swap(permutation: np.ndarray, bay: np.ndarray):
    """交换两个bay"""
    # 转换为二维数组
    array = permutationToArray(permutation, bay)
    if len(array) < 2:
        return permutation, bay  # 如果bay的数量小于2，则直接返回
    # 随机选择两个bay
    i, j = np.random.choice(len(array), 2, replace=False)
    # 交换两个bay
    array[i], array[j] = array[j], array[i]
    # 转换为排列和bay
    permutation, bay = arrayToPermutation(array)
    return permutation, bay


# 对区带shuffle
@log_action
def bay_shuffle(permutation: np.ndarray, bay: np.ndarray):
    """对区带shuffle"""
    fac_list = permutationToArray(permutation, bay)
    np.random.shuffle(fac_list)
    permutation, bay = arrayToPermutation(fac_list)
    return permutation, bay


# 对设施排列shuffle
@log_action
def facility_shuffle(permutation: np.ndarray, bay: np.ndarray):
    """对设施排列shuffle"""
    fac_list = permutationToArray(permutation, bay)
    for i in range(len(fac_list)):
        np.random.shuffle(fac_list[i])
    permutation, bay = arrayToPermutation(fac_list)
    return permutation, bay


# 对排列shuffle
@log_action
def permutation_shuffle(permutation: np.ndarray, bay: np.ndarray):
    """对排列shuffle"""
    np.random.shuffle(permutation)
    return permutation, bay


# 修复bay
@log_action
def repair(
    permutation: np.ndarray,
    bay: np.ndarray,
    fac_b: np.ndarray,
    fac_h: np.ndarray,
    fac_limit_aspect: np.ndarray,
):
    """修复bay"""
    # 转换为二维数组
    array = permutationToArray(permutation, bay)
    # 遍历每个bay
    for i, bay in enumerate(array):
        # 计算当前bay的设施的横纵比
        fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
        # 如果当前bay的设施的横纵比不满足条件，则进行修复
        if not (
            min(fac_limit_aspect[bay - 1])
            <= fac_aspect_ratio
            <= max(fac_limit_aspect[bay - 1])
        ):
            # 如果太宽了，说明这个bay中的设施过多，则将其对半分（太宽：横坐标长度/纵坐标长度 > 横纵比）
            if fac_aspect_ratio > max(fac_limit_aspect[bay - 1]):
                # 将当前bay的设施对半分
                array[i] = array[i][: len(array[i]) // 2]
                array.insert(i + 1, array[i])
            # 如果太窄了，说明这个bay中的设施过少，则将当前bay与相邻的bay进行合并（太窄：纵坐标长度/横坐标长度 > 横纵比）
            elif fac_aspect_ratio < min(fac_limit_aspect[bay - 1]):
                # 将当前bay的设施与相邻的bay进行合并
                array[i] = array[i] + array[i + 1]
                array.pop(i + 1)
    # 转换为排列和bay
    permutation, bay = arrayToPermutation(array)
    return permutation, bay


# -------------------------------------------------个体动作结束-------------------------------------------------
# TODO 2024-10-07 还未完成FBS的Step编写

# ---------------------------------------------------FBS动作空间结束---------------------------------------------------


def permutationToArray(permutation, bay):
    """将排列转换为二维数组"""
    bay[-1] = 1  # 将bay的最后一个元素设置为1
    array = []
    start = 0
    for i, val in enumerate(bay):
        if val == 1:
            array.append(permutation[start : i + 1])
            start = i + 1
    return array


# 将二维数组转换为排列和bay
def arrayToPermutation(array):
    permutation = []
    bay = []
    for sub_array in array:
        permutation.extend(sub_array)
        bay.extend([0] * (len(sub_array) - 1) + [1])
    permutation = np.array(permutation)
    bay = np.array(bay)
    return permutation, bay


def sayHello():
    logging.info("Hello World")
