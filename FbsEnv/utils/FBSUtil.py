import itertools
import math
import random
import gym
import numpy as np
import re
from itertools import permutations, product
import logging
import colorlog
from functools import wraps
from FbsEnv.envs.FBSModel import FBSModel
import copy
from loguru import logger


class FBSUtils:

    class MutateActions:

        @staticmethod
        def facility_swap(fbs_model: FBSModel):
            logging.info("执行设施交换")
            pass

        @staticmethod
        def bay_flip(fbs_model: FBSModel):
            logging.info("执行区带反转")
            pass

    class CrossoverActions:

        @staticmethod
        def order_crossover(parent1: FBSModel,
                            parent2: FBSModel) -> tuple[FBSModel, FBSModel]:
            parent1_perm = parent1.permutation
            parent2_perm = parent2.permutation
            parent1_bay = parent1.bay
            parent2_bay = parent2.bay
            # 类型转换
            if isinstance(parent1_perm, np.ndarray):
                parent1_perm = parent1_perm.tolist()
            if isinstance(parent2_perm, np.ndarray):
                parent2_perm = parent2_perm.tolist()
            if isinstance(parent1_bay, np.ndarray):
                parent1_bay = parent1_bay.tolist()
            if isinstance(parent2_bay, np.ndarray):
                parent2_bay = parent2_bay.tolist()
            size = len(parent1_perm)
            startPoint, endPoint = sorted(
                np.random.choice(size, 2, replace=False))
            logging.info(
                f"order_crossover-->startPoint: {startPoint}, endPoint: {endPoint}"
            )
            crossover_part_1 = parent1_perm[startPoint:endPoint + 1]
            crossover_part_2 = parent2_perm[startPoint:endPoint + 1]
            # 获取 parent1 中去除 crossover_part_2 的部分
            parent1_remaining = [
                elem for elem in parent1_perm if elem not in crossover_part_2
            ]
            # 获取 parent2 中去除 crossover_part_1 的部分
            parent2_remaining = [
                elem for elem in parent2_perm if elem not in crossover_part_1
            ]
            offspring_1_perm = parent1_remaining[:
                                                 startPoint] + crossover_part_2 + parent1_remaining[
                                                     startPoint:]
            offspring_2_perm = parent2_remaining[:
                                                 startPoint] + crossover_part_1 + parent2_remaining[
                                                     startPoint:]
            offspring_1_bay = parent1_bay[:startPoint] + parent2_bay[
                startPoint:endPoint + 1] + parent1_bay[endPoint + 1:]
            offspring_2_bay = parent2_bay[:startPoint] + parent1_bay[
                startPoint:endPoint + 1] + parent2_bay[endPoint + 1:]
            offspring_1 = FBSModel(offspring_1_perm, offspring_1_bay)
            offspring_2 = FBSModel(offspring_2_perm, offspring_2_bay)
            return offspring_1, offspring_2


def fill_without_duplicates(parent1: list[int], parent2: list[int],
                            startPoint: int,
                            endPoint: int) -> tuple[list[int], list[int]]:
    crossover_part_1 = parent1[startPoint:endPoint + 1]
    crossover_part_2 = parent2[startPoint:endPoint + 1]
    # 获取 parent1 中去除 crossover_part_2 的部分
    parent1_remaining = [
        elem for elem in parent1 if elem not in crossover_part_2
    ]
    # 获取 parent2 中去除 crossover_part_1 的部分
    parent2_remaining = [
        elem for elem in parent2 if elem not in crossover_part_1
    ]

    offspring_1 = parent1_remaining[:
                                    startPoint] + crossover_part_2 + parent1_remaining[
                                        startPoint:]
    offspring_2 = parent2_remaining[:
                                    startPoint] + crossover_part_1 + parent2_remaining[
                                        startPoint:]

    return offspring_1, offspring_2


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
def getAreaData(
    df,
) -> tuple[np.ndarray, float]:
    """
    从 DataFrame 中提取或计算面积相关数据。
    参数:
        df (pd.DataFrame): 输入的 DataFrame，可能包含面积、长度、宽度和横纵比数据。
    返回:
        tuple: 面积areas和横纵比aspects
    """
     # 获取包含特定关键词的列并转换为一维数组
    def get_column_data(df, pattern):
        cols = df.filter(regex=re.compile(pattern, re.IGNORECASE)).columns
        return df[cols].to_numpy().flatten() if not cols.empty else None
    areas = get_column_data(df, 'Area')
    aspects = get_column_data(df, 'Aspect')
    aspects = aspects[0] if aspects is not None else 99
    return areas, aspects


# 随机解生成器
def random_solution_generator(n: int) -> tuple[list[int], list[int]]:
    """生成随机解"""
    # 生成随机排列
    permutation = np.arange(1, n + 1)
    np.random.shuffle(permutation)

    # 生成随机bay划分
    bay = np.zeros(n, dtype=int)
    # 随机选择1-3个位置设置为1(不包括最后一个位置)
    num_ones = np.random.randint(1, min(4, n - 1))
    positions = np.random.choice(n - 1, num_ones, replace=False)
    bay[positions] = 1
    # 确保最后一个位置为1
    bay[-1] = 1

    return permutation.tolist(), bay.tolist()


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
    while k <= n:
        # 计算W的k分
        l = L / k
        w = area / l  # 每个设施的宽度
        aspect_ratio = np.maximum(w, l) / np.minimum(w, l)
        # 验证k分是否可行
        # print("a/l", a / l)
        # 合格个数
        if beta is not None:
            qualified_number = np.sum((aspect_ratio >= 1)
                                      & (aspect_ratio <= beta))
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
def getCoordinates_mao(fbs_model: FBSModel, area, H):
    permutation = fbs_model.permutation
    bay = fbs_model.bay
    bays = permutationToArray(permutation, bay) # 将排列按照划分点分割成多个子数组，每个子数组代表一个区段的排列
    # 初始化长度、宽度和坐标数组
    n  = len(permutation)
    lengths = np.zeros(n) # 每个设施的长度
    widths = np.zeros(n) # 每个设施的宽度
    fac_x = np.zeros(n) # 每个设施的x坐标
    fac_y = np.zeros(n) # 每个设施的y坐标
    # 计算每个区带的坐标和尺寸
    x = 0
    start = 0 # 记录当前子数组的起始索引
    # 从上向下排列
    for b in bays:
        indices = np.array(b) - 1
        bay_areas = area[indices]
        # 计算每个设施的长度和宽度
        widths[start:start + len(bay_areas)] = np.sum(bay_areas) / H
        lengths[start:start + len(bay_areas)] = bay_areas / widths[start:start + len(bay_areas)]
        # 计算设施的x坐标
        fac_x[start:start + len(bay_areas)] = widths[start:start + len(bay_areas)] * 0.5 + x
        x += np.sum(bay_areas) / H
        # 计算设施的y坐标
        y = np.cumsum(lengths[start:start + len(bay_areas)]) - lengths[start:start + len(bay_areas)] * 0.5
        fac_y[start:start + len(bay_areas)] = y
        start += len(bay_areas)
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
        np.array([[(x[i] - x[j])**2 + (y[i] - y[j])**2 for j in range(len(x))]
                  for i in range(len(x))]))


# 计算曼哈顿距离矩阵
def getManhattanDistances(x, y):
    """计算曼哈顿距离矩阵
    Args:
        x (np.ndarray): 设施x坐标
        y (np.ndarray): 设施y坐标
    """
    return np.array(
        [[abs(x[i] - x[j]) + abs(y[i] - y[j]) for j in range(len(x))]
         for i in range(len(x))],
        dtype=float,
    )


def permutationMatrix(a):
    P = np.zeros((len(a), len(a)))
    for idx, val in enumerate(a):
        logging.debug(f"idx: {idx}, val: {val}")
        P[idx][val - 1] = 1
    return P


def getTransportIntensity(D, F, fbs_model: FBSModel):
    logger.info("计算物流强度矩阵")
    logger.info(f"D: \n{D}")
    logger.info(f"F: \n{F}")
    permutation = fbs_model.permutation
    P = permutationMatrix(permutation)
    return np.dot(np.dot(D, P), np.dot(F, P.T))


# 计算MHC
def getMHC(D, F, fbs_model: FBSModel):
    permutation = fbs_model.permutation
    P = permutationMatrix(permutation)
    logger.info(f"P: \n{P}")
    # MHC = np.sum(np.tril(np.dot(P.T, np.dot(D, P))) * (F.T))
    # MHC = np.sum(np.triu(D) * (F))
    MHC = np.sum(D * F)
    # transport_intensity = np.dot(np.dot(D, P), np.dot(F, P.T))
    # MHC = np.trace(transport_intensity)
    return MHC


# 计算适应度
import numpy as np

def getFitness(mhc, fac_b, fac_h, fac_limit_aspect=None, k=3):
    """
    计算适应度。

    参数:
    mhc: float, MHC 的值
    fac_b: list or np.ndarray, 设施的宽度
    fac_h: list or np.ndarray, 设施的高度
    fac_limit_aspect: float or None, 宽高比的限制值，若为 None 则不限制宽高比
    k: int, 惩罚项的指数，默认为 3

    返回:
    fitness: float, 适应度值
    """
    # 将输入转换为 NumPy 数组
    fac_b = np.array(fac_b)
    fac_h = np.array(fac_h)
    MHC = mhc

    if fac_limit_aspect is None:
        # 检查宽度和高度是否都 >= 1
        non_feasible = (fac_b < 1) | (fac_h < 1)
    else:
        # 计算宽高比
        aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
        # 检查宽高比是否在 1 到 fac_limit_aspect 之间
        non_feasible = (aspect_ratio < 1) | (aspect_ratio > fac_limit_aspect)

    # 计算不可行设施的数量
    non_feasible_counter = np.sum(non_feasible)
    # 计算适应度
    fitness = MHC + MHC * (non_feasible_counter ** k)
    return fitness


def StatusUpdatingDevice(fbs_model: FBSModel, a, H, F, fac_limit_aspect_ratio):
    fac_x, fac_y, fac_b, fac_h = getCoordinates_mao(fbs_model, a, H)
    fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
    D = getManhattanDistances(fac_x, fac_y) # 曼哈顿距离
    # D = getEuclideanDistances(fac_x, fac_y) # 欧几里得距离
    TM = getTransportIntensity(D, F, fbs_model)
    mhc = getMHC(D, F, fbs_model)
    fitness = getFitness(mhc, fac_b, fac_h, fac_limit_aspect_ratio)
    return (fac_x, fac_y, fac_b, fac_h, fac_aspect_ratio, D, TM, mhc, fitness)


# ---------------------------------------------------FBS局部优化开始---------------------------------------------------
# Shuffle单区带优化
def shuffleOptimization(env, bay_index):
    tmp_env = copy.deepcopy(env)
    fac_list = permutationToArray(tmp_env.fbs_model.permutation,
                                  tmp_env.fbs_model.bay)
    child_permutation = fac_list[bay_index]
    # 对child_permutation进行shuffle n*n 次
    n = env.n
    max_not_improve_steps = n * 100  # 最大不改进步数
    best_fitness = tmp_env.fitness.copy()
    best_permutation = tmp_env.fbs_model.permutation.copy()
    not_improve_steps = 0
    for _ in range(n * n):
        np.random.shuffle(child_permutation)
        fac_list[bay_index] = child_permutation  # 更新fac_list
        permutation, bay = arrayToPermutation(fac_list)
        tmp_env.reset(fbs_model=FBSModel(permutation, bay))
        if tmp_env.fitness < best_fitness:
            best_fitness = tmp_env.fitness
            best_permutation = tmp_env.fbs_model.permutation
            not_improve_steps = 0
        else:
            not_improve_steps += 1
        if not_improve_steps > max_not_improve_steps:
            break
    return best_permutation, tmp_env.fbs_model.bay


# 全排列局部优化
def fullPermutationOptimization(permutation, bay, a, W, D, F,
                                fac_limit_aspect):
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
        facx, facy, facb, fach = getCoordinates_mao(convert_perm, bay, a, H)
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
    tmp_env = copy.deepcopy(env)
    fac_list = permutationToArray(tmp_env.fbs_model.permutation,
                                  tmp_env.fbs_model.bay)
    best_fitness = tmp_env.fitness.copy()
    best_permutation = tmp_env.fbs_model.permutation.copy()
    best_bay = tmp_env.fbs_model.bay.copy()
    child_permutation = fac_list[bay_index]
    child_permutations = itertools.permutations(child_permutation)
    for child_perm in child_permutations:  # 从单区带的全排列中选择最优的排列
        fac_list[bay_index] = child_perm  # 合并到fac_list
        permutation, bay = arrayToPermutation(fac_list)
        tmp_env.reset(fbs_model=FBSModel(permutation, bay))
        if tmp_env.fitness < best_fitness:
            best_fitness = tmp_env.fitness
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
            fac_x, fac_y, fac_b, fac_h = getCoordinates_mao(
                new_perm, bay, a, H)
            fitness = getFitness(mhc, fac_b, fac_h, fac_limit_aspect)
            if fitness < best_fitness:
                best_fitness = fitness
                best_perm = new_perm
                improved = True
        permutation = best_perm.copy()
    return best_perm


# 排列优化算法
def arrangementOptimization(permutation: np.ndarray, bay: np.ndarray,
                            instance: str):  # -> tuple[ndarray, ndarray]:
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

    @wraps(func)
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
    permutation, bay = arrayToPermutation(
        fac_list)  # 将fac_list转换为permutation和bay
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
def repair(
    permutation: np.ndarray,
    bay: np.ndarray,
    fac_b: np.ndarray,
    fac_h: np.ndarray,
    fac_limit_aspect: float,
):
    """修复bay"""
    # logger.info(f"{permutation}，{bay}")
    # 转换为二维数组
    array = permutationToArray(permutation, bay)
    # 遍历每个bay
    for i, bay in enumerate(array):
        # logger.info(f"当前第{i}个区带：{bay}")
        tmp_array = array[:]
        # 计算所有的设施的横纵比
        fac_aspect_ratio = np.maximum(fac_b, fac_h) / np.minimum(fac_b, fac_h)
        current_bay_fac_aspect_ratio = np.array(
            [fac_aspect_ratio[b - 1] for b in bay])
        current_bay_fac_hv_ratio = np.array(
            [fac_b[b - 1] / fac_h[b - 1] for b in bay])
        # 如果当前bay的设施的横纵比不满足条件，则进行修复
        if np.any((current_bay_fac_aspect_ratio < 1)
                  | (current_bay_fac_aspect_ratio > fac_limit_aspect)):
            # logger.info(f"区带{i}不满足条件")
            # 如果太宽了，说明这个bay中的设施过多，则将其对半分（太宽：横坐标长度/纵坐标长度 > 横纵比）这里使用bay的平均值
            if np.any(current_bay_fac_hv_ratio > fac_limit_aspect):
                # print(f"区带{i}有设施太宽了")
                # 将当前bay的设施随机对半分
                np.random.shuffle(tmp_array[i])
                split_array = np.array_split(tmp_array[i], 2)
                tmp_array[i] = split_array[0]
                tmp_array.insert(i + 1, split_array[1])
            # 如果太窄了，说明这个bay中的设施过少，则将当前bay与相邻的bay进行合并（太窄：纵坐标长度/横坐标长度 > 横纵比）
            else:
                # print(f"区带{i}有设施太窄了")
                # 将当前bay的设施与相邻的bay进行合并
                if i + 1 < len(tmp_array):
                    tmp_array[i] = np.concatenate(
                        (tmp_array[i], tmp_array[i + 1]))
                    tmp_array.pop(i + 1)
                else:
                    tmp_array[i] = np.concatenate(
                        (tmp_array[i], tmp_array[i - 1]))
                    tmp_array.pop(i - 1)
            array = tmp_array
            break
    # logger.info(f"修复后的bay：{array}")
    # 转换为排列和bay
    permutation, bay = arrayToPermutation(array)
    return permutation, bay


# -------------------------------------------------个体动作结束-------------------------------------------------
# -------------------------------------------------群体动作开始-------------------------------------------------
# 顺序交叉
def orderCrossover(parent1: FBSModel, parent2: FBSModel):
    """顺序交叉"""
    size = len(parent1.permutation)
    offspring1_permutation = [-1] * size  # 初始化第一个子代的设施序列
    offspring2_permutation = [-1] * size  # 初始化第二个子代的设施序列
    start, end = sorted(random.sample(range(size), 2))  # 随机选择交叉点范围
    offspring1_permutation[start:end + 1] = parent1.permutation[start:end + 1]
    offspring2_permutation[start:end + 1] = parent2.permutation[start:end + 1]
    pos1 = (end + 1) % size  # 填充第一个子代的起始位置
    pos2 = (end + 1) % size  # 填充第二个子代的起始位置
    for i in range(size):
        candidate1 = parent2.permutation[(end + 1 + i) % size]
        candidate2 = parent1.permutation[(end + 1 + i) % size]
        if candidate1 not in offspring1_permutation:
            offspring1_permutation[pos1] = candidate1
            pos1 = (pos1 + 1) % size
        if candidate2 not in offspring2_permutation:
            offspring2_permutation[pos2] = candidate2
            pos2 = (pos2 + 1) % size
    offspring1_bay = [-1] * size  # 初始化第一个子代的区带数组
    offspring2_bay = [-1] * size  # 初始化第二个子代的区带数组
    for i in range(size):
        facility1 = offspring1_permutation[i]
        facility2 = offspring2_permutation[i]
        # 在第一个亲本中找出该设施的位置，并继承对应的区带信息
        index_in_parent1_for_offspring1 = np.where(
            parent1.permutation == facility1)[0][0]
        index_in_parent2_for_offspring2 = np.where(
            parent2.permutation == facility2)[0][0]
        offspring1_bay[i] = parent1.bay[index_in_parent1_for_offspring1]
        offspring2_bay[i] = parent2.bay[index_in_parent2_for_offspring2]
    offspring1_bay[-1] = 1
    offspring2_bay[-1] = 1
    # 转换为整数
    offspring1_permutation = np.array(offspring1_permutation, dtype=int)
    offspring2_permutation = np.array(offspring2_permutation, dtype=int)
    offspring1_bay = np.array(offspring1_bay, dtype=int)
    offspring2_bay = np.array(offspring2_bay, dtype=int)
    return (
        offspring1_permutation,
        offspring1_bay,
        offspring2_permutation,
        offspring2_bay,
    )


# -------------------------------------------------群体动作结束-------------------------------------------------

# ---------------------------------------------------FBS动作空间结束---------------------------------------------------


def permutationToArray(permutation, bay):
    """将排列转换为二维数组"""
    bay[-1] = 1  # 将bay的最后一个元素设置为1
    array = []
    start = 0
    for i, val in enumerate(bay):
        if val == 1:
            array.append(permutation[start:i + 1])
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

