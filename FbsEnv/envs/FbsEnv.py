import sys
import pickle
import uuid
import gym
from gym import spaces
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from loguru import logger
from FbsEnv.envs.FBSModel import FBSModel
import FbsEnv.utils.FBSUtil as FBSUtil

# 设置日志处理级别
logger.remove()
logger.add(
    sys.stderr,
    level="INFO"
)
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


class FBSEnv(gym.Env):

    def __init__(self, instance=None, seed=None, options=None):
        super(FBSEnv, self).__init__()
        with open(
                r"/Users/maoyan/Codes/Python/gym-flp-fbs/FbsEnv/files/maoyan_cont_instances.pkl",
                "rb",
        ) as file:
            (
                self.problems,
                self.FlowMatrices,
                self.sizes,
                self.LayoutWidths,
                self.LayoutLengths,
            ) = pickle.load(file)
        self.instance = instance
        self.uuid = uuid.uuid4()
        self.F = self.FlowMatrices[self.instance]  # 物流强度矩阵
        self.n = self.problems[self.instance]  # 问题模型的设施数量
        self.areas,self.fac_limit_aspect = (
            FBSUtil.getAreaData(self.sizes[self.instance])
        )  # 面积，横纵比
        logger.debug(f"横纵比: {self.fac_limit_aspect}")
        logger.debug(f"面积: {self.areas}")
        self.H = self.LayoutWidths[self.instance]  # 厂房的长度
        self.W = self.LayoutLengths[self.instance]  # 厂房的宽度
        
        total_area = np.sum(self.areas)  # 设施的总面积
        self.actions = {
            # 0: "facility_swap_single",
            # 1: "shuffle_single",
            0: "facility_swap",
            1: "bay_flip",
            2: "bay_swap",
            # 5: "bay_shuffle",
            # 6: "facility_shuffle",
            # 7: "permutation_shuffle",
            3: "repair",
            4: "idle",
        }  # 动作空间
        self.action_space = spaces.Discrete(len(self.actions))  # 动作空间
        
        # 计算状态向量长度：设施数 * 8个特征
        state_size = self.n * 8
        
        # 为MLP策略初始化一维观察空间
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(state_size,), 
            dtype=np.float32
        )
        
        self.fitness = np.inf

        # ------------------调试信息------------------
        logger.debug("-------------------init初始化信息------------------")
        logger.debug(f"实例: {self.instance}")
        logger.debug(f"设施数量: {self.n}")
        logger.debug(f"设施信息: {self.sizes[self.instance]}")
        logger.debug(f"设施面积: {self.areas}")
        logger.debug(f"设施横纵比: {self.fac_limit_aspect}")
        logger.debug(f"设施总长度H: {self.H}")
        logger.debug(f"设施总宽度W: {self.W}")
        logger.debug("--------------------------------------------------")

    def reset(self, fbs_model = None):
        if fbs_model is None:
            permutation, bay = FBSUtil.binary_solution_generator(self.areas, self.n, self.fac_limit_aspect, self.W)  # 采用k分初始解生成器
            # permutation,bay = FBSUtil.random_solution_generator(self.n) # 采用随机初始解生成器
            bay[-1] = 1  # bay的最后一个位置必须是1，表示最后一个设施是bay的结束
            self.fbs_model = FBSModel(
                permutation.astype(int).tolist(),
                bay.astype(int).tolist())
        else:
            self.fbs_model = fbs_model
        (
            self.fac_x,
            self.fac_y,
            self.fac_h,
            self.fac_b,
            self.fac_aspect_ratio,
            self.D,
            self.TM,
            self.MHC,
            self.fitness,
        ) = FBSUtil.StatusUpdatingDevice(self.fbs_model, self.areas, self.H,
                                         self.F, self.fac_limit_aspect)
        self.previous_fitness = self.fitness  # 初始化上一次的适应度值
        # 更新状态字典
        self.state = self.constructState()
        logger.debug("-------------------reset调试信息------------------")
        logger.debug(f"设施x坐标: {self.fac_x}")
        logger.debug(f"设施y坐标: {self.fac_y}")
        logger.debug(f"设施宽度: {self.fac_b}")
        logger.debug(f"设施高度: {self.fac_h}")
        logger.debug(f"设施横纵比: {self.fac_aspect_ratio}")
        logger.debug(f"设施距离矩阵: {self.D}")
        logger.debug(f"设施移动矩阵: {self.TM}")
        logger.debug(f"设施移动矩阵: {self.MHC}")
        logger.debug(f"设施适应度: {self.fitness}")
        logger.debug(f"状态向量形状: {self.state.shape}")
        logger.debug("--------------------------------------------------")
        return self.state

    def calculate_reward_1(self):
        """
        计算奖励函数 - 结合多个因素的综合奖励
        1. 适应度改善程度（主要因素）
        2. MHC改善程度（次要因素）
        3. 约束满足情况（约束惩罚）
        """
        # 计算适应度改善程度（归一化到[-1, 1]范围）
        fitness_improvement = 0
        if self.previous_fitness > 0:
            fitness_improvement = (self.previous_fitness - self.fitness) / max(self.previous_fitness, self.fitness) 
        
        # 计算MHC改善程度
        mhc_improvement = 0
        if hasattr(self, 'previous_MHC') and self.previous_MHC > 0:
            mhc_improvement = (self.previous_MHC - self.MHC) / max(self.previous_MHC, self.MHC)
        
        # 计算约束违反惩罚 - 基于横纵比的约束
        aspect_ratio_violations = np.sum(
            (self.fac_aspect_ratio < 1) | (self.fac_aspect_ratio > self.fac_limit_aspect)
        ) / self.n  # 归一化到[0, 1]范围
        constraint_penalty = -aspect_ratio_violations * 0.5  # 轻微惩罚
        
        # 奖励动作选择的多样性 - 避免相同动作的重复选择
        action_diversity_bonus = 0.0
        if hasattr(self, 'previous_action') and self.previous_action != self.current_action:
            action_diversity_bonus = 0.05  # 小额奖励不同动作
        
        # 综合奖励计算
        reward = (
            0.7 * fitness_improvement  # 适应度改善的权重
            + 0.2 * mhc_improvement    # MHC改善的权重
            + constraint_penalty       # 约束惩罚
            + action_diversity_bonus   # 动作多样性奖励
        )
        
        # 对reward进行裁剪，避免过大的奖励
        reward = np.clip(reward, -1.0, 1.0)
        
        return reward

    def calculate_reward_2(self):
        return -self.fitness

    def step(self, action):
        # 保存上一个动作
        self.previous_action = getattr(self, 'current_action', None)
        self.current_action = action
        
        # 根据action执行相应的操作
        action_name = self.actions[int(action)]
        # if action_name == "facility_swap_single":
        # self.fbs_model.permutation, self.fbs_model.bay = (
        #     FBSUtil.facility_swap_single(self.fbs_model.permutation,
        #                                  self.fbs_model.bay))
        # elif action_name == "shuffle_single":
        # self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.shuffle_single(
        # self.fbs_model.permutation, self.fbs_model.Xbay)
        if action_name == "facility_swap":
            self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.facility_swap(
                self.fbs_model.permutation, self.fbs_model.bay)
        elif action_name == "bay_flip":
            self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.bay_flip(
                self.fbs_model.permutation, self.fbs_model.bay)
        elif action_name == "bay_swap":
            self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.bay_swap(
                self.fbs_model.permutation, self.fbs_model.bay)
            # elif action_name == "bay_shuffle":
            #     self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.bay_shuffle(
            #         self.fbs_model.permutation, self.fbs_model.bay)
            # elif action_name == "facility_shuffle":
            #     self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.facility_shuffle(
            #         self.fbs_model.permutation, self.fbs_model.bay)
            # elif action_name == "permutation_shuffle":
            self.fbs_model.permutation, self.fbs_model.bay = (
                FBSUtil.permutation_shuffle(self.fbs_model.permutation,
                                            self.fbs_model.bay))
        elif action_name == "repair":
            self.fbs_model.permutation, self.fbs_model.bay = FBSUtil.repair(
                self.fbs_model.permutation, self.fbs_model.bay, self.fac_b,
                self.fac_h, self.fac_limit_aspect)
        elif action_name == "idle":
            pass
        else:
            raise ValueError(f"Invalid action: {action_name}")

        self.previous_MHC = self.MHC  # 保存上一步的MHC
        self.previous_fitness = self.fitness  # 保存上一步的fitness

        # 刷新状态
        (
            self.fac_x,
            self.fac_y,
            self.fac_h,
            self.fac_b,
            self.fac_aspect_ratio,
            self.D,
            self.TM,
            self.MHC,
            self.fitness,
        ) = FBSUtil.StatusUpdatingDevice(self.fbs_model, self.areas, self.H,
                                         self.F, self.fac_limit_aspect)
        # 更新状态字典
        self.state = self.constructState()
        # 计算奖励函数
        reward = self.calculate_reward_1()
        self.previous_fitness = self.fitness
        # 更新info字典，包含更多的调试信息
        info = {
            "TimeLimit.truncated": False,
            "current_fitness": self.fitness,  # 当前适应度值
            "previous_fitness": self.previous_fitness,  # 上一次的适应度值
            "reward": reward,  # 当前步骤的奖励
            "facility_count": self.n,  # 设施数量
            "action_taken": action_name,  # 执行的动作名称
            "layout_dimensions": (self.H, self.W),  # 布局的长和宽
        }
        return (
            self.state,
            reward,
            False,
            info,
        )

    def render(self):
        # 创建图形和坐标轴
        fig, ax = plt.subplots()
        ax.set_title("Facility layout")
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")
        ax.set_xlim(0, self.W)
        ax.set_ylim(0, self.H)
        plt.grid(False)
        plt.gca().set_aspect("equal", adjustable="box")

        # 绘制设施矩形
        for i, facility_label in enumerate(self.fbs_model.permutation):
            facility_idx = facility_label - 1  # 设施索引从0开始
            x_from = self.fac_x[facility_idx] - self.fac_b[facility_idx] / 2
            x_to = self.fac_x[facility_idx] + self.fac_b[facility_idx] / 2
            y_from = self.fac_y[facility_idx] - self.fac_h[facility_idx] / 2
            y_to = self.fac_y[facility_idx] + self.fac_h[facility_idx] / 2

            # 边框颜色表示长宽比状态
            line_color = "red" if self.fac_aspect_ratio[facility_idx] > self.fac_limit_aspect else "green"
            
            # 获取设施的状态特征
            permutation = np.array(self.fbs_model.permutation)
            sources = np.sum(self.TM, axis=1)
            sinks = np.sum(self.TM, axis=0)
            
            # 归一化为RGB值
            if np.max(permutation) != np.min(permutation):
                R = (permutation[i] - np.min(permutation)) / (np.max(permutation) - np.min(permutation))
            else:
                R = 0.5
                
            if np.max(sources) != np.min(sources):
                G = (sources[facility_idx] - np.min(sources)) / (np.max(sources) - np.min(sources))
            else:
                G = 0.5
                
            if np.max(sinks) != np.min(sinks):
                B = (sinks[facility_idx] - np.min(sinks)) / (np.max(sinks) - np.min(sinks))
            else:
                B = 0.5
                
            face_color = (R, G, B, 0.7)

            rect = patches.Rectangle(
                (x_from, y_from),
                width=x_to - x_from,
                height=y_to - y_from,
                edgecolor=line_color,
                facecolor=face_color,  # 填充颜色
                linewidth=1
            )
            ax.add_patch(rect)

            # 显示设施ID
            ax.text(
                x_from + (x_to - x_from) / 2,
                y_from + (y_to - y_from) / 2,
                f"{int(facility_label)}",
                ha="center",
                va="center",
                color="white" if np.mean(face_color[:3]) < 0.5 else "black"  # 自适应文字颜色
            )

        # 显示MHC和Fitness
        plt.figtext(0.5, 0.93, f"MHC: {self.MHC:.2f}", ha="center", fontsize=12)
        plt.figtext(0.5, 0.96, f"Fitness: {FBSUtil.getFitness(self.MHC, self.fac_b, self.fac_h, self.fac_limit_aspect):.2f}", ha="center", fontsize=12)

        plt.show()
    def constructState(self):
        """
        构建适合MLP处理的1D状态表示
        """
        # 提取设施相关数据
        permutation = np.array(self.fbs_model.permutation)
        sources = np.sum(self.TM, axis=1)
        sinks = np.sum(self.TM, axis=0)
        
        # 归一化为0-1范围
        if np.max(permutation) != np.min(permutation):
            norm_permutation = (permutation - np.min(permutation)) / (np.max(permutation) - np.min(permutation))
        else:
            norm_permutation = np.ones(self.n) * 0.5
            
        if np.max(sources) != np.min(sources):
            norm_sources = (sources - np.min(sources)) / (np.max(sources) - np.min(sources))
        else:
            norm_sources = np.ones(self.n) * 0.5
            
        if np.max(sinks) != np.min(sinks):
            norm_sinks = (sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks))
        else:
            norm_sinks = np.ones(self.n) * 0.5
            
        # 将坐标、尺寸和流量数据合并为一维数组
        state_components = [
            norm_permutation,  # 设施ID
            self.fac_x / self.W,  # 归一化x坐标
            self.fac_y / self.H,  # 归一化y坐标
            self.fac_b / self.W,  # 归一化宽度
            self.fac_h / self.H,  # 归一化高度
            norm_sources,  # 归一化源流量
            norm_sinks,  # 归一化汇流量
            self.fac_aspect_ratio / self.fac_limit_aspect  # 归一化横纵比
        ]
        
        # 合并为一维状态向量
        state_vector = np.concatenate(state_components)
        
        # 确保observation_space匹配
        if hasattr(self, 'observation_space') and self.observation_space.shape != (len(state_vector),):
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(len(state_vector),), dtype=np.float32
            )
        
        return state_vector.astype(np.float32)