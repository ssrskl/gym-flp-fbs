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
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.n,3),dtype=np.uint8)  # 状态空间
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

    def reset(self, fbs_model: FBSModel = None):
        if fbs_model is None:
            permutation, bay = FBSUtil.binary_solution_generator(
                self.areas, self.n, self.fac_limit_aspect, self.W)  # 采用k分初始解生成器
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
        logger.debug(f"状态: {self.state}")
        logger.debug("--------------------------------------------------")
        return self.state

    def calculate_reward(self):
        # 计算MHC改善程度
        # mhc_improvement = ((self.previous_MHC - self.MHC) /
        #                    self.previous_MHC if self.previous_MHC else 0)

        # 计算约束违反惩罚
        # aspect_ratio_penalty = sum(
        #     max(0, ar - self.fac_limit_aspect) +
        #     max(0, self.fac_limit_aspect - ar) for ar in self.fac_aspect_ratio)

        # 计算fitness改善程度
        fitness_improvement = ((self.previous_fitness - self.fitness) /
                               self.fitness if self.previous_fitness else 0)

        # # 综合奖励计算
        reward = (
            # 0.4 * mhc_improvement  # MHC改善权重
            1 * fitness_improvement  # 整体fitness改善权重
            # + 0.2 * aspect_ratio_penalty  # 约束违反惩罚权重
        )
        # reward = -self.fitness
        # 适应度和MHC的惩罚
        # reward = self.MHC - self.fitness
        return reward

    def step(self, action):
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
        reward = self.calculate_reward()
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
            
            # 填充颜色表示成本（RGB）
            R = self.state[facility_idx, 0] / 255
            G = self.state[facility_idx, 1] / 255
            B = self.state[facility_idx, 2] / 255
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
        state = np.zeros((self.n, 3))
        permutation = self.fbs_model.permutation
        TM = self.TM
        sources = np.sum(TM, axis=1)
        sinks = np.sum(TM, axis=0) 
        R = np.array(
            ((permutation - np.min(permutation)) / (np.max(permutation) - np.min(permutation)))
            * 255
        ).astype(np.uint8)
        G = np.array(
            ((sources - np.min(sources)) / (np.max(sources) - np.min(sources)))
            * 255
        ).astype(np.uint8)
        B = np.array(
            ((sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks)))
            * 255
        ).astype(np.uint8)
        state[:, 0] = R
        state[:, 1] = G
        state[:, 2] = B
        return state