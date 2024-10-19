import pickle
import gym
from gym import spaces
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import logging
import colorlog
import FbsEnv.utils.FBSUtil as FBSUtil

# 设置中文
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
# 设置日志
# Create a color formatter
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

# Set up the handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Configure the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


class FBSEnv(gym.Env):
    def __init__(self, instance=None, seed=None, options=None):
        with open(
            "E://projects//pythonprojects//gym-flp-fbs//FbsEnv//files//maoyan_cont_instances.pkl",
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
        self.F = self.FlowMatrices[self.instance]  # 物流强度矩阵
        self.n = self.problems[self.instance]  # 问题模型的设施数量
        self.fac_limit_aspect, self.l, self.w, self.area, self.min_side_length = (
            FBSUtil.getAreaData(self.sizes[self.instance])
        )
        self.L = self.LayoutLengths[self.instance]  # 厂房的
        self.W = self.LayoutWidths[self.instance]  # 厂房的宽度
        total_area = np.sum(self.area)  # 设施的总面积
        self.actions = {
            0: "facility_swap_single",
            1: "shuffle_single",
            2: "facility_swap",
            3: "bay_flip",
            4: "bay_swap",
            5: "bay_shuffle",
            6: "facility_shuffle",
            7: "permutation_shuffle",
            # 8: "repair",
        }  # 动作空间
        self.action_space = spaces.Discrete(len(self.actions))  # 动作空间

        # TODO 需要归一化处理
        # 定义观察空间
        facility_distance = spaces.Box(
            low=0, high=np.inf, shape=(self.n, self.n), dtype=np.float32
        )  # 设施距离矩阵（nxn）
        facility_arrangement = spaces.Box(
            low=0, high=np.inf, shape=(2, self.n), dtype=np.int32
        )  # 设施排列（nx2）
        facility_information = spaces.Box(
            low=0, high=np.inf, shape=(5, self.n), dtype=np.float32
        )  # 设施信息（长，宽，面积，坐标x和y）（nx5）
        self.observation_space = spaces.Dict(
            {
                "facility_distance": facility_distance,
                "facility_arrangement": facility_arrangement,
                "facility_information": facility_information,
            }
        )

    def reset(self, layout=None):
        if layout is not None:
            self.permutation, self.bay = layout
        else:
            self.permutation, self.bay = FBSUtil.binary_solution_generator(
                self.area, self.n, self.fac_limit_aspect, self.L
            )  # 采用k分初始解生成器
            self.bay[-1] = 1  # bay的最后一个位置必须是1，表示最后一个设施是bay的结束
        (
            self.fac_x,
            self.fac_y,
            self.fac_b,
            self.fac_h,
            self.fac_aspect_ratio,
            self.D,
            self.TM,
            self.MHC,
            self.Fitness,
        ) = FBSUtil.StatusUpdatingDevice(
            self.permutation, self.bay, self.area, self.W, self.F, self.fac_limit_aspect
        )
        self.previous_fitness = self.Fitness  # 初始化上一次的适应度值
        # 需要封装state
        self.state = {
            "facility_distance": self.D,
            "facility_arrangement": np.array([self.permutation, self.bay]),
            "facility_information": np.array(
                [
                    self.fac_h,
                    self.fac_b,
                    self.area,
                    self.fac_x,
                    self.fac_y,
                ]
            ),
        }
        return self.state

    def step(self, action):
        # 根据action执行相应的操作
        action_name = self.actions[int(action)]
        if action_name == "facility_swap_single":
            self.permutation, self.bay = FBSUtil.facility_swap_single(
                self.permutation, self.bay
            )
        elif action_name == "shuffle_single":
            self.permutation, self.bay = FBSUtil.shuffle_single(
                self.permutation, self.bay
            )
        elif action_name == "facility_swap":
            self.permutation, self.bay = FBSUtil.facility_swap(
                self.permutation, self.bay
            )
        elif action_name == "bay_flip":
            self.permutation, self.bay = FBSUtil.bay_flip(self.permutation, self.bay)
        elif action_name == "bay_swap":
            self.permutation, self.bay = FBSUtil.bay_swap(self.permutation, self.bay)
        elif action_name == "bay_shuffle":
            self.permutation, self.bay = FBSUtil.bay_shuffle(self.permutation, self.bay)
        elif action_name == "facility_shuffle":
            self.permutation, self.bay = FBSUtil.facility_shuffle(
                self.permutation, self.bay
            )
        elif action_name == "permutation_shuffle":
            self.permutation, self.bay = FBSUtil.permutation_shuffle(
                self.permutation, self.bay
            )
        else:
            raise ValueError(f"Invalid action: {action_name}")
        # 刷新状态
        (
            self.fac_x,
            self.fac_y,
            self.fac_b,
            self.fac_h,
            self.fac_aspect_ratio,
            self.D,
            self.TM,
            self.MHC,
            self.Fitness,
        ) = FBSUtil.StatusUpdatingDevice(
            self.permutation, self.bay, self.area, self.W, self.F, self.fac_limit_aspect
        )
        # 更新状态
        self.state = {
            "facility_distance": self.D,
            "facility_arrangement": np.array([self.permutation, self.bay]),
            "facility_information": np.array(
                [
                    self.fac_h,
                    self.fac_b,
                    self.area,
                    self.fac_x,
                    self.fac_y,
                ]
            ),
        }
        # 计算奖励函数
        reward = self.Fitness - self.previous_fitness
        self.previous_fitness = self.Fitness
        return self.state, reward, False, {}

    def render(self):
        # 创建图形和坐标轴
        fig, ax = plt.subplots()
        ax.set_title("设施布局图")
        ax.set_xlabel("X轴")
        ax.set_ylabel("Y轴")
        # 设置坐标范围
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.W)
        # 添加网格
        plt.grid(False)
        plt.gca().set_aspect("equal", adjustable="box")
        for i, facility_label in enumerate(self.permutation):
            x_from = self.fac_x[facility_label - 1] - self.fac_b[facility_label - 1] / 2
            x_to = self.fac_x[facility_label - 1] + self.fac_b[facility_label - 1] / 2
            y_from = self.fac_y[facility_label - 1] - self.fac_h[facility_label - 1] / 2
            y_to = self.fac_y[facility_label - 1] + self.fac_h[facility_label - 1] / 2
            line_color = "black"
            if (
                self.fac_aspect_ratio[facility_label - 1]
                > self.fac_limit_aspect[facility_label - 1][1]
            ):
                line_color = "red"
            else:
                line_color = "green"
            rect = patches.Rectangle(
                (x_from, y_from),
                width=x_to - x_from,
                height=y_to - y_from,
                edgecolor=line_color,
                facecolor="none",
                linewidth=0.5,
                angle=0.5,
            )
            ax.add_patch(rect)
            # 显示设施ID
            ax.text(
                x_from + (x_to - x_from) / 2,
                y_from + (y_to - y_from) / 2,
                # f"{int(label)}, AR={aspect_ratio[i]:.2f}",
                f"{int(facility_label)}",
                ha="center",
                va="center",
            )

        # 显示MHC
        plt.figtext(
            0.5,
            0.93,
            "MHC: {:.2f}".format(FBSUtil.getMHC(self.D, self.F, self.permutation)),
            ha="center",
            fontsize=12,
        )
        # 显示Fitness
        plt.figtext(
            0.5,
            0.96,
            "Fitness: {:.2f}".format(
                FBSUtil.getFitness(
                    self.MHC, self.fac_b, self.fac_h, self.fac_limit_aspect
                )
            ),
            ha="center",
            fontsize=12,
        )
        # 显示图形
        plt.show()
