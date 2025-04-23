# gym-flp-fbs

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Gym-0.21.0+-green.svg" alt="Gym Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## 项目概述

gym-flp-fbs 是一个基于 OpenAI Gym 框架的工厂布局问题 (FLP) 解决方案，专注于柔性湾结构 (FBS, Flexible Bay Structure) 布局方式的无约束区域设施布局问题 (UAFLP)。本项目结合了强化学习与启发式算法，通过自定义的环境和模型，提供了一种创新的解决复杂布局优化问题的方法。

### 核心特点

- 📊 基于 Gym 和 Stable-Baselines3 框架设计的自定义环境
- 🧠 结合强化学习与遗传算法解决复杂布局问题
- 📐 柔性湾结构 (FBS) 布局模型实现
- 🔄 多种启发式算子支持动态布局优化
- 📈 内置可视化功能与详细的评估机制

## 环境配置

### 基础依赖

```bash
# 安装PyTorch (根据您的CUDA版本可能需要调整)
pip3 install torch torchvision torchaudio

# 核心框架和工具
pip install gym numpy pygame matplotlib
pip install stable-baselines3[extra]

# 日志工具
pip install loguru colorlog

# 工具库
pip install ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openpyxl
```

### 安装本项目

在克隆仓库后，执行以下命令来安装自定义 Gym 环境：

```bash
# 在项目根目录下
pip install -e .
```

安装完成后，您可以直接使用 `gym.make("FbsEnv-v0")` 来创建和使用自定义环境。

## 项目结构

```
gym-flp-fbs/
│
├── FbsEnv/                  # 核心环境包
│   ├── envs/                # 环境实现
│   │   ├── FbsEnv.py        # 主环境类
│   │   └── FBSModel.py      # FBS模型定义
│   ├── utils/               # 工具函数
│   └── files/               # 数据文件
│
├── Algorithm/               # 算法实现
│   ├── GA.py                # 遗传算法实现
│   ├── GAAlgorithm.py       # GA结合强化学习
│   ├── SAAlgorithm.py       # 模拟退火算法
│   └── RL/                  # 强化学习算法
│
├── test/                    # 测试脚本
│   ├── 基础模型训练.py       # 模型训练示例
│   ├── 基础模型测试.py       # 模型测试示例
│   └── ...                  # 其他测试脚本
│
├── Files/                   # 结果和数据存储
│   ├── ExpResult/           # 实验结果
│   └── SolutionSet/         # 解决方案集
│
├── models/                  # 训练好的模型
├── requirements.txt         # 项目依赖
└── setup.py                 # 安装脚本
```

## FBS模型介绍

柔性湾结构 (Flexible Bay Structure, FBS) 是一种常用于设施布局问题的结构化表示方法，特别适用于工厂车间布局。

### FBS模型的核心概念

- **置换序列 (Permutation)**: 表示设施的排列顺序
- **湾结构 (Bay)**: 定义设施的分组方式，决定湾的划分位置
- **二维数组表示**: 根据置换序列和湾结构动态计算的二维布局结构

### 模型特点

- 灵活的设施布局表示方法
- 高效的空间利用
- 符合实际工厂布局原则

## 使用方法

### 创建和使用环境

```python
import gym
import FbsEnv
from FbsEnv.envs.FBSModel import FBSModel

# 创建环境实例
env = gym.make("FbsEnv-v0", instance=0)  # 选择特定的布局问题实例

# 重置环境，获取初始状态
observation = env.reset()

# 与环境交互
action = env.action_space.sample()  # 随机采样一个动作
observation, reward, done, info = env.step(action)

# 可视化当前布局
env.render()
```

### 使用遗传算法优化布局

```python
from Algorithm.GAAlgorithm import GAAlgorithm

# 创建GA算法实例
ga = GAAlgorithm(instance=0, population_size=50, generations=100)

# 运行算法
best_solution = ga.run()
print(f"最佳适应度值: {best_solution.fitness}")
```

### 结合强化学习训练模型

```python
from stable_baselines3 import DQN

# 创建环境
env = gym.make("FbsEnv-v0", instance=0)

# 创建和训练DQN模型
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 保存模型
model.save("dqn_fbs_model")
```

## 问题解决

### 常见错误

1. **NumPy属性错误**:
   
   报错：`AttributeError: module 'numpy' has no attribute 'bool8'. Did you mean: 'bool'?`
   
   解决方案：
   ```bash
   pip install -U nptyping
   ```

2. **环境初始化问题**:
   
   如果遇到环境初始化错误，请检查数据文件路径是否正确，特别是实例数据的位置。

3. **布局约束违反**:
   
   当横纵比超出限制时，可以使用 repair 功能进行修复：
   ```python
   repaired_solution = FBSUtil.repair(solution)
   ```

## 编程注意事项

### 函数参数设计

#### 可变类型参数处理

当函数参数包含可变类型时，需要注意默认参数的陷阱：

```python
# 错误方式
def func(n, a: list[int]=[]):
    a.append(n)
    return a

# 正确方式
def func(n, a: list[int]=None):
    if a is None:
        a = []
    a.append(n)
    return a
```

#### 对象引用与深拷贝

传递可变对象时，需注意引用问题：

```python
# 潜在问题代码
def func(lst):
    lst2 = lst  # 浅拷贝，会修改原始列表
    lst2.append(1)
    return lst2

# 推荐方式
def func(lst):
    lst2 = lst.copy()  # 或使用 lst[:]、deepcopy 等
    lst2.append(1)
    return lst2
```

### 代码编写建议

1. **函数返回值设计**: 
   - 无返回值的函数可以直接修改原参数
   - 有返回值的函数应返回深拷贝的结果，避免修改原参数

2. **对象封装**:
   - 使用类的属性访问器 (getter/setter) 管理数据，确保数据一致性
   - 利用属性装饰器实现动态计算属性

3. **状态管理**:
   - 保持环境状态的一致性，特别是在更新布局后
   - 使用 `reset()` 方法重置环境到初始状态

## 贡献指南

欢迎对本项目提出改进建议或直接贡献代码。请确保您的代码符合项目的编码风格，并通过添加适当的测试来验证功能。

## 许可证

本项目基于 MIT 许可证开源。
