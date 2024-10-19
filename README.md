# gym-flp-fbs

## 介绍

使用 gym 框架结合 stable-baselines3 框架，并搭建对应的 FBS 模型，使用强化学习结合启发式算法来解决使用 FBS 的布局方式的 UAFLP 问题。

## 环境配置

```bash
pip3 install torch torchvision torchaudio
pip install gym
pip install numpy pygame
pip install matplotlib
pip install stable-baselines3[extra]
# 安装colorlog，彩色日志
pip install colorlog
# 安装ipykernel，用于在jupyter notebook中使用自定义的gym环境
pip install ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 使用

首先需要安装好自定义的 gym 环境，在根目录下运行

```bash
pip install -e .
```

安装好之后就可以正常使用 gym 的环境了
