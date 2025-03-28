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
# 安装loguru，日志模块
pip install loguru
# 安装colorlog，彩色日志
pip install colorlog
# 安装ipykernel，用于在jupyter notebook中使用自定义的gym环境
pip install ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple
# 安装openpyxl，用于读取excel文件
pip install openpyxl
```

## 使用

首先需要安装好自定义的 gym 环境，在根目录下运行

```bash
pip install -e .
```

安装好之后就可以正常使用 gym 的环境了

## 问题解决

报错：`AttributeError: module 'numpy' has no attribute 'bool8'. Did you mean: 'bool'?`
参考：[How to fix nptyping AttributeError: module 'numpy' has no attribute 'bool8'. Did you mean: 'bool'?
](https://techoverflow.net/2024/09/20/how-to-fix-nptyping-attributeerror-module-numpy-has-no-attribute-bool8-did-you-mean-bool/)

```bash
pip install -U nptyping
```

## 易错点

### 函数参数包含可变类型

当函数参数包含可变类型时，需要注意

```python
def func(n,a: list[int]=[]):
    a.append(n)
    return a
func1 = func(1) # [1]
func2 = func(2) # [1,2]
```

可以稍做改变

```python
def func(n,a: list[int]=None):
    if a is None:
        a = []
    a.append(n)
    return a
```

当把字典，列表等传递给函数的时候，需要注意

```python
def func(lst):
    lst2 = lst
    lst2.append(1)
    return lst2
lst = [1,2,3]
func(lst) # lst = [1,2,3,1]
```

稍微不注意就会导致列表被修改，所以可以稍做改变

```python
def func(lst):
    lst2 = lst[:] # lst.copy()
    lst2.append(1)
    return lst2
lst = [1,2,3]
func(lst) # lst = [1,2,3,1]
```

## 代码编写，注意要点

- 函数包含可变类型参数的时候，如果没有返回值，可以改变原参数，如果有返回值，则需要返回一个深拷贝的参数，防止原参数被修改
