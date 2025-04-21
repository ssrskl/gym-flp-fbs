# 实验工具
import pandas as pd
import os
import numpy as np
import FbsEnv
import datetime

def save_experiment_result(
    exp_instance, exp_algorithm ,exp_iterations,exp_solution, exp_fitness, 
    exp_start_time,
    exp_fast_time, 
    exp_end_time,
    exp_remark = ""
):
    """
    保存实验结果
    """
    # 保存实验结果
    exp_result = pd.DataFrame(
        {
            "实例": [exp_instance],
            "算法": [exp_algorithm],
            "迭代次数": [exp_iterations],
            "解": [exp_solution],
            "适应度值": [exp_fitness],
            "开始时间": [exp_start_time],
            "最快时间": [exp_fast_time],
            "结束时间": [exp_end_time],
            "运行时间": [(exp_end_time - exp_start_time).total_seconds()],
            "最快最佳结果时间": [(exp_fast_time - exp_start_time).total_seconds()],
            "备注": [exp_remark],
        }
    )
    
    # 确保目录存在
    os.makedirs("/Users/maoyan/Codes/Python/gym-flp-fbs/Files/ExpResult", exist_ok=True)
    
    file_path = f"/Users/maoyan/Codes/Python/gym-flp-fbs/Files/ExpResult/{exp_instance}-{exp_algorithm}.xlsx"
    
    if os.path.exists(file_path):
        # 如果文件存在，则读取文件并追加数据
        existing_data = pd.read_excel(file_path)
        exp_result = pd.concat([existing_data, exp_result], ignore_index=True)
        # 保存并保留表头
        exp_result.to_excel(file_path, index=False)
    else:
        # 如果文件不存在，则创建新文件
        exp_result.to_excel(file_path, index=False)
    
    return exp_result

if __name__ == "__main__":
    # 测试保存实验结果
    exp_instance = "SC35-maoyan"
    exp_algorithm = "模拟退火算法"
    exp_iterations = 100
    exp_solution = np.array2string(np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    ),separator=',')
    exp_fitness = 100
    exp_start_time = datetime.datetime.now()
    exp_fast_time = datetime.datetime.now()
    exp_end_time = datetime.datetime.now()
    exp_result = save_experiment_result(
        exp_instance, exp_algorithm, exp_iterations, exp_solution, exp_fitness, 
        exp_start_time, exp_fast_time, exp_end_time
    )
    print(exp_result)
