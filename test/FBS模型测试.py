from FbsEnv.envs.FBSModel import FBSModel
import logging

logging.basicConfig(level=logging.INFO)

permutation = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bay = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fbs_model = FBSModel(permutation, bay)
logging.info(fbs_model.permutation)
logging.info(fbs_model.bay)

fbs_model.permutation = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
fbs_model.bay = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
logging.info(fbs_model.permutation)
logging.info(fbs_model.bay)


# second将会和fbs_model指向同一个对象
logging.info(f"fbs_model: {fbs_model}")
logging.info(f"fbs_model.permutation: {fbs_model.permutation}")
logging.info(f"fbs_model.bay: {fbs_model.bay}")
second_fbs_model = fbs_model
logging.info(f"second_fbs_model: {second_fbs_model}")
logging.info(f"second_fbs_model.permutation: {second_fbs_model.permutation}")
logging.info(f"second_fbs_model.bay: {second_fbs_model.bay}")
second_fbs_model.permutation = [3, 2, 1]
second_fbs_model.bay = [3, 2, 1]
logging.info(f"second_fbs_model: {second_fbs_model}")
logging.info(f"second_fbs_model.permutation: {second_fbs_model.permutation}")
logging.info(f"second_fbs_model.bay: {second_fbs_model.bay}")
logging.info(f"fbs_model: {fbs_model}")
logging.info(f"fbs_model.permutation: {fbs_model.permutation}")
logging.info(f"fbs_model.bay: {fbs_model.bay}")

logging.info(f"fbs_model == second_fbs_model: {fbs_model == second_fbs_model}")
