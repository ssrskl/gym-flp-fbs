from gym.envs.registration import register

register(
    id="FbsEnv-v0",
    entry_point="FbsEnv.envs.FbsEnv:FBSEnv",
)
