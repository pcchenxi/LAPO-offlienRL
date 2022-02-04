from gym.envs.registration import register

register(
    id='multi-walker2d-forward-v1',
    entry_point='multitask_env.envs:Walker2dForwardEnv',
    max_episode_steps=1000,
)
register(
    id='multi-walker2d-backward-v1',
    entry_point='multitask_env.envs:Walker2dBackwardEnv',
    max_episode_steps=1000,
)
register(
    id='multi-walker2d-jump-v1',
    entry_point='multitask_env.envs:Walker2dJumpEnv',
    max_episode_steps=1000,
)