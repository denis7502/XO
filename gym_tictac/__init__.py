from gym.envs.registration import register

register(
    id='tictac-v1',
    entry_point='gym_tictac.envs:TicTacEnv',
)
