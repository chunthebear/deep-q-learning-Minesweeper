from gym.envs.registration import register

register(
    id='mibexx-gym-minesweeper-v0',
    entry_point='mibexx_gym_minesweeper.envs:MinesweeperEnv'
)