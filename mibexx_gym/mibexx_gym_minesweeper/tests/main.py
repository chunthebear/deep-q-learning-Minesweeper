import gym

env = gym.make('mibexx_gym_minesweeper:mibexx-gym-minesweeper-v0')
env.reset()

print("Observation space: ", env.observation_space.shape)
print("Action space: ", env.action_space.nvec)

state, reward, done, _ = env.step((0, 0))
print(f"Reward: {reward} Done: {done}")
state, reward, done, _ = env.step((0, 0))
print(f"Reward: {reward} Done: {done}")

env.render()