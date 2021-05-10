import gym
import numpy as np


env = gym.make("MountainCar-v0")
env.reset()

# Dimension of the environment
env_high = env.observation_space.high
env_low = env.observation_space.low

# Discretisation
N_grid = 10
grid_size = [N_grid]*len(env_high)
n_actions = [env.action_space.n]
discrete_size  = (env_high - env_low)/grid_size


q_table = -np.ones((grid_size + [n_actions]), float)
q_table.shape
# Training
done = False
while not done:
    # TODO
    # - come scegliere l'azione
    # - esecuzione dell'azione
    # - calcolo ed update reward
    #print(env.action_space.sample( ))
    state, reward, done, _ = env.step(0)
    print(state, reward, done)
    env.render()

env.close()
