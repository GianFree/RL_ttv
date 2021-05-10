import gym
import numpy as np


env = gym.make("MountainCar-v0")
env.reset()

env_high = env.observation_space.high
env_low = env.observation_space.low

N_grid = 10

grid_size = [N_grid]*len(env_high)
discrete_size  = (env_high - env_low)/grid_size
discrete_size


done = False
while not done:
    # TODO
    # - come scegliere l'azione
    # - esecuzione dell'azione
    # - calcolo ed update reward
    #print(env.action_space.sample( ))
    state, reward, done, _ = env.step(2)
    env.render()

env.close()
