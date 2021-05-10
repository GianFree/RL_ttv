import gym

import numpy as np


env = gym.make("MountainCar-v0")
env.reset()

for i in range(10):
    print(env.action_space.sample())
    #env.render()
    #env.step(2)

env.close()
