import gym

import numpy as np


env = gym.make("MountainCar-v0")
env.reset()


env.render()
env.close()
