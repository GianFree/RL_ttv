"""
I'm only Sleeping
"""
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

print(torch.__version__)
a = np.linspace(0,10,1000)
b = np.cos(a)

plt.plot(a,b)
plt.set_title('riciu')
plt.show()
