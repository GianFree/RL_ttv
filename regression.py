# Regression:
# approximate an unknown function via Artificial Neural Networks.
#
# 1. Define the "unknown" function and generate data
# 2. Define the ANN
# 3. Training
# 4. Test
#
# Possible upgrade:
# - ANN for house pricing
# - Comparison for stock pricing:
#   - ANN
#   - Long short-term memory
#   - Bayesian approach to NN

import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fnc
import torch.optim as optim
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

outdir_name = 'out_images'
try:
    os.mkdir(outdir_name)
except FileExistsError:
    print("Directory already created")
# 1. Defining the input set (and allowing gradients)
def f(x):
    """Unknown function"""
    return x**3


x = torch.unsqueeze(torch.linspace(-2, 2,100, requires_grad=True),dim=1)
y = f(x) + torch.rand(x.size())


sns.scatterplot(x=x.data.numpy().squeeze(),y=y.data.numpy().squeeze())

# 2. Defining the neural network
class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(in_features=1,  out_features=10)
        #self.fc2 = nn.Linear(in_features=20, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=1)


    def forward(self, x):
        x = fnc.leaky_relu(self.fc1(x))
        #x = fnc.relu(self.fc2(x))
        x = self.fc3(x)
        return x



regression_net = Regression()

# Loss function - Do we need to specify a `reduction` as keyarg?
loss_function = nn.MSELoss()
# optimizer
optimizer = optim.SGD(regression_net.parameters(),lr = 1e-3)

# 3. Training the Network
epochs = 750
loss_history = []
for epoch in range(epochs):
    predictions = regression_net(x)
    loss = loss_function(predictions,y)

    optimizer.zero_grad()
    if epoch == (epochs-1):
        loss.backward()
    else:
        loss.backward(retain_graph=True)


    optimizer.step()
    print(f"Epoch {epoch} with Loss: {loss.item()}")
    loss_history.append(loss.item())

    plt.clf()
    plt.plot(x.data.numpy().squeeze(), y.data.numpy().squeeze(), 'b.')
    plt.plot(x.data.numpy().squeeze(), predictions.data.numpy().squeeze(), 'r-')
    plt.title("Leaky ReLu activation")
    plt.savefig(f"{outdir_name}/regression_{epoch}.png")


while True:
    user_input = input("Tell me a number")
    if user_input == 'q':
        break
    user_input = torch.float(user_input)
    y_pred = regression_net(torch.unsqueeze(user_input, dim=1))
    print(f"Correct value: {f(user_input)} - Prediction {y_pred.data.numpy().squeeze()}")
