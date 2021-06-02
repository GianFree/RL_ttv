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

from itertools import combinations
import torch
from torch.functional import Tensor
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as fnc
import torch.optim as optim
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import pandas as pd


# Using cuda or not
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


# Plotting environment
sns.set_style("darkgrid") 
sns.set_palette(sns.color_palette())


# Creating and cleaning folder for images
outdir_name = 'out_images'
try:
    os.mkdir(outdir_name)
except FileExistsError:
    print("Directory already created")

for f in glob.glob(f"{outdir_name}/*.png"):
    os.remove(f)


# 1. Defining the input set (and allowing gradients)
def f(x):
    """Unknown function"""
    return x**3


x = torch.unsqueeze(torch.linspace(-3, 3, 500, requires_grad=True, device=device),dim=1)
y = f(x) + 3*torch.rand(x.size(), device=device)

if device == 'cuda':
    x_cpu = x.cpu()
    y_cpu = y.cpu()
 


# Plotting for debugging purpose
# sns.scatterplot(x=x_cpu.data.numpy().squeeze(),y=y_cpu.data.numpy().squeeze())

# 2. Defining the neural network
class Regression(nn.Module):
    def __init__(self, hidden_feat1=200, hidden_feat2=100, hidden_feat3=50):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(in_features=1,            out_features=hidden_feat1)
        self.fc2 = nn.Linear(in_features=hidden_feat1, out_features=hidden_feat2)
        self.fc3 = nn.Linear(in_features=hidden_feat2, out_features=hidden_feat3)
        self.fc4 = nn.Linear(in_features=hidden_feat3,  out_features=1)


    def forward(self, x):
        x = fnc.leaky_relu(self.fc1(x))
        x = fnc.leaky_relu(self.fc2(x))
        x = fnc.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


# For loop
neurons_list = [200, 150, 100, 75, 50, 25]
loss_dict = {}
for hid1, hid2, hid3 in combinations(neurons_list, 3):
    print("init model")
    regression_net = Regression(hidden_feat1=hid1, hidden_feat2=hid2, hidden_feat3=hid3)
    if device == 'cuda':
        regression_net.cuda()

    # Loss function - Do we need to specify a `reduction` as keyarg?
    loss_function = nn.MSELoss()
    # optimizer
    optimizer = optim.SGD(regression_net.parameters(),lr = 1e-3)

    # 3. Training the Network
    epochs = 1000
    loss_history = []

    for epoch in range(epochs):
        predictions = regression_net(x)
        loss = loss_function(predictions,y)

        optimizer.zero_grad()
        if epoch == (epochs-1):
            loss.backward(retain_graph=True)
        else:
            loss.backward(retain_graph=True)


        optimizer.step()
        if epoch%250 == 0:
            print(f"Epoch {epoch} with Loss: {loss.item()}")

        loss_history.append(loss.item())


        if False:
            if device == 'cuda':
                predictions_cpu = predictions.cpu()
            if epoch == 0:
                old_predictions = predictions_cpu  # to provide shading


            plt.clf()
            sns.scatterplot(x=x_cpu.data.numpy().squeeze(), y=y_cpu.data.numpy().squeeze(), label="$x^3+\eta_{noise}$", s=20, palette='deep')
            sns.lineplot(x=x_cpu.data.numpy().squeeze(), y=predictions_cpu.data.numpy().squeeze(), label="$f_{approx}(x)$", color=sns.color_palette()[3], palette='deep')
            # for shading
            sns.lineplot(x=x_cpu.data.numpy().squeeze(), y=old_predictions.data.numpy().squeeze(), color=sns.color_palette()[3], palette='deep', alpha=0.2)
            plt.text(-0.97, 25., f"MSE: {loss.item():>6.3f}", horizontalalignment='left', color='black', weight='semibold')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title("ANN w/ Leaky ReLu activation")
            plt.legend()
            plt.savefig(f"{outdir_name}/regression_{epoch}.png")
            #plt.show()

            # Storing old predictions
            if epoch %10 == 0:
                old_predictions = predictions_cpu

    loss_dict[f"{hid1}-{hid2}-{hid3}"] = loss_history
    del regression_net

df = pd.DataFrame.from_dict(loss_dict)
df.to_csv("regression_hyperparams_nn.csv")

plt.clf()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Cubic Function")
plt.show()
#while True:
#    user_input = input("Tell me a number")
#    if user_input == 'q':
#        break
#    user_input = torch.float(user_input)
#    y_pred = regression_net(torch.unsqueeze(user_input, dim=1))
#    print(f"Correct value: {f(user_input)} - Prediction {y_pred.data.numpy().squeeze()}")
 
