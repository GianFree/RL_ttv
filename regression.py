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
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as fnc
import torch.optim as optim
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import sklearn
from sklearn.model_selection import train_test_split




class CustomRegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



sns.set_style("darkgrid")
#sns.set_context("paper")


# Using cuda or not
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


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




x = torch.unsqueeze(torch.linspace(-5, 5, 100, requires_grad=True, device=device),dim=1)
y = f(x) + 10*torch.rand(x.size(), device=device)


dataset = CustomRegressionDataset(x,y)

train, val, test = random_split(dataset, [60,20,20])

train_x = x[train.indices]
train_y = y[train.indices]
val_x = x[val.indices]
val_y = y[val.indices]
test_x = x[test.indices]
test_y = y[test.indices]

train_dataset = CustomRegressionDataset(train_x,train_y)
val_dataset   = CustomRegressionDataset(val_x, val_y)
test_dataset  = CustomRegressionDataset(test_x,test_y)

b_size = 5
train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
val_dataloader   = DataLoader(val_dataset)
test_dataloader  = DataLoader(test_dataset)

if device == 'cuda':
    train_x_cpu = train_x.cpu()
    train_y_cpu = train_y.cpu()
    val_x_cpu   = val_x.cpu()
    val_y_cpu   = val_y.cpu()
    test_x_cpu  = test_x.cpu()
    test_y_cpu  = test_y.cpu()



sns.scatterplot(x=train_x_cpu.data.numpy().squeeze(),y=train_y_cpu.data.numpy().squeeze(), color='C0');
sns.scatterplot(x=val_x_cpu.data.numpy().squeeze(),y=val_y_cpu.data.numpy().squeeze(), color='C1');
sns.scatterplot(x=test_x_cpu.data.numpy().squeeze(),y=test_y_cpu.data.numpy().squeeze(), color='C2');
plt.show()

# 2. Defining the neural network
class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.fc1 = nn.Linear(in_features=1,  out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=50)
        self.fc4 = nn.Linear(in_features=50, out_features=1)


    def forward(self, x):
        x = fnc.leaky_relu(self.fc1(x))
        x = fnc.leaky_relu(self.fc2(x))
        x = fnc.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x



regression_net = Regression()
if device == 'cuda':
    regression_net.cuda()
# Loss function - Do we need to specify a `reduction` as keyarg?
loss_function = nn.MSELoss()
# optimizer
optimizer = optim.SGD(regression_net.parameters(),lr = 1e-3)


### Training loop
def training_routine(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        #prediction
        pred = model(X)
        loss = loss_fn(pred,y)

        #backpropagation
        optimizer.zero_grad()
        if epoch == (epochs-1):
            loss.backward()
        else:
            loss.backward(retain_graph=True)
        optimizer.step()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}")
    return loss


def test_routine(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    test_loss = 0

    with torch.no_grad():
        for X,y in test_dataloader:
            # prediction
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= size

    print(f"Test Error:\n Avg loss: {test_loss:>8f} \n")
    return test_loss



# 3. Training the Network
epochs = 1500
val_loss_history = []
for epoch in range(epochs):
    training_routine(dataloader=train_dataloader, model=regression_net, loss_fn=loss_function, optimizer=optimizer)
    predictions = regression_net(dataset.x)
    loss_fake = loss_function(predictions,y)# for graphics purpose
    val_loss_history.append(test_routine(val_dataloader, model=regression_net, loss_fn=loss_function))

    if device == "cuda":
        predictions_cpu = predictions.cpu()

    plt.clf()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=train_x_cpu.data.numpy().squeeze(), y=train_y_cpu.data.numpy().squeeze(), label="$x^3+\eta_{noise}$", color='C0',alpha = 0.6, ax=ax)
    sns.lineplot(x=x.cpu().data.numpy().squeeze(), y=predictions_cpu.data.numpy().squeeze(), label="$f_{approx}(x)$", color='C3', ax=ax)
    plt.ylim(-150, 150)
    plt.text(-1.97, 101, f"MSE: {loss_fake.item():>6.3f}", horizontalalignment='left', color='black', weight='semibold')
    plt.title("Leaky ReLu activation")
    plt.savefig(f"{outdir_name}/regression_{epoch}.png", dpi=150)
    plt.close()

plt.clf()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Cubic Function")
#while True:
#    user_input = input("Tell me a number")
#    if user_input == 'q':
#        break
#    user_input = torch.float(user_input)
#    y_pred = regression_net(torch.unsqueeze(user_input, dim=1))
#    print(f"Correct value: {f(user_input)} - Prediction {y_pred.data.numpy().squeeze()}")
