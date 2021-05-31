# Task: classify images based on the image itself

# import dataset : FashionMNIST

# 1. transform and split the dataset
#   - transform: join all the images and convert them into a tensor
#   - normalization: the images whose pixel goes from 0 to 255 into 0.0 to 1.0
#   - split dataset: training set and test set;
#                    - training set: to train the model and change "some coefficient"
#                    - test set: to check how the model works on data it has never seen
#
#
# 2. model creation: Neural network for classification.
#   - convolutions layers: to extract relevant features from images to obtain
#                          other "features"
#   - flattening: from 2d to 1d
#   - n layers `Dense` with X neurons
#   - 1 layer `Dense` with # of classes
#
#2b.- loss function: to compare the label with the computed value
#   - optimizer: to backpropagate the info obtained by the loss-function
#
#
# 3. training the model
#
# 4. test the model

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import seaborn as sns
import torchvision
from torchvision import transforms
# to import layers
import torch.nn as nn
# to import activation functions
import torch.nn.functional as fcn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

transf_ = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.5, std=0.5)])

batch_size = 8
train_set = datasets.FashionMNIST(root="data",download=True, transform=transf_)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = datasets.FashionMNIST(root="data",download=True,train=False, transform=transf_)
test_loader = DataLoader(test_set, batch_size=batch_size)


#datait_ = iter(train_loader)
#images, labels = datait_.next()
#plt.imshow(images.squeeze())
#sns.heatmap(images.squeeze(), square=True)

# creaton of the model:
class BUBU(nn.Module):
    def __init__(self):
        super(BUBU, self).__init__()
        # processing 2d image
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=2, padding=1)
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=2, padding=1)
        self.pool = nn.MaxPool2d((2,2))
        # linear layers
        self.fc1 = nn.Linear(in_features=7*7*32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64,  out_features=10)
        #self.fc5 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(fcn.relu(self.conv1(x)))
        x = self.pool(fcn.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = fcn.relu(self.fc1(x))
        x = fcn.relu(self.fc2(x))
        x = fcn.relu(self.fc3(x))
        x = fcn.relu(self.fc4(x))
        #x = self.fc5(x)
        return x


net = BUBU()
print(net)

loss_ = nn.CrossEntropyLoss()
#opt_  = optim.SGD(net.parameters(),lr = 1e-3,momentum = 0.9)
opt_  = optim.Adam(net.parameters(),lr = 1e-3)



### Training loop
def training_routine(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        #prediction
        pred = model(X)
        loss = loss_fn(pred,y)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss


def test_routine(test_dataloader, model, loss_fn):
    size = len(test_dataloader.dataset)
    test_loss, correct = 0,0

    with torch.no_grad():
        for X,y in test_dataloader:
            # prediction
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(f"Test Error:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct*100


n_epochs = 30
test_losses = []
train_losses = []
for t in range(n_epochs):
    print(f"Epoch {t+1}\n")
    training_routine(train_loader, net, loss_, opt_)
    test_losses.append(test_routine(test_loader, net, loss_))

print("Done!")

torch.save(net.state_dict(), "adam_fashionMnist_statedict.pth")
#device = torch.device('cpu')
#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH, map_location=device))
# or
#device = torch.device("cuda")
#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH))
#model.to(device)
torch.save(net, "adam_fashionMnist_model.pth")
# Model class must be defined somewhere
#model = torch.load(PATH)
#model.eval()

#plt.plot(train_losses, label="training set")
plt.plot(test_losses, label="test set")
plt.legend(loc='best')
plt.show()
