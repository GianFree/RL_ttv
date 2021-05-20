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
from torchvision import datasets
import seaborn as sns
from torchvision.transforms import ToTensor, Lambda
# to import layers
import torch.nn as nn
# to import activation functions
import torch.nn.functional as fcn


train_set = datasets.FashionMNIST(root="data",download=True, transform=ToTensor())
test_set = datasets.FashionMNIST(root="data",download=True,train=False)

train_data = train_set.data/255.0
train_labels = train_set.targets
test_data = test_set.data/255.0
test_labels = test_set.targets


sns.heatmap(train_data[1], square=True)
print(train_labels.unique())



# creaton of the model:
class BUBU(nn.Module):
    def __init__(self):
        super(BUBU, self).__init__()
        self.conv1 = nn.Conv2d(1,16,3)
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(16,32,(3,3))
        self.pool = nn.MaxPool2d((2,2))

        #self.fc1 = nn.Linear()
        #self.fc2 = nn.Softmax(10)

    def forward(self, x):
        x = self.pool(fcn.relu(self.conv1(x)))
        x = self.pool(fcn.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        #x = fcn.relu(self.fc1(x))
        #x = self.fc2(x)
        return x


net = BUBU()
print(net)
gneggne = train_set.data.unsqueeze(1)
net(gneggne)
