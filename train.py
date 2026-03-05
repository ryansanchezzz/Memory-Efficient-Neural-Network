#Import Necessary Libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#Load and Normalize Dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

batch_size = 16

#Using Fashion MNIST
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg.transpose((1, 2, 0)))
    plt.show()

# get some random training images

trainiter = iter(trainloader)
images, labels = next(trainiter)


# show images
imshow(torchvision.utils.make_grid(images))


# print labels
print("True Labels:")
print(' '.join(f'{classes[labels[j]]:10s}' for j in range(batch_size)))

#Define a CNN


import torch.nn as nn
import torch.nn.functional as F

#===============================================
#Custom Outer Product Representation of FC
#===============================================
class OuterProductLinear(nn.Module): 
    def __init__(self, in_features, out_features):
        super(OuterProductLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.u = nn.Parameter(torch.randn(out_features, 1) * 0.1)
        self.v = nn.Parameter(torch.randn(1, in_features) *0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
      W = self.u @ self.v
      return x @ W.T + self.bias
#===============================================


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = OuterProductLinear(256, 256)
        self.fc2B = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)






    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2B(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)

#Define a Loss Function and Optimizer
import torch.optim as optim


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)





#Train the Network
epochs = 10

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0

    for inputs, labels in trainloader:

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()





    

        # print statistics
        running_loss += loss.item()

    # Printing the loss for each epoch
    print("Epoch {} -> Loss: {}".format(epoch + 1, running_loss / len(trainloader)))

print('...')
print('Finished Training')

#Save Model
PATH = './fashion_mnist_net.pth'
torch.save(net.state_dict(), PATH)

#Predict on Sample Images
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:10s}' for j in range(16)))

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:10s}'
                              for j in range(16)))

#Test the network on the test data. 
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs

with torch.no_grad():
    for data in testloader:

        images, labels = data
        total += labels.size(0)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()





print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#Analyse Results

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:10s} is {accuracy:.1f} %')
