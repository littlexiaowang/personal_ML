import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class Alex_net(nn.Module):
    def __init__(self)->None:
        super().__init__()
        # outputsize(h,w) = (inputsize - 1) * stride - 2 * padding + kernel_size + outputpadding
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernal_size = 7, padding = 1)
        self.pool1 = nn.MaxPool2d(kernal_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 256, kernal_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 384, kernal_size = 3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 384, out_channels = 384, kernal_size = 3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 384, out_channels = 256, kernal_size = 3, padding = 1)
        self.pool2 = nn.MaxPool2d(kernal_size = 2, stride = 2)
        self.pool3 = nn.MaxPool2d(kernal_size = 2, stride = 2)
        self.fc1 = nn.Linear(in_features = 3*3*256, out_features = 1024)
        self.fc2 = nn.Linear(in_features = 1024, out_features = 512)
        self.fc3 = nn.Linear(in_features = 512, out_features = 10)
        self.drop1 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = x.view(-1, 3*3*256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = self.fc3(x)
        return x

net = Alex_net().to(device)
print(net)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
epochs = 1

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss_ = loss(outputs, labels)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, epochs, i+1, len(train_loader), loss_.item()))

torch.save(net, 'mnist_cnn.pth') 
net = torch.load('mnist_cnn.pth') 
print("---------------------------------------------")
print(net)
correct = 0
for data in test_loader:
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == labels).sum().item()
print('Accuracy of the network: {} %'.format(correct / len(test_loader)))