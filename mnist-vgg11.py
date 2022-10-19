from cgi import test
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root='./Dataset/mnist', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./Dataset/mnist', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class vgg11(nn.Module):
    def __init__(self):
        super().__init__()
        # (inputsize - kernel_size + 2*padding) / stride + 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1) #28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #14
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) #14
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #7
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, padding=1)#8
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)#8
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) #4
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)#4
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)#4
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)#2
        self.fc1 = nn.Linear(in_features=1*1*512, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)
        self.fc4 = nn.Linear(in_features=1000, out_features=10)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool3(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool4(x)
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv6(x))
        x = self.pool4(x)
        #print(x.shape)
        x = x.view(-1, 1*1*512)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

net = vgg11()
print(net)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
epochs = 10

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = net(images)
        loss_value = loss(outputs, labels)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}'.format(epoch+1, epochs, i, len(train_loader), loss_value.item()))

print(net)
correct = 0
for data in test_loader:
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    correct += (predicted == labels).sum().item()
print('Accuracy: {:.2f}%'.format(100 * correct / len(test_dataset)))