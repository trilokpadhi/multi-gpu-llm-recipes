import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed.pipeline.sync as sync
import torchvision
from torchvision import transforms
from time import time

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ).to("cuda:0")
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ).to("cuda:1")
        
    def forward(self, x):
        x = self.stage1(x)
        return self.stage2(x)
    
    
model = CNN()

# 8 equals batch_size/number_of_Gpus
model = sync.Pipe(model, chunks=8)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=True, 
    download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True, num_workers=2, pin_memory=True)


start = time()
for epoch in range(10):
    running_loss = 0.0
    running_corrects = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to("cuda:0"), labels.to("cuda:1")
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(
        epoch + 1, 10, epoch_loss, epoch_acc))
    
print(f"Time taken: {time() - start} seconds")