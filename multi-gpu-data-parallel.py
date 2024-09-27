import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x
    
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
Step 3: Create the data loader and move the model to GPUs. In this example, we will use the CIFAR-10 dataset and split it into batches of 64 images. We will also move the model to GPUs using the nn.DataParallel() module. Essentially, nn.DataParallel() wraps the model, and by doing so, it replicates your model on each GPU, splits the input data, and aggregates the output from each GPU.
"""
model = nn.DataParallel(model, device_ids=[0, 1])
model.to(device)

start = time.time()
for epoch in range(10):
    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
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
    
    print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch+1, 10, epoch_loss, epoch_acc))
    
print(f"Training time: {time.time()-start} seconds")