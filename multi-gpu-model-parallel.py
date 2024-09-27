import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

class ModelPart1(nn.Module):
    def __init__(self):
        super(ModelPart1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        return x

class ModelPart2(nn.Module):
    def __init__(self):
        super(ModelPart2, self).__init__()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(64, -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using devices: {device1}, {device2}")

model_part1 = ModelPart1().to(device1)
model_part2 = ModelPart2().to(device2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    list(model_part1.parameters()) + list(model_part2.parameters()),
    lr=0.001, momentum=0.9
)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=True, download=True, 
                                 transform=transforms.ToTensor()),
    batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

time_start = time.time()
for epoch in range(10):
    running_loss = 0.0
    running_corrects = 0
    for i, (inputs, labels) in enumerate(train_loader):
        # print(f"Batch {i}: Input shape: {inputs.shape}, Labels shape: {labels.shape}")
        
        inputs = inputs.to(device1)
        labels = labels.to(device2)
        
        # print(f"Batch {i}: Inputs device: {inputs.device}, Labels device: {labels.device}")

        optimizer.zero_grad()

        intermediates = model_part1(inputs)
        # print(f"Batch {i}: Intermediates shape: {intermediates.shape}, device: {intermediates.device}")

        intermediates = intermediates.to(device2)
        # print(f"Batch {i}: Intermediates after moving to device2: shape: {intermediates.shape}, device: {intermediates.device}")

        outputs = model_part2(intermediates)
        # print(f"Batch {i}: Outputs shape: {outputs.shape}, device: {outputs.device}")
        # print(f"Batch {i}: Labels shape: {labels.shape}, device: {labels.device}")

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        if i == 0:  # Only print for the first batch
            break  # Exit after the first batch to avoid flooding the output

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.float() / len(train_loader.dataset)
    print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch+1, 10, epoch_loss, epoch_acc))
    
    
print(f"Total training time: {time.time() - time_start:.2f} seconds")