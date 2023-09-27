import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


train_dataset_path = "training\\training"
test_dataset_path = "validation\\validation"

mean = torch.tensor([0.4363, 0.4328, 0.3291])
std = torch.tensor([0.2137, 0.2083, 0.2046])

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transforms)

"""train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)

def get_mean_and_std(loader):
    mean = 0
    std = 0
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)
        images = images.view(image_count_in_a_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch
    
    mean /= total_images_count
    std /= total_images_count
    return mean, std

print(get_mean_and_std(train_loader))"""

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

def set_device():
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    return torch.device(dev)

resnet18_model = models.resnet18(pretrained=True)
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 10
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
device = set_device()
resnet18_model = resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss()

optimiser = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            predicted_correctly_on_epoch += (predicted == labels).sum().item()
    
    epoch_acc = 100.0 * predicted_correctly_on_epoch/ total
    
    print(predicted_correctly_on_epoch, total, epoch_acc)

    return epoch_acc

def save_checkpoint(model, epoch, optimiser, best_acc):
    state = {
        "epoch" : epoch + 1,
        "model" : model.state_dict(),
        "best accuracy" : best_acc,
        "optimiser" : optimiser.state_dict()
    }
    torch.save(state, "model_best_checkpoint.pth.tar")



def train_nn(model, train_loader, test_loader, criterion, optimiser, n_epochs):
    device = set_device()
    best_acc = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0 
        running_correct = 0
        total = 0
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimiser.zero_grad()

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()
        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100*(running_correct/total)
        print("running loss %d, running accuracy %d, epoch number %d"%(epoch_loss, epoch_acc, epoch))
        
        test_dataset_acc = evaluate_model_on_test_set(resnet18_model, test_loader)

        if test_dataset_acc > best_acc:
            best_acc = test_dataset_acc
            save_checkpoint(model, epoch, optimiser, best_acc)

    print("finished")
    return model


train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimiser, 20)

checkpoint = torch.load('model_best_checkpoint.pth.tar')

resnet18_model = models.resnet18(pretrained=False)
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 10
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
resnet18_model.load_state_dict(checkpoint["model"])

torch.save(resnet18_model, "best_model.pth")