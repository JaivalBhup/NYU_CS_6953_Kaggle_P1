import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import pandas as pd
from torchsummary import summary
import torch_optimizer as optim
import os
from model import ResNet, ResNet_Block # Import models

model_save_path = './model_save' # Save path for the final model

# Function for data augmentation for train and test dataset
def get_transformations():
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=6), # Randomly crops the image and resizes it. If the crop image size is same as the image, it randomizes the image patches
        transforms.RandomHorizontalFlip(), # Horizontally flips the image with the probability of 50%
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # increases brightness, contrast, sturation and hue of the image
        transforms.RandomRotation(20), # Randomly rotates the image with degree 20
        transforms.RandomAffine(degrees=10, translate=(0.2, 0.2)), # Applies random affine
        transforms.ToTensor(), # Converts the image to tensors, also scales it from 0-1
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), # Normalize
    ])

    test_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(), # Converts the image to tensors, scales it from 0-1
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)), # Normalize
    ])

    return train_transform, test_transform

# Function to download data, convert to datasets, transforms, and convert to dataloaders
def get_data_loaders():
    # Dataset
    train_transform, test_transform = get_transformations()
    train = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
    # Data Loader
    train_data_loader  = torch.utils.data.DataLoader(train, batch_size=128,shuffle=True)
    validation_data_loader  = torch.utils.data.DataLoader(test, batch_size=128,shuffle=False)
    return train_data_loader, validation_data_loader

# Train the model
'''
    Arguments:
        model
        train_data_loader
        validation_data_loader
        label_smoothing
        learning rate
        weight decay
        momentum
        nesterov
        lookahead
        epochs

    Returns:
        Trained model
        train Loss
        validation Loss
        train accuracy
        validation accuracy
'''
def train_model(model, train_data_loader, validation_data_loader, label_smoothing = 0.1, lr = 0.1, weight_decay = 0.0005, momentum=0.9, nesterov=True, lookahead=False, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Get the device
    loss = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing) # Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay = weight_decay, momentum=momentum, nesterov=nesterov) # SGD optimizer with initial learning rate, weight decay, momentum, and nesterov
    if lookahead: # if lookahead is selected the update the optimizer 
        optimizer = optim.Lookahead(optimizer, k=5, alpha=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001) # scheduler 
    train_loss = []
    val_loss = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        trainloss = 0.0
        valloss = 0.0
        val_correct = 0
        train_correct = 0
        val_total = 0
        train_total = 0
        model.train() # telling python that we are intereseted in updating any trainable parameters in the network

        for images, labels in train_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() # makes sure we have zeroes out gradients for trainable parameters from the previous iteration
            pred = model(images) # forward pass
            fit = loss(pred, labels) # Calculate loss
            fit.backward() # backpropogation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # Clipping gradients to norm value of 1 --> [-1, 1]
            optimizer.step() # updates the weight
            trainloss += fit.item()
            _, predicted = torch.max(pred, 1) # Get indexes of the most confident guess
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        model.eval() # Switch the model to evaluate mode. 
        for images, labels in validation_data_loader:
            with torch.no_grad(): # Makes sure that gradient calculation is disabled
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images) # forward pass
                fit = loss(pred, labels) # calculate loos
                valloss += fit.item()
                _, predicted = torch.max(pred, 1) # Get indexes of the most confident guess
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        trainloss = trainloss/len(train_data_loader)
        valloss = valloss/len(validation_data_loader)
        val_loss.append(valloss)
        train_loss.append(trainloss)

        val_accuracy = 100 * val_correct/val_total
        val_accuracies.append(val_accuracy)
        train_accuracy = 100 * train_correct/train_total
        train_accuracies.append(train_accuracy)


        scheduler.step() # Update the learning rate

        print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {trainloss:.2f} | val loss: {valloss:.2f} | val Accuracy: {val_accuracy:.2f}%')

    return model, train_loss, val_loss, train_accuracies, val_accuracies

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# Here layers are [Block, Block... <Number of blocks>], Block = Tuple(<number of layers>, <should this layer have SE block>)
layers = [(4, True),(4, True),(3, True)]
# ResNet model takes 2 parameters ==> What block to use, layers
model = ResNet(ResNet_Block, layers).to(device)

# Get data loaders
train_data_loader, validation_data_loader = get_data_loaders()

# Train the model
model, train_loss, val_loss, train_accuracies, val_accuracies = train_model(model = model, train_data_loader=train_data_loader, validation_data_loader=validation_data_loader, label_smoothing = 0.1, lr = 0.1, weight_decay = 0.0005, momentum=0.9, nesterov=True, lookahead=True, epochs=100)

# Save the model
os.mkdir(model_save_path)
torch.save(model.state_dict(), os.path.join(model_save_path,'/SE_ResNet_4_4_3.pth'))

# Plot the loss and accuracy
fig, ax = plt.subplots(1,2,figsize=(10,4))
ax[0].plot(train_loss, label="Train Loss")
ax[0].plot(val_loss, label="Validation Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")
ax[0].legend()

ax[1].plot(train_accuracies, label="Train Accuracy")
ax[1].plot(val_accuracies, label="Validation Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
plt.tight_layout()
plt.show()


