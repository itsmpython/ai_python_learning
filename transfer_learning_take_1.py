#Import Libraries
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import helper

# Prepare the data
data_dir = 'Cat_Dog_data'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(223),
                                      transforms.ToTensor()
                                     ])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

#Transfer Model from desenet

model = models.densenet121(pretrained=True)
#model


### ARJUN
#Create a Classifier (Last Layer of the network based on the last layer of the TRANSFERRED NETWORK) 
# Freeze parameters so we don't backprop through them

# TODO: Train a model with a pre-trained network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for params in model.parameters():
    params.requires_grad = False

from collections import OrderedDict

myclassifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 2)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = myclassifier
## Arjun Note : This is important in TRANSFER LEARNING. 
## This is where the classifier of the transferred model, is replaced with the one we created

#define Loss Criterion and Optimizer
criterion = nn.NLLLoss()
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
#print(model)

# Use the pretrained - densenet model with replaced classifier as defiend above. 
# SINCE THIS IS PRETRAINED - NO NEED TO TRAIN. Only the forward and backword pass is what we need.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

epochs = 2
steps = 0
running_loss = 0
print_every = 40

for e in range (epochs):
    running_loss = 0    
    for images, labels in iter(trainloader):
        steps += 1
        
        #helper.imshow(images[0])
        #inputs.resize_(inputs.size()[0],1024)
        inputs, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        #Forward and Backword passes
        output = model.forward(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step() ## To be done : Learn more about what .step() does
        # QUESTION TO MD : what does step() do and why we need it here?
				
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Loss : {:.4f}".format(running_loss/print_every))
            
            running_loss = 0

# Find out Accuracy of the network : Copied from solution notebook

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
				# QUESTION TO MD : What is the meaning of _' here in " _, predicted"
				
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

Accuracy of the network on the 10000 test images: 50 %

