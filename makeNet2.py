import torch 
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import os
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


# Hyper Parameters
num_epochs = 30
batch_size = 16
learning_rate = 0.001

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'pi_split'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                             shuffle = True, num_workers=4),
               'val' : torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                             shuffle = False, num_workers=4)}
 
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()
#use_gpu = False

# Data Loader (Input Pipeline)
train_loader = dataloaders['train']

test_loader = dataloaders['val']

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(100352, 5)
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2))
        # 
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=5, padding=2),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2))
        
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


        
cnn = CNN()

if use_gpu:
    cnn = cnn.cuda()


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

#Train the Model
best_acc = 0
dictStore = {}
best_model_wts = cnn.state_dict()
for epoch in range(num_epochs):
    cnn.train()
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        if use_gpu:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        running_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        
        print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, len(image_datasets['train'])//batch_size, loss.data[0]))

    # Test the Model
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    total_basophil = 0
    correct_basophil = 0
    total_eosinophil = 0
    correct_eosinophil = 0
    total_lymphocyte = 0
    correct_lymphocyte = 0
    total_monocyte = 0
    correct_monocyte = 0
    total_neutrophil = 0
    correct_neutrophil = 0
    for imagesv, labelsv in test_loader:
        if use_gpu:
            imagesv = Variable(imagesv.cuda())
            labelsv = Variable(labelsv.cuda())
        else:
            imagesv = Variable(imagesv)
            labelsv = Variable(labelsv)

        outputs = cnn(imagesv)
        _, predicted = torch.max(outputs.data, 1)
        total += labelsv.size(0)
        correct += (predicted == labelsv.data).sum()
        # print (labelsv.data== 0)
        # print (predicted == 0)
        # print ((labelsv.data == 0) * (predicted == 0))
        total_basophil += (labelsv.data == 0).sum() 
        correct_basophil += ((labelsv.data == 0) * (predicted == 0)).sum()
        total_eosinophil += (labelsv.data == 1).sum()
        correct_eosinophil += ((labelsv.data == 1) * (predicted == 1)).sum()
        total_lymphocyte += (labelsv.data == 2).sum()
        correct_lymphocyte += ((labelsv.data == 2) * (predicted == 2)).sum()
        total_monocyte += (labelsv.data == 3).sum()
        correct_monocyte += ((labelsv.data == 3) * (predicted == 3)).sum()
        total_neutrophil += (labelsv.data == 4).sum()
        correct_neutrophil += ((labelsv.data == 4) * (predicted == 4)).sum()
        

    val_acc = 100 * correct / total
    basophil_acc = 100 * correct_basophil / total_basophil
    eosinophil_acc = 100 * correct_eosinophil / total_eosinophil
    lymphocyte_acc = 100 * correct_lymphocyte / total_lymphocyte
    monocyte_acc = 100 * correct_monocyte / total_monocyte
    neutrophil_acc = 100 * correct_neutrophil / total_neutrophil

    print('Val Accuracy of the model %d %%' % (val_acc))
    #print('basophil Accuracy of the model %d %%' % (basophil_acc))
    if val_acc > best_acc:
        best_model_wts = cnn.state_dict()
        best_acc = val_acc
        dictStore['baso'] = basophil_acc
        dictStore['eosino'] = eosinophil_acc
        dictStore['lymph'] = lymphocyte_acc
        dictStore['mono'] = monocyte_acc
        dictStore['neutro'] = neutrophil_acc
        print('model saved')


#Save the Trained Model
torch.save(best_model_wts, 'cnn_2_layer.pkl')
print ('best val acc: %d' %(best_acc))
print (dictStore)
