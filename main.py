import os
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet50
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, optimizer
from torch.optim.lr_scheduler import *
from torch.autograd import Variable
import math
import h5py
import torch.utils.data as data
from PIL import Image

# hyper parameter
epoch_num = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# use cuda or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA-VISIBLE-DEVICES']='0'

class Dataset(data.Dataset):
    def __init__(self, split="Training", transform = None):
        self.split = split
        self.transform = transform
        self.data = h5py.File('./data/data.h5', 'r', driver='core')
        if self.split == "Training":
            self.train_data = np.asarray(self.data['train_data']).reshape((28709,48,48))
            self.train_label = self.data['train_label']
        elif self.split == "Test":
            self.test_data = np.asarray(self.data['test_data']).reshape((7178,48,48))
            self.test_label = self.data['test_label']
    
    def __getitem__(self, index):
        if self.split == "Training":
            img, label = self.train_data[index], self.train_label[index]
        elif self.split == "Test":
            img, label = self.test_data[index], self.test_label[index]
        # RGB
        img = img[:,:,np.newaxis]
        img = np.concatenate((img,img,img),axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        if self.split == "Training":
            return len(self.train_data)
        elif self.split == "Test":
            return len(self.test_data)

# data augmentation
trans_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
    ])

# dataloader
train_data = Dataset(split="Training", transform=trans_train)
test_data = Dataset(split="Test", transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# network structure
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.linear_layer = nn.Linear(2048, 7)
    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layer(x)
        return x

# def get_acc(output, label):
#     return torch.sum(torch.argmax(output, dim=1) == label.data)
#     count = output.shape[0]
#     _, pred_label = output.max(1)
#     correct_sum = (pred_label == label).sum().item()
#     return correct_sum / count

# train
def train(net):
    # scheduler.step()
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # scheduler = StepLR(optimizer, step_size=3)
    criterion = nn.CrossEntropyLoss()
    # optimizer = SGD(net.parameters(), lr = LEARNING_RATE, weight_decay = 1e-5, momentum = 0.9)
    for epoch in range(epoch_num):
        loss_sum = 0
        acc_sum = 0
        # batch_count = 0
        for idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            # batch_count += 1
            optimizer.zero_grad()
            data = data.view(-1, 3, 48, 48)
            output = net(data.float())
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.cpu().item()
            acc_sum += (torch.argmax(output, dim=1) == label).sum().item()
        loss_mean = loss_sum / len(train_loader.dataset)
        acc_mean = acc_sum / len(train_loader.dataset)
        print("epoch{}  loss:{:.6f} acc: {:.6f}".format(epoch+1, loss_mean, acc_mean))
        torch.save(net, "./ResNet50_ckpt/checkpoint_epoch_"+str(epoch+1)+".pth")

# test
def test(net):
    acc_sum = 0
    with torch.no_grad():
        for idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            data = data.view(-1, 3, 48, 48)
            output = net(data.float())
            acc_sum += (torch.argmax(output, dim=1) == label).sum().cpu().item()
    return acc_sum / len(test_loader.dataset)

# evaluation
def evaluation(net):
    net.eval()
    BER = 0
    MCC = 0
    Sensitivity = 0
    Specificity = 0
    y_pred = []
    y_test = []
    predict_score_list = []
    with torch.no_grad():
        for idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = net(data.float())
            predict_score_list.extend(output.cpu().detach().numpy())
            y_pred.extend(output.argmax(1).cpu().detach().numpy())
            y_test.extend(label.cpu())
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')

    con_mat = metrics.confusion_matrix(y_test, y_pred)
    for i in range(7):
        number = np.sum(con_mat[:,:])
        TP = con_mat[i][i]
        FN = np.sum(con_mat[i,:]) - TP
        FP = np.sum(con_mat[:,i]) - TP
        TN = number - TP - FN - TP
        BER += (FP/(FP+TN)+FN/(FN+TP))/2
        MCC += (TP*TN-FP*FN)/math.sqrt((TP+FP)*(FP+TN)*(TN+FN)*(FN+TP))
        Sensitivity += TP/(TP+FN)
        Specificity += TN/(TN+FP)
    BER = BER/10
    MCC = MCC/10
    Sensitivity = Sensitivity/10
    Specificity = Specificity/10

    print(metrics.classification_report(y_test,y_pred))
    print("ACC:{}\nBER:{}\nMCC:{}\nRecall:{}\nPrecision:{}\nSensitivity:{}\nSpecificity:{}\nF1:{}\n".format(accuracy,BER,MCC,recall,precision,Sensitivity,Specificity,f1))


if __name__ == '__main__':
    resnet = resnet50(pretrained = True)
    model = Net(resnet).to(device)
    # print(Net(resnet))
    train(model)
    for i in range(epoch_num):
        test_model = Net(resnet).to(device)
        test_model = torch.load("./ResNet50_ckpt/checkpoint_epoch_"+str(i+1)+".pth")
        acc = test(test_model)
        print("epoch{}  test accuracy: {:.6f}".format(i+1, acc))
    # test_model = Net(resnet).to(device)
    # test_model = torch.load("./ResNet50_ckpt/checkpoint_epoch_44.pth")
    # print(test(test_model))
    # evaluation(test_model)

