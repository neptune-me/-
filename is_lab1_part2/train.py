# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import pickle

from tools.my_dataset import CharacterDataset
from model.CNN import CNN, ResNet18, ResNet50

checkpoint_name = "best_cnn"

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
def save_checkpoint(net):
    if checkpoint_name is None:
        return
    checkpoint = {
        'net':net
    }
    filename = '%s.pkl' % (checkpoint_name)
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    
set_seed(112)  # 设置随机种子

# 参数设置
MAX_EPOCH = 30
BATCH_SIZE = 16
LR = 1e-2
log_interval = 100
val_interval = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # ============================ step 1/5 数据 ============================
    
    train_dir = os.path.join("train_data", "train")
    valid_dir = os.path.join("train_data", "valid")
    
    norm_mean = [0.835]
    norm_std = [0.137]
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0), ratio=(0.7, 1.3)), # scale=(0.8, 1), ratio=(0.8, 1.25)
        transforms.RandomRotation(15),  # 10
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),  # translate = 0.1
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
        ])
    
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    
    # 构建MyDataset实例
    train_data = CharacterDataset(data_dir=train_dir, transform=train_transform)
    valid_data = CharacterDataset(data_dir=valid_dir,transform=valid_transform)
    
    # 构建DataLoder，使用实例化后的数据集作为dataset
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_data , batch_size=BATCH_SIZE, num_workers=2)
    
    # ============================ step 2/5 模型 ============================
    
    net = CNN(classes=12) # 最高99.33
    net.initialize_weights()
    # net = ResNet18(num_classes=12) # 最高99.60
    net = net.to(device)
    
    # ============================ step 3/5 损失函数 ============================
    
    criterion = nn.CrossEntropyLoss() # 选择损失函数
    
    # ============================ step 4/5 优化器 ============================
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9) #选择优化器
    # optimizer = optim.Adagrad(net.parameters(), lr=LR) #选择优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # 设置学习率下降策略
    # 
    # ============================ step 5/5 训练 ============================
    train_curve = list()
    valid_curve = list()
    best_acc = 0.
    
    for epoch in range(MAX_EPOCH):
        
        loss_mean = 0.
        correct = 0.
        total = 0.
        
        net.train()
        for i, data in enumerate(train_loader):
            
            # forward
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            
            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            
            # update weights
            optimizer.step()
            
            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()
            
            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2f}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.
            
        scheduler.step()  # 更新学习率
        
        # validate the model
        if (epoch+1) % val_interval == 0:
    
            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
    
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted.cpu() == labels.cpu()).squeeze().sum().numpy()
    
                    loss_val += loss.item()
                
                # 保存最好网络的参数
                if (correct_val / total_val > best_acc):
                    best_acc = correct_val / total_val
                    save_checkpoint(net)
    
                valid_curve.append(loss_val/valid_loader.__len__())
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.4%}".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val, correct_val / total_val))
    
    print("Best valid acc: {:.2%}".format(best_acc))
    
    train_x = range(len(train_curve))
    train_y = train_curve
    
    train_iters = len(train_loader)
    valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve
    
    plt.figure(figsize=(8, 8))
    
    plt.plot(train_x[5:], train_y[5:], label='Train')
    plt.plot(valid_x[5:], valid_y[5:], label='Valid')
    
    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.show()
 


if __name__ == '__main__':
    main()
    