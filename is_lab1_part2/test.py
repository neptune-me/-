import torch
import pickle
import os
import torchvision.transforms as transforms
from tools.my_dataset import CharacterDataset
from torch.utils.data import DataLoader

def load_checkpoint(filename):
    try:
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
    except EOFError:
        print("文件为空")
        return None
    return checkpoint
checkpoint = load_checkpoint(pkl_path)
# batch size
batch_size = 
#data loader
pkl_path = ""

valid_dir = os.path.join("train_data", "valid")

norm_mean = [0.835]
norm_std = [0.137]

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# 构建MyDataset实例
valid_data = CharacterDataset(data_dir=valid_dir,transform=valid_transform)

# 构建DataLoder，使用实例化后的数据集作为dataset
valid_loader = DataLoader(dataset=valid_data , batch_size=BATCH_SIZE)
# net
net = load_checkpoint(pkl_path)["net"]
total_val = 0.
correct_val = 0.

for j, data in enumerate(valid_loader):
    inputs, labels = data
    outputs = net(inputs)

    _, predicted = torch.max(outputs.data, 1)
    total_val += labels.size(0)
    correct_val += (predicted == labels).squeeze().sum().numpy()


print("Valid:\t Acc:{:.2%}".format(correct_val / total_val))