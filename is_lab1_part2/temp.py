# import numpy as np
# from PIL import Image
import os
# import torch
import shutil

def load_dataset():

    for i in range(2, 13):
        train_root = "train_data/train/" + str(i)
        valid_root = "train_data/valid/" + str(i)
        for j in range(497, 621):
            src_path = train_root + "/" + str(j) +".bmp"
            dest_path = valid_root + "/" + str(j) + ".bmp"
            # img = Image.open(path).convert('L')
            # X_train.append(np.array(img))
            # y_train.append(i - 1)
            mymovefile(src_path, dest_path)
        print ("move %s -> %s"%( train_root,valid_root))
            
        # for j in range(497, 621):
        #     path = root + "/" + j +".bmp"
        #     img = Image.open(path)
        #     X_test.append(np.array(img))
        #     y_test.append(i - 1)
            
    # return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        

# data = {}
# data["X_train"], data["y_train"], data["X_test"], data["y_test"] = load_dataset()


# X = torch.from_numpy(data["X_train"])

# X = X.view([X.shape[0]*X.shape[1]*X.shape[2]])

# X1 = X.numpy()

# mean = np.mean(X1)
# std = np.var(X1)
load_dataset()

# print(mean / 255)
# print(std / 255 /255)
        