import os
from PIL import Image
from torch.utils.data import Dataset


class CharacterDataset(Dataset):
    def __init__(self, data_dir,transform=None):
        """
        

        Parameters
        ----------
        data_dir : str
            path of dataset.
        transform : torch.transform
            pretrain

        Returns
        -------
        None.

        """
        self.label = [i for i in range(12)]
        self.data_info = self.get_img_info(data_dir);
        self.transform = transform
        
    def __getitem__(self, index): #根据index返回数据
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('L'); # 转灰度图
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self): # 查看样本的数量
        return len(self.data_info)
    
    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir): #就一个dir train_data
            for sub_dir in dirs: # 遍历类别目录 1-12
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.bmp'), img_names))

                for i in range(len(img_names)): # 遍历图片
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = int(sub_dir) -1  #文件夹名就是类别
                    data_info.append((path_img, int(label)))

        return data_info