import torch
import torchvision
import numpy as np
import h5py
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class ATLASDataset(Dataset):
    def __init__(self, file_point, len_data, isTransform=None):
        self.file_point = file_point
        self.len_data = len_data
        self.isTransform = isTransform
        self.transform = transforms.RandomChoice([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
        ])

    def __getitem__(self, item):
        imgs = torch.from_numpy(self.file_point["data"][item].transpose(2, 0, 1))
        label = torch.from_numpy(self.file_point["label"][item].transpose(2, 0, 1))
        data = torch.cat((imgs, label), dim=0)
        if self.isTransform:
            data = self.transform(data)
        return data

    def __len__(self):
        return self.len_data




# f = h5py.File(r"/home/administrator/File/h5/train")
# transform = transforms.RandomChoice([
#             # transforms.ToPILImage(),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
#             transforms.RandomRotation(20),
#             transforms.RandomAffine(0, (0.1, 0), fillcolor=None),
#             transforms.RandomAffine(0, (0, 0.1), fillcolor=None),
#             transforms.RandomAffine(0, translate=None, scale=(0.8 ,0.8), fillcolor=None),
#             transforms.RandomAffine(0, translate=None, scale=(1.2,1.2), fillcolor=None),
#             # transforms.ToTensor()
#         ])
# dataset = ATLASDataset(file_point=f, len_data=25893, transform=transform)
# dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)
# for data in dataloader:
#     i = 1
#     data_ = data.numpy()
#     print(data_.shape)
#     for d in data_:
#         print(d.shape)
#         for img in d:
#             print(img.shape)
#             plt.subplot(1, 5, i)
#             plt.imshow(img, cmap="gray")
#             i += 1
#         plt.pause(1)
#         i = 1
    # break