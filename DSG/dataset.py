import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import torchvision
import torch


class V2ttDataset(Dataset):
    def __init__(self, x, y, z):
        self.vis = x
        self.inf = y
        self.id = z

        self.trans = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        self.trans_id = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        n = item % len(self.vis)
        inf = self.trans(Image.open(self.inf[n]).convert('RGB'))
        vis = self.trans(Image.open(self.vis[n]).convert('RGB'))
        id = self.trans_id(Image.open(
            self.id[np.random.randint(0, len(self.id))]).convert('RGB'))
        return id, vis, inf

    def __len__(self):
        return len(self.id)


def make_v2tt_dataset(vis_path, inf_path, celeba_root, celeba_text):
    vis_dic = []
    inf_dic = []
    id_dic = []

    vis_folder = torchvision.datasets.ImageFolder(vis_path)
    for i in range(len(vis_folder)):
        vis_dic.append(vis_folder.imgs[i][0])

    inf_folder = torchvision.datasets.ImageFolder(inf_path)
    for i in range(len(inf_folder)):
        inf_dic.append(inf_folder.imgs[i][0])

    with open(celeba_text, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            id_dic.append(celeba_root + line)
    f.close()
    print("")
    print("The numbers of all kinds of images are:")
    print(f"Vis {len(vis_dic)}, Inf {len(inf_dic)}, ID {len(id_dic)}")
    print('................')
    dataset = V2ttDataset(vis_dic, inf_dic, id_dic)
    return dataset


class CelebADataset(Dataset):
    def __init__(self, vis):
        self.vis = vis
        self.trans = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        vis = self.trans(Image.open(self.vis[item]).convert('RGB'))
        return vis

    def __len__(self):
        return len(self.vis)


def make_celebA_dataset(celeba_path):
    id_dic = []
    path_label = torchvision.datasets.ImageFolder(celeba_path)
    for i in range(len(path_label)):
        id_dic.append(path_label.imgs[i][0])
    print("")
    print("The number of CelebA images is : ", len(id_dic))
    print('................')
    dataset = CelebADataset(id_dic)
    return dataset


class FinetuneDataset(Dataset):
    def __init__(self, x, y):
        self.vis = x
        self.id = y
        self.trans = transforms.Compose([

            transforms.ToTensor(),
            transforms.Resize((128, 128)),
        ])

    def __getitem__(self, item):
        n = np.random.randint(0, len(self.vis))
        vis = self.trans(Image.open(self.vis[n]).convert('RGB'))
        id = self.trans(Image.open(self.id[item]).convert('RGB'))
        return id, vis

    def __len__(self):
        return len(self.id)


def make_finetune_dataset(vis_path, celeba_path):
    vis_dic = []
    id_dic = []

    vis_folder = torchvision.datasets.ImageFolder(vis_path)
    for i in range(len(vis_folder)):
        vis_dic.append(vis_folder.imgs[i][0])

    path_label = torchvision.datasets.ImageFolder(celeba_path)
    for i in range(len(path_label)):
        id_dic.append(path_label.imgs[i][0])
    print("")
    print("The numbers of all kinds of images are:")
    print(f"Vis {len(vis_dic)}, ID {len(id_dic)}")
    print('................')
    dataset = FinetuneDataset(vis_dic, id_dic)
    return dataset


class TuftsDataset(Dataset):
    def __init__(self, x, y, z):
        self.vis = x
        self.inf = y
        self.label = z
        self.trans = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, item):
        inf = self.trans(Image.open(self.inf[item]).convert('RGB'))
        vis = self.trans(Image.open(self.vis[item]).convert('RGB'))
        return vis, inf, self.label[item]

    def __len__(self):
        return len(self.vis)


def make_tufts_dataset(vis_path, inf_path):
    vis_dic = []
    inf_dic = []
    label = []

    vis_folder = torchvision.datasets.ImageFolder(vis_path)
    for i in range(len(vis_folder)):
        path = vis_folder.imgs[i][0]
        vis_dic.append(vis_folder.imgs[i][0])
        label.append(path.split('/')[-1].split('-')[0])

    inf_folder = torchvision.datasets.ImageFolder(inf_path)
    for i in range(len(inf_folder)):
        inf_dic.append(inf_folder.imgs[i][0])

    print("")
    print("The numbers of all kinds of images are:")
    print(f"Vis {len(vis_dic)}, Inf {len(inf_dic)}")
    print('................')
    dataset = TuftsDataset(vis_dic, inf_dic, label)
    return dataset


class GenerationDataset(Dataset):
    def __init__(self, x, c):
        self.vis = x
        self.c = c
        self.trans = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),

            ])

    def __getitem__(self, item):

        vis = []
        n = [j for j in range(len(self.c)) if self.c[j] == self.c[item]]
        for i in range(len(n)):
            tmp = self.trans(Image.open(self.vis[n[i]]).convert('RGB')).unsqueeze(0)
            vis.append(tmp)
        return torch.cat(vis, dim=0), self.c[item]

    def __len__(self):
        return len(self.vis)


class VisDataset(Dataset):
    def __init__(self, x, c):
        self.vis = x
        self.c = c
        self.trans = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),

            ])

    def __getitem__(self, item):
        vis = self.trans(Image.open(self.vis[item]).convert('RGB'))
        return vis, self.c[item]

    def __len__(self):
        return len(self.vis)


def make_generation_dataset(txt1, celeba_root, celeba_text):
    vis_dic = []
    vis_c = []

    with open(txt1, 'r') as f:
        for line in f:
            image_path = line.split('\n')[0]
            image_c = image_path.split('/')[-1].split('-')[0]
            if os.path.exists(image_path):
                vis_dic.append(image_path)
                vis_c.append(int(image_c))
    f.close()

    id_c = []
    id_dic = []
    zjr = 0
    with open(celeba_text, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            id_dic.append(celeba_root + line)
            id_c.append(zjr)
            zjr += 1
    f.close()

    print("")
    print("The numbers of all kinds of images are:")
    print(f"Vis {len(vis_dic)}, ID {len(id_dic)}")
    print('................')
    dataset1 = GenerationDataset(vis_dic,vis_c)
    dataset2 = VisDataset(id_dic, id_c)
    return dataset1, dataset2
