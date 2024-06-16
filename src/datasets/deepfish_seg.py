import pandas as pd
import numpy as np
from src import datasets
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from . import augmentation as aug

class DeepfishSeg(Dataset):
    def __init__(self, split, transform=None,
                 root_dir="", n_samples=None):
        super(DeepfishSeg, self).__init__()

        self.split = split
        self.root_dir = root_dir
        self.transform = transform

        self.img_names, self.labels, self.mask_names = get_seg_data(self.root_dir, split)

        if n_samples:
            self.img_names = self.img_names[:n_samples]
            self.mask_names = self.mask_names[:n_samples]
            self.labels = self.labels[:n_samples]
        self.path = self.root_dir  # + "/images/"

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, item):
        name = self.img_names[item]
        image_pil = Image.open(self.path + "/images/" + name + ".jpg")
        mask_pil = Image.open(self.path + "/masks/" + self.mask_names[item] + ".png").convert('L')

        batch = {"images": image_pil,
                 "labels": torch.tensor([self.labels[item]]),
                 "masks": mask_pil}

        # if self.split == 'train':
        #     batch = aug.augment(batch)
        batch['images'] = self.transform(batch['images'])
        batch = aug.mask_to_tensor(batch)

        return batch

    def get_raw_image(self, item):
        name = self.img_names[item]
        image_pil = Image.open(self.path + "/images/" + name + ".jpg")
        mask_pil = Image.open(self.path + "/masks/" + self.mask_names[item] + ".png").convert('L')

        batch = {"images": image_pil,
                 "labels": self.labels[item],
                 "masks": mask_pil}

        return batch

# for seg
def get_seg_data(root_dir, split):
    df = pd.read_csv(os.path.join(root_dir, '%s.csv' % split))
    img_names = np.array(df['ID'])
    mask_names = np.array(df['ID'])
    labels = np.array(df['labels'])
    return img_names, labels, mask_names