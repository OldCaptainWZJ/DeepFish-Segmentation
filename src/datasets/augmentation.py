from torchvision import transforms
from torchvision.transforms import functional as F
import numpy as np
import random
import torch

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['images'], sample['masks']
        image = transforms.ToTensor()(image)
        label = torch.as_tensor(np.array(label) / 255., dtype=torch.long)
        return {'images': image, 'masks': label, 'labels': sample['labels']}

class MaskToTensor(object):
    def __call__(self, sample):
        image, label = sample['images'], sample['masks']
        label = torch.as_tensor(np.array(label) / 255., dtype=torch.long)
        return {'images': image, 'masks': label, 'labels': sample['labels']}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, label = sample['images'], sample['masks']
        if random.random() > 0.5:
            image = F.hflip(image)
            label = F.hflip(label)
        return {'images': image, 'masks': label, 'labels': sample['labels']}

class RandomRotation(object):
    def __init__(self, rotation_range=(-15, 15)):
        self.rotation_range = rotation_range

    def __call__(self, sample):
        image, label = sample['images'], sample['masks']
        angle = random.uniform(*self.rotation_range)
        image = F.rotate(image, angle, transforms.InterpolationMode.BILINEAR)
        label = F.rotate(label, angle, transforms.InterpolationMode.NEAREST)
        return {'images': image, 'masks': label, 'labels': sample['labels']}


augment = transforms.Compose([
    RandomHorizontalFlip(),
    RandomRotation()
])

to_tensor = ToTensor()
mask_to_tensor = MaskToTensor()