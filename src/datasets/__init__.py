from . import deepfish_seg
from torchvision import transforms

def get_dataset(dataset_name, split, transform_name, root_dir):
    transform = get_transform(transform_name)

    if dataset_name == "deepfish_seg":
        dataset = deepfish_seg.DeepfishSeg(split, transform=transform, root_dir=root_dir)
    
    return dataset

def get_transform(transform):
    if transform == "resize_normalize":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize_transform = transforms.Normalize(mean=mean, std=std)

        return transforms.Compose(
            [transforms.Resize((512, 512)),
             transforms.ToTensor(),
             normalize_transform])

    if transform == "rgb_normalize":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalize_transform = transforms.Normalize(mean=mean, std=std)

        return transforms.Compose(
            [
             transforms.ToTensor(),
             normalize_transform])