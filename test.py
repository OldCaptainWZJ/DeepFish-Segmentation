import argparse
import torch
from src import datasets, models, training
from torch.utils.data import DataLoader
import os


def load_checkpoint_state(path, device, model, plain):
    if plain:
        path = os.path.join(path, "model_state_dict.pth")
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        return model

    path = os.path.join(path, "checkpoint.tar")

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def test(args):
    if args.resize:
        test_data = datasets.get_dataset('deepfish_seg', args.split, 'resize_normalize', args.data_dir)
    else:
        test_data = datasets.get_dataset('deepfish_seg', args.split, 'rgb_normalize', args.data_dir)

    test_loader = DataLoader(test_data,
                             shuffle=False,
                             batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.get_model(args.network).to(device)
    load_checkpoint_state(args.model_dir, device, model, args.plain)

    model_params = models.params_init(args)

    training.val_on_loader(model, test_loader, args.resize, 'test', model_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model')

    parser.add_argument('-d', '--data_dir', required=True, help="Data directory")
    parser.add_argument('-b', '--batch_size', type=int, default=1, help="Batch size for testing")
    parser.add_argument('-m', '--model_dir', default=None, help="Experiment directory (where you put generated models)")
    parser.add_argument('-n', '--network', default="unet", help="Network structure used for experiment")
    parser.add_argument('-rs', '--resize', default=True, help="Whether to resize images before input")
    parser.add_argument('-sp', '--split', default="test", help="Run on 'test' or 'val' set")
    parser.add_argument('-p', '--plain', type=bool, default=False, help="Whether to use plain model dict")

    args = parser.parse_args()

    test(args)
