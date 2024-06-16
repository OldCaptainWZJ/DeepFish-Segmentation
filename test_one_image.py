import argparse
import torch
from src import datasets, models, training
import os
from torchvision.transforms import ToPILImage

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


@torch.no_grad()
def test_one_image(args):
    if args.resize:
        test_data = datasets.get_dataset('deepfish_seg', 'test', 'resize_normalize', args.data_dir)
    else:
        test_data = datasets.get_dataset('deepfish_seg', 'test', 'rgb_normalize', args.data_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.get_model(args.network).to(device)
    load_checkpoint_state(args.model_dir, device, model, args.plain)

    raw_batch = test_data.get_raw_image(args.index)
    batch = test_data[args.index]
    batch['images'] = torch.unsqueeze(batch['images'], dim=0)
    batch['masks'] = torch.unsqueeze(batch['masks'], dim=0)
    batch['labels'] = torch.unsqueeze(batch['labels'], dim=0)

    model_params = models.params_init(args)

    model.eval()
    output, metric = training.run_on_batch(model, batch, device, args.resize, model_params)

    output = torch.squeeze(output, dim=0)
    output = (output > args.threshold)

    output_gray = output[1]
    output_rgb = torch.stack([output_gray, output_gray, output_gray], dim=0)
    output_rgb = (output_rgb * 255).type(torch.uint8)
    to_pil_image = ToPILImage()
    output_pil = to_pil_image(output_rgb)

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    raw_batch['images'].save(os.path.join(args.result_dir, 'input.png'))
    raw_batch['masks'].save(os.path.join(args.result_dir, 'output_gt.png'))
    output_pil.save(os.path.join(args.result_dir, 'output_pd.png'))

    print("Images saved at path: ", args.result_dir)
    print("mIoU for this test: ", metric['score'])
    if model_params['with_cgm']:
        print("Classification correct: ", (metric['score_class'].item() > 0.))
    if metric['num_empty'] != 0:
        print("Oversegmentation: ", metric['over_seg'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model on one image')

    parser.add_argument('-d', '--data_dir', required=True, help="Data directory")
    parser.add_argument('-i', '--index', type=int, default=0, help="Index for image you want to test")
    parser.add_argument('-r', '--result_dir', default=os.path.join(".", "test_result"), help="Path for result image(s) to be saved")
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help="Threshold for judging if a pixel is fish or not")
    parser.add_argument('-m', '--model_dir', default=None, help="Experiment directory (where you put generated models)")
    parser.add_argument('-n', '--network', default="unet", help="Network structure used for experiment")
    parser.add_argument('-rs', '--resize', default=True, help="Whether to resize images before input")
    parser.add_argument('-p', '--plain', type=bool, default=False, help="Whether to use plain model dict")

    args = parser.parse_args()

    test_one_image(args)
