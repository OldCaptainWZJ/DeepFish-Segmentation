from . import losses, metrics
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class MetricCalculator:
    def __init__(self):
        self.loss_list = []
        self.score_list = []
        self.dice_score_list = []
        self.score_class_list = []
        self.num_empty = 0
        self.over_seg = 0.
        self.num_valid = 0
        self.miou_on_valid = 0.

    def update(self, metric, with_cgm=False):
        self.loss_list.append(metric['loss'].item())
        self.score_list.append(metric['score'])
        self.dice_score_list.append(metric['dice_score'])
        if with_cgm:
            self.score_class_list.append(metric['score_class'].item())
        self.num_empty += metric['num_empty']
        self.over_seg += metric['over_seg']
        self.num_valid += metric['num_valid']
        self.miou_on_valid += metric['miou_on_valid']

    def output(self, split, with_cgm=False):
        self.score = np.mean(np.array(self.score_list), axis=0)
        self.dice_score = np.mean(np.array(self.dice_score_list), axis=0)
        self.loss = np.array(self.loss_list).mean()
        if with_cgm:
            self.score_class = np.mean(np.array(self.score_class_list), axis=0)
        self.over_seg /= self.num_empty
        self.miou_on_valid /= self.num_valid

        print('mean loss on %s set: %f' % (split, self.loss))
        print('mIoU over all images (class wise): ', self.score)
        print('mIoU over all images (combining all classes): ', self.score.mean())
        print('Dice over all images (class wise): ', self.dice_score)
        print('Dice over all images (combining all classes): ', self.dice_score.mean())
        if with_cgm:
            print('Classification precision: ', self.score_class)

        print('Oversegmentation over empty set: ', self.over_seg)
        print('mIoU over valid set: ', self.miou_on_valid)

    def get_return_val(self):
        return self.loss, self.score.mean()


def train_on_loader(model, optimizer, data_loader, resize=False, with_cgm=False, lambda_cgm=0.1):
    # train on loader for 1 epoch
    model.train()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    metric_total = MetricCalculator()

    for index, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        optimizer.zero_grad()

        outputs, metric = run_on_batch(model, batch, device, resize, with_cgm, lambda_cgm)

        metric_total.update(metric, with_cgm)

        metric['loss'].backward()
        optimizer.step()

    metric_total.output('train', with_cgm)

    return metric_total.get_return_val()


@torch.no_grad()
def val_on_loader(model, data_loader, resize=False, split='val', with_cgm=False, lambda_cgm=0.1):
    # validate on loader
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_total = MetricCalculator()

    for index, batch in tqdm(enumerate(data_loader), total=len(data_loader)):

        outputs, metric = run_on_batch(model, batch, device, resize, with_cgm, lambda_cgm)

        metric_total.update(metric, with_cgm)

    metric_total.output(split, with_cgm)

    return metric_total.get_return_val()


def run_on_batch(model, batch, device, resize, with_cgm, lambda_cgm):
    Dice = losses.DiceLoss()
    BCE = nn.BCELoss()

    inputs = batch['images'].to(device)
    targets = batch['masks'].to(device)
    score_class = 0.

    if not with_cgm:
        outputs = model(inputs)

        labels = batch['labels'].squeeze(dim=1).to(device)  # labels: [n,1] -> [n]

        if resize:
            outputs = F.interpolate(outputs, size=targets.size()[1:], mode='bilinear', align_corners=True)

        loss = Dice(outputs, targets)  # outputs: [n,2,h,w], targets: [n,h,w]

        score = metrics.mIoU(outputs, targets)
        dice_score = metrics.Dice(outputs, targets)
        num_empty, over_seg = metrics.OverSeg(outputs, labels)
        num_valid, miou_on_valid = metrics.mIoU_on_valid(outputs, targets, labels)
    else:
        outputs, classes = model(inputs)  # outputs: [n,2,h,w], classes: [n,2,1,1]

        pd_class = torch.argmax(classes, dim=1).squeeze(2).squeeze(1)  # pd_class: [n,1,1] -> [n]
        labels = batch['labels'].squeeze(dim=1).to(device)  # labels: [n,1] -> [n]
        score_class = (pd_class == labels).float().mean()

        if resize:
            outputs = F.interpolate(outputs, size=targets.size()[1:], mode='bilinear', align_corners=True)

        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.einsum("ijkl, i -> ijkl", [outputs, pd_class])
        outputs[:, 0] = (1. - outputs[:, 1])

        num_empty, over_seg = metrics.OverSeg(outputs, labels, softmax=False)
        num_valid, miou_on_valid = metrics.mIoU_on_valid(outputs, targets, labels, softmax=False)

        classes = classes.squeeze(3).squeeze(2)
        labels = torch.stack([1 - labels, labels], dim=1).float()  # labels: [n] -> [n,2]
        loss = Dice(outputs, targets, softmax=False) + lambda_cgm * BCE(classes, labels)
        # outputs: [n,2,h,w], targets: [n,h,w]
        # classes: [n,2], labels: [n,2]

        score = metrics.mIoU(outputs, targets, softmax=False)
        dice_score = metrics.Dice(outputs, targets, softmax=False)

    metric = {"loss": loss,
              "score": score,
              "dice_score": dice_score,
              "score_class": score_class,
              "num_empty": num_empty,
              "over_seg": over_seg,
              "num_valid": num_valid,
              "miou_on_valid": miou_on_valid}

    return outputs, metric
