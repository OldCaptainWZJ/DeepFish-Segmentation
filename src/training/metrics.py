import torch


def mIoU(inputs, targets, threshold=0.5, num_classes=2, softmax=True, hard=True):
    if softmax:
        inputs = torch.softmax(inputs, dim=1)
    if hard:
        inputs = (inputs > threshold)

    class_wise_iou = []

    for i in range(num_classes):
        input = inputs[:, i]
        target = (targets == i)
        intersection = torch.sum(input * target, dim=(1, 2))
        total = torch.sum(input, dim=(1, 2)) + torch.sum(target, dim=(1, 2))

        iou = (intersection + 1e-8) / (total - intersection + 1e-8)

        class_wise_iou.append(iou.mean().item())

    return class_wise_iou


def Dice(inputs, targets, threshold=0.5, num_classes=2, softmax=True, hard=True):
    if softmax:
        inputs = torch.softmax(inputs, dim=1)
    if hard:
        inputs = (inputs > threshold)

    class_wise_dice = []

    for i in range(num_classes):
        input = inputs[:, i]
        target = (targets == i)
        intersection = torch.sum(input * target, dim=(1, 2))
        total = torch.sum(input, dim=(1, 2)) + torch.sum(target, dim=(1, 2))

        dice = (2. * intersection + 1e-8) / (total + 1e-8)

        class_wise_dice.append(dice.mean().item())

    return class_wise_dice


# Pixel Accuracy
def PA(inputs, targets, threshold=0.5, num_classes=2, softmax=True, hard=True):
    if softmax:
        inputs = torch.softmax(inputs, dim=1)
    if hard:
        inputs = (inputs > threshold)

    class_wise_tp = []

    for i in range(num_classes):
        input = inputs[:, i]
        target = (targets == i)
        intersection = torch.sum(input * target, dim=(1, 2))

        class_wise_tp.append(intersection)

    tp = class_wise_tp[0] + class_wise_tp[1]
    PA = tp / (inputs.size()[2] * inputs.size()[3])
    PA = PA.mean().item()

    return PA

def OverSeg(inputs, labels, threshold=0.5, softmax=True, hard=True):
    if softmax:
        inputs = torch.softmax(inputs, dim=1)
    if hard:
        inputs = (inputs > threshold)

    inputs = inputs[:, 1]
    batch_size = inputs.size()[0]

    num_empty = 0
    total_seg = 0.

    for i in range(batch_size):
        if labels[i].item() == 0:
            num_empty += 1
            total_seg += inputs[i].float().mean().item()

    return num_empty, total_seg


def mIoU_on_valid(inputs, targets, labels, threshold=0.5, softmax=True, hard=True):
    if softmax:
        inputs = torch.softmax(inputs, dim=1)
    if hard:
        inputs = (inputs > threshold)

    inputs = inputs[:, 1]
    batch_size = inputs.size()[0]

    intersection = torch.sum(inputs * targets, dim=(1, 2))
    total = torch.sum(inputs, dim=(1, 2)) + torch.sum(targets, dim=(1, 2))
    iou = (intersection + 1e-8) / (total - intersection + 1e-8)

    num_valid = 0
    total_iou = 0.

    for i in range(batch_size):
        if labels[i].item() == 1:
            num_valid += 1
            total_iou += iou[i].float().item()

    return num_valid, total_iou
