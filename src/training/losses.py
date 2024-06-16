import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, num_classes=2, smooth=1e-5, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        if weight is None:
            weight = [1.] * self.num_classes

        batch_size = inputs.size()[0]
        class_wise_dice_loss = []
        loss = None

        for i in range(self.num_classes):
            image = inputs[:, i]
            image = torch.squeeze(image, dim=1)
            target = (targets == i)
            intersection = torch.sum(image * target, dim=(1, 2))
            total = torch.sum(image, dim=(1, 2)) + torch.sum(target, dim=(1, 2))

            class_dice = (2. * intersection + self.smooth) / (total + self.smooth)

            class_wise_dice_loss.append(1.0 - class_dice)
            if loss is None:
                loss = (1.0 - class_dice) * weight[i]
            else:
                loss += (1.0 - class_dice) * weight[i]

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss