import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision


class Conv1x1(nn.Module):
    """convolution => [BN] => ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=2, bilinear=False):
        super().__init__()
        self.n_classes = n_classes

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.resnet50(pretrained=True)

        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()

        self.resnet50_32s = resnet50_32s

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv1 = Conv1x1(2048, 1024)
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = Conv1x1(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = Conv1x1(512, 256)
        self.outc = nn.Conv2d(256, self.n_classes, kernel_size=1)

    def forward(self, x):
        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x1 = self.resnet50_32s.maxpool(x)
        x2 = self.resnet50_32s.layer1(x1)
        x3 = self.resnet50_32s.layer2(x2)
        x4 = self.resnet50_32s.layer3(x3)
        x5 = self.resnet50_32s.layer4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.upconv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(x)

        logits = self.outc(x)
        return logits

class UNet_DeepSup(nn.Module):
    def __init__(self, n_classes=2, bilinear=False):
        super().__init__()
        self.n_classes = n_classes

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.resnet50(pretrained=True)

        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()

        self.resnet50_32s = resnet50_32s

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv1 = Conv1x1(2048, 1024)
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = Conv1x1(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = Conv1x1(512, 256)

        self.intconv5 = nn.Conv2d(2048, 64, 1)
        self.intconv4 = nn.Conv2d(1024, 64, 1)
        self.intconv3 = nn.Conv2d(512, 64, 1)
        self.intconv2 = nn.Conv2d(256, 64, 1)

        self.outconv5 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(64, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, n_classes, 3, padding=1)

    def forward(self, x):
        input_size = x.size()[2:]

        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x1 = self.resnet50_32s.maxpool(x)
        x2 = self.resnet50_32s.layer1(x1)
        x3 = self.resnet50_32s.layer2(x2)
        x4 = self.resnet50_32s.layer3(x3)
        x5 = self.resnet50_32s.layer4(x4)

        out5 = self.intconv5(x5)
        out5 = F.interpolate(out5, size=input_size, mode='bilinear', align_corners=True)
        out5 = self.outconv5(out5)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.upconv1(x)
        out4 = self.intconv4(x)
        out4 = F.interpolate(out4, size=input_size, mode='bilinear', align_corners=True)
        out4 = self.outconv4(out4)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(x)
        out3 = self.intconv3(x)
        out3 = F.interpolate(out3, size=input_size, mode='bilinear', align_corners=True)
        out3 = self.outconv3(out3)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(x)
        out2 = self.intconv2(x)
        out2 = F.interpolate(out2, size=input_size, mode='bilinear', align_corners=True)
        out2 = self.outconv2(out2)

        return out5, out4, out3, out2

class CGM(nn.Module):
    """Classification Guided Module"""
    def __init__(self, in_channels, mid_channels, n_classes, feature_size):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels, n_classes, 1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.cls(x)
        return x


class UNet_CGM(nn.Module):
    def __init__(self, n_classes=2, bilinear=False):
        super().__init__()
        self.n_classes = n_classes

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.resnet50(pretrained=True)

        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()

        self.resnet50_32s = resnet50_32s

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.upconv1 = Conv1x1(2048, 1024)
        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = Conv1x1(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = Conv1x1(512, 256)
        self.outc = nn.Conv2d(256, self.n_classes, kernel_size=1)

        self.cgm = CGM(2048, 16, 2, 16)

    def forward(self, x):
        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x1 = self.resnet50_32s.maxpool(x)
        x2 = self.resnet50_32s.layer1(x1)
        x3 = self.resnet50_32s.layer2(x2)
        x4 = self.resnet50_32s.layer3(x3)
        x5 = self.resnet50_32s.layer4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.upconv1(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(x)
        logits = self.outc(x)  # logits: [n,2,h,w]

        classes = self.cgm(x5)  # classes: [n,2,1,1]

        return logits, classes


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class Attention_UNet(nn.Module):
    def __init__(self, n_classes=2, bilinear=False):
        super().__init__()
        self.n_classes = n_classes

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.resnet50(pretrained=True)

        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()

        self.resnet50_32s = resnet50_32s

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.att1 = Attention_block(1024, 1024, 256)
        self.upconv1 = Conv1x1(2048, 1024)

        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att2 = Attention_block(512, 512, 128)
        self.upconv2 = Conv1x1(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = Attention_block(256, 256, 64)
        self.upconv3 = Conv1x1(512, 256)

        self.outc = nn.Conv2d(256, self.n_classes, kernel_size=1)

    def forward(self, x):
        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x1 = self.resnet50_32s.maxpool(x)
        x2 = self.resnet50_32s.layer1(x1)
        x3 = self.resnet50_32s.layer2(x2)
        x4 = self.resnet50_32s.layer3(x3)
        x5 = self.resnet50_32s.layer4(x4)

        x = self.up1(x5)
        x = torch.cat([x, self.att1(g=x, x=x4)], dim=1)
        x = self.upconv1(x)

        x = self.up2(x)
        x = torch.cat([x, self.att2(g=x, x=x3)], dim=1)
        x = self.upconv2(x)

        x = self.up3(x)
        x = torch.cat([x, self.att3(g=x, x=x2)], dim=1)
        x = self.upconv3(x)

        logits = self.outc(x)
        return logits

class Attention_UNet_CGM(nn.Module):
    def __init__(self, n_classes=2, bilinear=False):
        super().__init__()
        self.n_classes = n_classes

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = torchvision.models.resnet50(pretrained=True)

        # Create a linear layer -- we don't need logits in this case
        resnet50_32s.fc = nn.Sequential()

        self.resnet50_32s = resnet50_32s

        # # FREEZE BATCH NORMS
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.att1 = Attention_block(1024, 1024, 256)
        self.upconv1 = Conv1x1(2048, 1024)

        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att2 = Attention_block(512, 512, 128)
        self.upconv2 = Conv1x1(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = Attention_block(256, 256, 64)
        self.upconv3 = Conv1x1(512, 256)

        self.outc = nn.Conv2d(256, self.n_classes, kernel_size=1)

        self.cgm = CGM(2048, 16, 2, 16)

    def forward(self, x):
        x = self.resnet50_32s.conv1(x)
        x = self.resnet50_32s.bn1(x)
        x = self.resnet50_32s.relu(x)
        x1 = self.resnet50_32s.maxpool(x)
        x2 = self.resnet50_32s.layer1(x1)
        x3 = self.resnet50_32s.layer2(x2)
        x4 = self.resnet50_32s.layer3(x3)
        x5 = self.resnet50_32s.layer4(x4)

        x = self.up1(x5)
        x = torch.cat([x, self.att1(g=x, x=x4)], dim=1)
        x = self.upconv1(x)

        x = self.up2(x)
        x = torch.cat([x, self.att2(g=x, x=x3)], dim=1)
        x = self.upconv2(x)

        x = self.up3(x)
        x = torch.cat([x, self.att3(g=x, x=x2)], dim=1)
        x = self.upconv3(x)

        logits = self.outc(x)
        classes = self.cgm(x5)  # classes: [n,2,1,1]

        return logits, classes