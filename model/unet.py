
import torch.nn as nn
import torch
from .utils import double_conv

class UNet(nn.Module):
    def __init__(self,backbone,last=512):
        super(UNet, self).__init__()

        self.backbone = nn.ModuleList()

        tmp = []

        for i in range(28):
            if (i+1)%7==0:
                self.backbone.append(nn.Sequential(*tmp))
                tmp =  [backbone.features[i]]
                pass
            else:
                tmp.append(backbone.features[i])


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.classifier = nn.ModuleList()

        self.classifier.append(double_conv(last//2 + last, last//2))
        self.classifier.append(double_conv(last//4 + last//2, last//4))
        self.classifier.append(double_conv(last//8 + last//4, last//8))
        self.classifier.append(nn.Conv2d(last//8, 13, 1))

    def forward(self,x):


        layer1 = self.backbone[0](x)

        layer2 = self.backbone[1](layer1)
        layer3 = self.backbone[2](layer2)

        layer4 = self.backbone[3](layer3)

        out = {}

        out['feature'] = layer4

        x = self.upsample(layer4)
        x = torch.cat([x, layer3], dim=1)

        x = self.classifier[0](x)
        x = self.upsample(x)
        x = torch.cat([x, layer2], dim=1)

        x = self.classifier[1](x)
        x = self.upsample(x)
        x = torch.cat([x, layer1], dim=1)

        x = self.classifier[2](x)


        out['out'] = self.classifier[3](x)


        return out



class ResUNet(nn.Module):
    def __init__(self,backbone,last=512):
        super(ResUNet, self).__init__()

        self.backbone = nn.ModuleList()

        self.backbone.append(
            nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu
            ))
        self.backbone.append(backbone.layer1)
        self.backbone.append(backbone.layer2)
        self.backbone.append(backbone.layer3)
        self.backbone.append(backbone.layer4)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.classifier = nn.ModuleList()

        self.classifier.append(double_conv(last//2 + last, last//2))
        self.classifier.append(double_conv(last//4 + last//2, last//4))
        self.classifier.append(double_conv(last//8 + last//4, last//8))
        self.classifier.append(nn.Conv2d(last//8, 13, 1))

    def forward(self,x):

        stem = self.backbone[0](x)

        layer1 = self.backbone[1](stem)

        layer2 = self.backbone[2](layer1)
        layer3 = self.backbone[3](layer2)

        layer4 = self.backbone[4](layer3)

        out = {}

        out['feature'] = layer4

        x = self.upsample(layer4)
        x = torch.cat([x, layer3], dim=1)

        x = self.classifier[0](x)
        x = self.upsample(x)
        x = torch.cat([x, layer2], dim=1)

        x = self.classifier[1](x)
        x = self.upsample(x)
        x = torch.cat([x, layer1], dim=1)

        x = self.classifier[2](x)


        out['out'] = self.upsample(self.classifier[3](x))


        return out
