import torch.nn as nn
import torch.nn.functional as F
from .utils import ASPP

class deeplabv2(nn.Module):
    def __init__(self,backbone):
        super(deeplabv2,self).__init__()
        self.backbone = backbone
        self.classifier = ASPP(2048, 13)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.distortion = nn.Linear(2048,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out = {}

        b,c,h,w = x.shape

        out['feature'] = self.backbone(x)['out']
        out['out'] = F.interpolate(self.classifier(out['feature']), size=x.shape[2:], mode="bilinear", align_corners=False)

        out['distortion'] = self.sigmoid(self.distortion(self.avg_pool(out['feature']).view(b,-1)))

        return out
