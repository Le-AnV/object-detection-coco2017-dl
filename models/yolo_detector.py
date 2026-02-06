import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# Backbone
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        self.stage1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.stage2 = resnet.layer1  # /4   (256)
        self.stage3 = resnet.layer2  # /8   (512)
        self.stage4 = resnet.layer3  # /16  (1024)
        self.stage5 = resnet.layer4  # /32  (2048)

        # FREEZE stage1, stage2
        for p in self.stage1.parameters():
            p.requires_grad = False

        for p in self.stage2.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stage1(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5


# Neck
class FPN(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=256):
        super().__init__()

        self.lateral5 = nn.Conv2d(in_channels[2], out_channels, 1)
        self.lateral4 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral3 = nn.Conv2d(in_channels[0], out_channels, 1)

        self.smooth5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")

        return (
            self.smooth3(p3),
            self.smooth4(p4),
            self.smooth5(p5),
        )


# Head YOLO-Style
class DecoupledHead(nn.Module):
    def __init__(self, num_classes=10, in_channels=256):
        super().__init__()

        self.cls_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)

        self.reg_convs = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
        )
        self.reg_pred = nn.Conv2d(in_channels, 4, 1)
        self.obj_pred = nn.Conv2d(in_channels, 1, 1)

        self._init_bias()

    def _init_bias(self):
        nn.init.constant_(self.obj_pred.bias, -4.6)  # type: ignore

    def forward(self, x):
        cls_feat = self.cls_convs(x)
        reg_feat = self.reg_convs(x)

        cls_out = self.cls_pred(cls_feat)
        reg_out = self.reg_pred(reg_feat)
        obj_out = self.obj_pred(reg_feat)

        return torch.cat([reg_out, obj_out, cls_out], dim=1)


class Detector(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=True)
        self.fpn = FPN()
        self.head = DecoupledHead(num_classes=num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        features = self.fpn(c3, c4, c5)

        outputs = []
        for feat in features:
            outputs.append(self.head(feat))

        return outputs  # [P3, P4, P5]
