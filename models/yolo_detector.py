import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            resnet = resnet50(weights=None)

        self.stage1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )  # /4

        self.stage2 = resnet.layer1  # C2 (/4)
        self.stage3 = resnet.layer2  # C3 (/8)
        self.stage4 = resnet.layer3  # C4 (/16)
        self.stage5 = resnet.layer4  # C5 (/32)

        # Freeze Stage 1 và Stage 2
        for param in self.stage1.parameters():
            param.requires_grad = False
        for param in self.stage2.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.stage1(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        return c3, c4, c5


class FPN(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=256):
        super().__init__()

        # Lateral convolutions: đưa feature maps từ backbone
        # về cùng số kênh (out_channels)
        self.lateral3 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lateral4 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lateral5 = nn.Conv2d(in_channels[2], out_channels, 1)

        # Smooth convolutions: làm mượt feature maps sau khi fusion
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.lateral5(c5)

        p4 = self.lateral4(c4) + nn.functional.interpolate(
            p5, size=c4.shape[2:], mode="nearest"
        )

        p3 = self.lateral3(c3) + nn.functional.interpolate(
            p4, size=c3.shape[2:], mode="nearest"
        )

        p3 = self.smooth3(p3)
        p4 = self.smooth4(p4)
        p5 = self.smooth5(p5)

        return p3, p4, p5


class YOLOHead(nn.Module):
    def __init__(self, num_classes=80, in_channels=256, num_layers=2):
        super().__init__()

        # 1. Nhánh Phân loại (Classification Branch)
        # semantic
        self.cls_convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.SiLU(),
                )
                for _ in range(num_layers)
            ]
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 3, padding=1)

        # 2. Nhánh Định vị (Regression Branch)
        # geometric
        self.reg_convs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.SiLU(),
                )
                for _ in range(num_layers)
            ]
        )
        self.reg_pred = nn.Conv2d(in_channels, 4, 3, padding=1)

        # 3. Nhánh Objectness (Độ tin cậy có vật thể hay không)
        self.obj_pred = nn.Conv2d(in_channels, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        # Khởi tạo bias cho classification & objectness
        for m in [self.cls_pred, self.obj_pred]:
            nn.init.constant_(m.bias, -4.59)  # type: ignore

        # Khởi tạo weight cho các layer còn lại
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Nhánh classification
        x_cls = self.cls_convs(x)
        cls_output = self.cls_pred(x_cls)

        # Nhánh regression + objectness
        x_reg = self.reg_convs(x)
        reg_output = self.reg_pred(x_reg)
        obj_output = self.obj_pred(x_reg)

        return torch.cat([reg_output, obj_output, cls_output], dim=1)


class Detector(nn.Module):
    def __init__(self, backbone, fpn, num_classes=80):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = YOLOHead(num_classes=num_classes)

    def forward(self, x):
        # 1. Backbone Extraction
        c3, c4, c5 = self.backbone(x)

        # 2. FPN Fusion
        p3, p4, p5 = self.fpn(c3, c4, c5)

        # 3. Detection Head (Shared weights across levels)
        out_p3 = self.head(p3)
        out_p4 = self.head(p4)
        out_p5 = self.head(p5)

        return [out_p3, out_p4, out_p5]
