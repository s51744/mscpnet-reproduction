"""Baseline classifiers used for Table 4 (main comparison) and Table 9 (truncation ablation)."""
import torch.nn as nn
import torchvision.models as tvm

NUM_CLASSES = 4


def build_pretrained_classifier(name, num_classes=NUM_CLASSES, pretrained=True):
    """Standard ImageNet-pretrained backbones fine-tuned with a fresh classifier head."""
    if name == "densenet121":
        weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.densenet121(weights=weights)
        net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    elif name == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "shufflenet_v2":
        weights = tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.shufflenet_v2_x1_0(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "squeezenet":
        weights = tvm.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.squeezenet1_1(weights=weights)
        net.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        net.num_classes = num_classes
    elif name == "mobilenet_v2":
        weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.mobilenet_v2(weights=weights)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone: {name}")
    return net


class TruncatedMobileNetV2Classifier(nn.Module):
    """Truncated MobileNetV2 + plain GAP + FC head (no MSCP block) -- the
    'Truncated MobileNetV2' row of Table 4, used to isolate the truncation
    effect from the MSCP block effect."""

    def __init__(self, num_classes=NUM_CLASSES, truncate_at=7, pretrained=True):
        super().__init__()
        from model import TruncatedMobileNetV2
        self.backbone = TruncatedMobileNetV2(truncate_at=truncate_at, pretrained=pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        f = self.gap(f).flatten(1)
        return self.fc(f)
