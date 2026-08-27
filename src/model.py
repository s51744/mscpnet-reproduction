"""
MSCPNet architecture (paper Section III.E, Figure 3, Algorithm 2).

  1. Truncated MobileNetV2 backbone -- keeps only the early, general-purpose
     feature layers (Section III.E: "truncate the pretrained backbone").
  2. Multi-Scale Convolutional PoolFormer (MSCP) Block:
       Fr = Conv1x1(Fb)                                  channel reduction
       F_k = PoolFormer_k( Conv_kxk(Fr) ),  k in {2,3,5}  3 parallel scales
       F_1x1 = Conv1x1(Fr)                                identity-scale branch (no PoolFormer)
       F_concat = Concat(F_1x1, F_2, F_3, F_5)
       F_merged = Conv1x1(F_concat)                       project to 128 ch (Fig. 3)
       F_out = LayerNorm(F_merged)
  3. Classification head: GAP -> FC(128->64) -> ReLU -> FC(64->num_classes),
     channel sizes taken from Figure 3's diagram.

TRUNCATE_AT=7 reproduces the paper's disclosed truncated-backbone FLOPs
(132.47M) almost exactly, which pins down the truncation depth. The paper
does not disclose MSCP block channel widths; REDUCED_CH/BRANCH_CH are set to
match the paper's total FLOPs (315.26M) as closely as possible.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

TRUNCATE_AT = 7
REDUCED_CH = 64        # Fr channels after the Step-2 channel-reduction 1x1 conv
BRANCH_CH = 48         # output channels of each of the 4 parallel branches
MERGED_CH = 128        # channels after concat -> 1x1 conv (Figure 3's "128 -> 64" FC head)
FC_HIDDEN = 64
NUM_CLASSES = 4


class TruncatedMobileNetV2(nn.Module):
    """MobileNetV2 backbone truncated to its first TRUNCATE_AT feature blocks."""

    def __init__(self, truncate_at=TRUNCATE_AT, pretrained=True):
        super().__init__()
        weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        full = tvm.mobilenet_v2(weights=weights)
        self.features = nn.Sequential(*list(full.features.children())[:truncate_at])
        # infer output channel count
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.features(dummy)
        self.out_channels = out.shape[1]

    def forward(self, x):
        return self.features(x)


class PoolFormerBlock(nn.Module):
    """
    Standard PoolFormer block (Yu et al., CVPR 2022 -- ref [14] in the paper):
        x = x + Pool(Norm(x))            token mixing (non-parametric pooling, no attention)
        x = x + ChannelMLP(Norm(x))      per-token channel mixing
    Norm = GroupNorm(1, C) (a.k.a. "LayerNorm2d" for NCHW tensors), matching
    the "Norm -> Pooling -> (+) -> Norm -> Channel MLP -> (+)" diagram in Figure 3.
    """

    def __init__(self, dim, pool_size=3, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)
        self.norm2 = nn.GroupNorm(1, dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

    def forward(self, x):
        x = x + (self.pool(self.norm1(x)) - self.norm1(x))  # pooling token mixer (identity-subtracted, as in PoolFormer)
        x = x + self.mlp(self.norm2(x))
        return x


class MSCPBlock(nn.Module):
    """Multi-Scale Convolutional PoolFormer Block (paper Section III.E.1)."""

    def __init__(self, in_channels, reduced_ch=REDUCED_CH, branch_ch=BRANCH_CH, merged_ch=MERGED_CH):
        super().__init__()
        # Normalizes the backbone's raw feature map before the block. Not in the
        # paper's diagram, but needed for stable training on backbones without
        # BatchNorm (e.g. SqueezeNet) under mixed precision.
        self.input_norm = nn.GroupNorm(1, in_channels)
        self.reduce = nn.Conv2d(in_channels, reduced_ch, 1)

        self.conv2 = nn.Conv2d(reduced_ch, branch_ch, 2, padding=0)
        self.conv3 = nn.Conv2d(reduced_ch, branch_ch, 3, padding=1)
        self.conv5 = nn.Conv2d(reduced_ch, branch_ch, 5, padding=2)
        self.pf2 = PoolFormerBlock(branch_ch)
        self.pf3 = PoolFormerBlock(branch_ch)
        self.pf5 = PoolFormerBlock(branch_ch)

        self.conv1x1 = nn.Conv2d(reduced_ch, branch_ch, 1)  # identity-scale branch, no PoolFormer

        self.project = nn.Conv2d(branch_ch * 4, merged_ch, 1)
        self.norm_out = nn.GroupNorm(1, merged_ch)

    def forward(self, x):
        fr = self.reduce(self.input_norm(x))

        f2 = self.pf2(self.conv2(fr))
        f2 = nn.functional.interpolate(f2, size=fr.shape[-2:], mode="nearest")  # 2x2 conv (no pad) shrinks HxW by 1; re-align for concat
        f3 = self.pf3(self.conv3(fr))
        f5 = self.pf5(self.conv5(fr))
        f1 = self.conv1x1(fr)

        f_concat = torch.cat([f1, f2, f3, f5], dim=1)
        f_merged = self.project(f_concat)
        f_out = self.norm_out(f_merged)
        return f_out


class MSCPNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, truncate_at=TRUNCATE_AT, pretrained=True):
        super().__init__()
        self.backbone = TruncatedMobileNetV2(truncate_at=truncate_at, pretrained=pretrained)
        self.mscp = MSCPBlock(self.backbone.out_channels)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(MERGED_CH, FC_HIDDEN)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(FC_HIDDEN, num_classes)

    def forward(self, x):
        f = self.backbone(x)
        f = self.mscp(f)
        f = self.gap(f).flatten(1)
        z = self.relu(self.fc1(f))
        out = self.fc2(z)
        return out


def build_backbone_with_mscp(name, num_classes=NUM_CLASSES, pretrained=True):
    """Used for the Table-5 ablation: plug the MSCP block onto other pretrained backbones."""
    if name == "densenet121":
        weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.densenet121(weights=weights)
        feat = net.features
        out_ch = 1024
    elif name == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet50(weights=weights)
        feat = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                              net.layer1, net.layer2, net.layer3, net.layer4)
        out_ch = 2048
    elif name == "shufflenet_v2":
        weights = tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.shufflenet_v2_x1_0(weights=weights)
        feat = nn.Sequential(net.conv1, net.maxpool, net.stage2, net.stage3, net.stage4, net.conv5)
        out_ch = 1024
    elif name == "squeezenet":
        weights = tvm.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.squeezenet1_1(weights=weights)
        feat = net.features
        out_ch = 512
    elif name == "mobilenet_v2":
        weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.mobilenet_v2(weights=weights)
        feat = net.features
        out_ch = 1280
    else:
        raise ValueError(name)

    class _Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = feat
            self.mscp = MSCPBlock(out_ch)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(MERGED_CH, FC_HIDDEN)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(FC_HIDDEN, num_classes)

        def forward(self, x):
            f = self.backbone(x)
            f = self.mscp(f)
            f = self.gap(f).flatten(1)
            z = self.relu(self.fc1(f))
            return self.fc2(z)

    return _Wrapped()


def profile_model(model, input_size=(1, 3, 224, 224)):
    from thop import profile
    dummy = torch.zeros(*input_size)
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    return flops, params


if __name__ == "__main__":
    m = MSCPNet(pretrained=False)
    flops, params = profile_model(m)
    print(f"MSCPNet: params={params:,.0f}  FLOPs={flops:,.0f}")
    print(f"  (paper reports: params=998,084  FLOPs=315,258,752)")

    y = m(torch.randn(2, 3, 224, 224))
    print("output shape:", y.shape)
