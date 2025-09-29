import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class Swish(nn.Module):
    """Swish activation (x * sigmoid(x))"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation Block for 3D"""
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excite = nn.Sequential(
            nn.Conv3d(in_channels, se_channels, 1),
            Swish(),
            nn.Conv3d(se_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excite(scale)
        return x * scale

class MBConv3D(nn.Module):
    """Mobile Inverted Bottleneck Convolution for 3D"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate

        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv3d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm3d(expanded_channels),
                Swish()
            )
        else:
            self.expand = None

        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Sequential(
            nn.Conv3d(expanded_channels, expanded_channels, kernel_size,
                     stride=stride, padding=padding, groups=expanded_channels, bias=False),
            nn.BatchNorm3d(expanded_channels),
            Swish()
        )

        # Squeeze-and-Excitation
        self.se = SEBlock3D(expanded_channels, se_ratio)

        # Output projection
        self.project = nn.Sequential(
            nn.Conv3d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        identity = x

        # Expansion
        if self.expand is not None:
            x = self.expand(x)

        # Depthwise + SE
        x = self.depthwise(x)
        x = self.se(x)

        # Projection
        x = self.project(x)

        # Skip connection with drop connect
        if self.use_residual:
            if self.training and self.drop_connect_rate > 0:
                keep_prob = 1 - self.drop_connect_rate
                random_tensor = keep_prob + torch.rand([x.shape[0], 1, 1, 1, 1],
                                                       dtype=x.dtype, device=x.device)
                binary_mask = torch.floor(random_tensor)
                x = x / keep_prob * binary_mask
            x = x + identity

        return x

class EfficientNet3D(nn.Module):
    """3D EfficientNet for Multi-Label Classification"""
    
    def __init__(self, width_mult=1.0, depth_mult=1.0, in_channels=3, 
                 num_classes=14, dropout=0.3, drop_connect_rate=0.2):
        super().__init__()
        
        # Base configuration for EfficientNet-B0
        base_config = [
            # (expand_ratio, channels, num_blocks, stride, kernel_size)
            (1, 16, 1, 1, 3),    # Stage 1
            (6, 24, 2, 2, 3),    # Stage 2
            (6, 40, 2, 2, 3),    # Stage 3
            (6, 80, 3, 2, 3),    # Stage 4
            (6, 112, 3, 1, 3),   # Stage 5
            (6, 192, 4, 2, 3),   # Stage 6
            (6, 320, 1, 1, 3),   # Stage 7
        ]

        # Adjust channels and depth based on multipliers
        def round_channels(channels):
            return int(ceil(channels * width_mult / 8) * 8)

        def round_repeats(repeats):
            return int(ceil(repeats * depth_mult))

        # Stem
        stem_channels = round_channels(32)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(stem_channels),
            Swish()
        )

        # Build MBConv blocks
        self.blocks = nn.ModuleList()
        in_ch = stem_channels
        total_blocks = sum([round_repeats(cfg[2]) for cfg in base_config])
        block_idx = 0

        for expand_ratio, channels, num_blocks, stride, kernel_size in base_config:
            out_ch = round_channels(channels)
            num_blocks = round_repeats(num_blocks)

            for i in range(num_blocks):
                # Drop connect rate scales with block depth
                drop_rate = drop_connect_rate * block_idx / total_blocks
                
                self.blocks.append(
                    MBConv3D(
                        in_ch, out_ch, kernel_size,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand_ratio,
                        drop_connect_rate=drop_rate
                    )
                )
                in_ch = out_ch
                block_idx += 1

        # Head
        head_channels = round_channels(1280)
        self.head = nn.Sequential(
            nn.Conv3d(in_ch, head_channels, 1, bias=False),
            nn.BatchNorm3d(head_channels),
            Swish()
        )

        # Classification
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(head_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def efficientnet_b0_3d(in_channels=3, num_classes=14, dropout=0.3):
    """EfficientNet-B0 for 3D (baseline, fastest)"""
    return EfficientNet3D(1.0, 1.0, in_channels, num_classes, dropout)

def efficientnet_b1_3d(in_channels=3, num_classes=14, dropout=0.3):
    """EfficientNet-B1 for 3D"""
    return EfficientNet3D(1.0, 1.1, in_channels, num_classes, dropout)

def efficientnet_b2_3d(in_channels=3, num_classes=14, dropout=0.3):
    """EfficientNet-B2 for 3D"""
    return EfficientNet3D(1.1, 1.2, in_channels, num_classes, dropout)

def efficientnet_b3_3d(in_channels=3, num_classes=14, dropout=0.4):
    """EfficientNet-B3 for 3D (recommended balance)"""
    return EfficientNet3D(1.2, 1.4, in_channels, num_classes, dropout)

def efficientnet_b4_3d(in_channels=3, num_classes=14, dropout=0.4):
    """EfficientNet-B4 for 3D"""
    return EfficientNet3D(1.4, 1.8, in_channels, num_classes, dropout)

def efficientnet_b5_3d(in_channels=3, num_classes=14, dropout=0.5):
    """EfficientNet-B5 for 3D"""
    return EfficientNet3D(1.6, 2.2, in_channels, num_classes, dropout)

def efficientnet_b6_3d(in_channels=3, num_classes=14, dropout=0.5):
    """EfficientNet-B6 for 3D"""
    return EfficientNet3D(1.8, 2.6, in_channels, num_classes, dropout)

def efficientnet_b7_3d(in_channels=3, num_classes=14, dropout=0.5):
    """EfficientNet-B7 for 3D (highest accuracy, slowest)"""
    return EfficientNet3D(2.0, 3.1, in_channels, num_classes, dropout)
