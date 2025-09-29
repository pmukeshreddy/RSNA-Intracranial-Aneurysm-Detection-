import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1),
            Swish(),
            nn.Conv3d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        se = self.squeeze(x)
        se = self.excitation(se)
        return x * se

class MBConv3D(nn.Module):
    """Mobile Inverted Bottleneck Convolution in 3D"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate
        
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                Swish()
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                     stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            Swish()
        ])
        
        self.conv = nn.Sequential(*layers)
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            self.se = SEBlock3D(hidden_dim, reduction=max(1, int(in_channels * se_ratio)))
        else:
            self.se = nn.Identity()
        
        # Output projection
        self.project = nn.Sequential(
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        x = self.conv(x)
        x = self.se(x)
        x = self.project(x)
        
        # Stochastic depth / drop connect
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
    """3D EfficientNet for medical image classification"""
    
    def __init__(self, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2,
                 in_channels=3, num_classes=14):
        super().__init__()
        
        # Base configuration [expansion, channels, repeats, stride, kernel_size]
        base_config = [
            [1, 16, 1, 1, 3],   # Stage 1
            [6, 24, 2, 2, 3],   # Stage 2
            [6, 40, 2, 2, 5],   # Stage 3
            [6, 80, 3, 2, 3],   # Stage 4
            [6, 112, 3, 1, 5],  # Stage 5
            [6, 192, 4, 2, 5],  # Stage 6
            [6, 320, 1, 1, 3],  # Stage 7
        ]
        
        # Scale channels and depth
        def round_filters(filters, multiplier=width_mult, divisor=8):
            filters *= multiplier
            new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
            if new_filters < 0.9 * filters:
                new_filters += divisor
            return int(new_filters)
        
        def round_repeats(repeats, multiplier=depth_mult):
            return int(math.ceil(multiplier * repeats))
        
        # Stem
        out_channels = round_filters(32)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, 
                     padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            Swish()
        )
        
        # Build blocks
        self.blocks = nn.ModuleList([])
        in_ch = out_channels
        
        total_blocks = sum([round_repeats(r) for _, _, r, _, _ in base_config])
        block_idx = 0
        
        for expand_ratio, channels, repeats, stride, kernel_size in base_config:
            out_ch = round_filters(channels)
            repeats = round_repeats(repeats)
            
            for i in range(repeats):
                # Drop connect rate increases linearly
                drop_rate = dropout_rate * block_idx / total_blocks
                
                self.blocks.append(
                    MBConv3D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand_ratio,
                        se_ratio=0.25,
                        drop_connect_rate=drop_rate
                    )
                )
                in_ch = out_ch
                block_idx += 1
        
        # Head
        final_channels = round_filters(1280)
        self.head = nn.Sequential(
            nn.Conv3d(in_ch, final_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(final_channels),
            Swish()
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(final_channels, num_classes)
        
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
        x = self.classifier(x)
        
        return x

# Model variants
def efficientnet3d_b0(in_channels=3, num_classes=14):
    """EfficientNet-B0 for 3D (lightest)"""
    return EfficientNet3D(width_mult=1.0, depth_mult=1.0, dropout_rate=0.2,
                          in_channels=in_channels, num_classes=num_classes)

def efficientnet3d_b1(in_channels=3, num_classes=14):
    """EfficientNet-B1 for 3D"""
    return EfficientNet3D(width_mult=1.0, depth_mult=1.1, dropout_rate=0.2,
                          in_channels=in_channels, num_classes=num_classes)

def efficientnet3d_b2(in_channels=3, num_classes=14):
    """EfficientNet-B2 for 3D"""
    return EfficientNet3D(width_mult=1.1, depth_mult=1.2, dropout_rate=0.3,
                          in_channels=in_channels, num_classes=num_classes)

def efficientnet3d_b3(in_channels=3, num_classes=14):
    """EfficientNet-B3 for 3D (sweet spot)"""
    return EfficientNet3D(width_mult=1.2, depth_mult=1.4, dropout_rate=0.3,
                          in_channels=in_channels, num_classes=num_classes)

def efficientnet3d_b4(in_channels=3, num_classes=14):
    """EfficientNet-B4 for 3D"""
    return EfficientNet3D(width_mult=1.4, depth_mult=1.8, dropout_rate=0.4,
                          in_channels=in_channels, num_classes=num_classes)
