import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialChannelAttention(nn.Module):
    """
    Spatial-Channel Attention Block (SCAB)
    Combines channel attention and spatial attention for feature enhancement
    """
    def __init__(self, channels, reduction=16, init_weight=0.1):
        super(SpatialChannelAttention, self).__init__()
        
        # Learnable weight to control attention strength
        self.attention_weight = nn.Parameter(torch.tensor(init_weight))
        
        # Channel attention module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Conservative initialization to preserve pre-trained features
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use smaller initialization for attention modules
                if m.out_channels == 1:  # Spatial attention output
                    # Initialize to output near 1.0 (identity-like)
                    nn.init.constant_(m.weight, 0.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)  # sigmoid(0) = 0.5
                else:
                    # Small random initialization for other layers
                    nn.init.normal_(m.weight, 0.0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        # Initialize the final conv layers of attention to output near 1.0
        # This makes SCAB initially behave like identity
        # Access the conv layers (not the Sigmoid activation)
        final_conv_channel = self.channel_attention[-2]  # Second to last layer (Conv2d)
        final_conv_spatial = self.spatial_attention[-2]  # Second to last layer (Conv2d)
        
        # Initialize channel attention final conv
        if hasattr(final_conv_channel, 'weight'):
            nn.init.constant_(final_conv_channel.weight, 0.0)
            if hasattr(final_conv_channel, 'bias') and final_conv_channel.bias is not None:
                nn.init.constant_(final_conv_channel.bias, 0.0)  # sigmoid(0) = 0.5, but we want closer to 1
        
        # Initialize spatial attention final conv
        if hasattr(final_conv_spatial, 'weight'):
            nn.init.constant_(final_conv_spatial.weight, 0.0)
            if hasattr(final_conv_spatial, 'bias') and final_conv_spatial.bias is not None:
                nn.init.constant_(final_conv_spatial.bias, 1.0)  # sigmoid(1) ≈ 0.73
    
    def forward(self, x):
        """
        Args:
            x: input feature map [B, C, H, W]
        Returns:
            enhanced feature map [B, C, H, W]
        """
        # Store original input
        identity = x
        
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial attention
        # Generate spatial attention map using average and max pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        spatial_att = self.spatial_attention(spatial_input)  # [B, 1, H, W]
        x = x * spatial_att
        
        # Weighted residual connection for gradual learning
        attention_effect = x - identity
        x = identity + self.attention_weight * attention_effect
        
        return x


class MultiScaleSCAB(nn.Module):
    """
    Multi-scale Spatial-Channel Attention Block
    Applies SCAB with different kernel sizes for multi-scale feature enhancement
    """
    def __init__(self, channels, reduction=16, scales=[1, 3, 5]):
        super(MultiScaleSCAB, self).__init__()
        
        self.scales = scales
        self.scabs = nn.ModuleList([
            SpatialChannelAttention(channels, reduction) for _ in scales
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * len(scales), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: input feature map [B, C, H, W]
        Returns:
            multi-scale enhanced feature map [B, C, H, W]
        """
        scale_features = []
        
        for i, scab in enumerate(self.scabs):
            if self.scales[i] == 1:
                scale_features.append(scab(x))
            else:
                # Apply different kernel sizes by using different padding
                kernel_size = self.scales[i]
                padding = kernel_size // 2
                pooled = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
                enhanced = scab(pooled)
                scale_features.append(enhanced)
        
        # Concatenate and fuse
        fused = torch.cat(scale_features, dim=1)
        output = self.fusion(fused)
        
        return output + x  # Residual connection


class LightweightSCAB(nn.Module):
    """
    Lightweight version of SCAB for efficiency
    """
    def __init__(self, channels, reduction=16):
        super(LightweightSCAB, self).__init__()
        
        # Shared conv for efficiency
        self.shared_conv = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        
        # Channel attention
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention (simplified)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels // reduction, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Shared feature extraction
        shared_feat = F.relu(self.shared_conv(x))
        
        # Channel attention
        channel_att = self.channel_fc(shared_feat)
        
        # Spatial attention  
        spatial_att = self.spatial_conv(shared_feat)
        
        # Apply attention
        out = x * channel_att * spatial_att
        
        return out + x  # Residual connection