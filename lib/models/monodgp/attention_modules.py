import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialChannelAttention(nn.Module):
    """
    Spatial-Channel Attention Block (SCAB)
    Combines channel attention and spatial attention for feature enhancement
    """
    def __init__(self, channels, reduction=16, init_weight=0.1, from_scratch=True):
        """
        空间-通道注意力模块（SCAB）
        
        参数：
            channels: 输入通道数
            reduction: 通道缩减比例
            init_weight: 初始注意力权重（用于渐进式训练）
            from_scratch: 是否从头训练
                        True: 使用适合从头训练的初始化
                        False: 使用适合预训练微调的初始化
        """
        super(SpatialChannelAttention, self).__init__()
        
        self.channels = channels  # 保存通道数供_init_weights使用
        self.from_scratch = from_scratch
        
        # 可学习的权重参数，控制注意力强度
        self.attention_weight = nn.Parameter(torch.tensor(init_weight))
        
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),  # 确保有bias
            nn.Sigmoid()
        )
        
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=True),  # 确保有bias
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        超激进初始化策略，专为从头训练设计
        关键改进：
        1. 使用Kaiming初始化保证梯度流动
        2. 将最终层初始化为输出≈0.98（而不是0.5），几乎完全保留特征
        3. 避免训练早期的信息流断裂
        
        数学原理：
        - sigmoid(4.0) ≈ 0.982  → 保留98%的特征
        - sigmoid(5.0) ≈ 0.993  → 保留99%的特征
        相比之前的sigmoid(0.0) ≈ 0.5（只保留50%），信息保留提升近2倍
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化，确保前向和反向传播中信号的方差保持稳定
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # ===== 通道注意力最终层：超激进初始化 =====
        # 目标：sigmoid(bias) ≈ 0.98，让通道几乎完全通过
        channel_layers = [m for m in self.channel_attention if isinstance(m, nn.Conv2d)]
        if len(channel_layers) >= 2:
            final_conv = channel_layers[-1]  # 获取最后一个卷积层
            # 权重初始化为接近0，让输出主要由bias控制
            nn.init.constant_(final_conv.weight, 0.0)
            if final_conv.bias is not None:
                # ⚠️ 关键修改：bias从3.0改为4.5，让sigmoid输出从0.95提升到0.989
                nn.init.constant_(final_conv.bias, 4.5)
        
        # ===== 空间注意力最终层：同样的超激进初始化 =====
        spatial_layers = [m for m in self.spatial_attention if isinstance(m, nn.Conv2d)]
        if spatial_layers:
            final_conv = spatial_layers[0]
            nn.init.constant_(final_conv.weight, 0.0)
            if final_conv.bias is not None:
                # ⚠️ 关键修改：bias=4.5，保证空间位置几乎完全保留
                nn.init.constant_(final_conv.bias, 4.5)
    
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