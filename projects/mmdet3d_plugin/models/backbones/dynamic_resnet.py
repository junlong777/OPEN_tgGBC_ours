import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from mmdet.models.builder import BACKBONES
from mmdet.models.backbones.resnet import ResNet


@BACKBONES.register_module()
class DynamicResNet(ResNet):
    """Temporal routing dynamic ResNet backbone.
    
    It extracts global contextual features in shallow layers (stem -> layer1 -> layer2),
    then uses Temporal-Guided RoI Patches to dynamically skip computation for background 
    in deep layers (layer3 -> layer4).
    """

    def __init__(self, roi_size=(32, 32), **kwargs):
        super(DynamicResNet, self).__init__(**kwargs)
        self.roi_size = roi_size  # RoIAlign 提取时的固定分辨率尺寸

    def scatter_back(self, patches, rois, target_shape, stride_ratio):
        """
        特征回填 (Scatter): 将局部特征块贴回全图零张量的对应坐标中。
        
        Args:
            patches (Tensor):  [num_rois, C, H_p, W_p] 深度层处理后的离散特征块
            rois (Tensor):     [num_rois, 5] (batch_idx, x1, y1, x2, y2) 发生在原图分辨率 (stride=1) 
            target_shape (tuple): (B, C, H_f, W_f) 期望恢复的完整特征图 Shape
            stride_ratio (int): 当前特征图相对于原图的空间下采样倍数 (如 layer3 为 16, layer4 为 32)
            
        Returns:
            out_feat (Tensor): [B, C, H_f, W_f] 重组后的完整特征图
        """
        B, C, H_f, W_f = target_shape
        device = patches.device
        dtype = patches.dtype
        
        # 1. 创建零填充背景张量
        # Shape: [B, C, H_f, W_f]
        out_feat = torch.zeros(target_shape, dtype=dtype, device=device)
        
        for i in range(rois.size(0)):
            b_idx = int(rois[i, 0].item())
            x1, y1, x2, y2 = rois[i, 1:]
            
            # 2. 坐标转换：原图尺度 -> 目标特征图尺度
            x1_f = max(0, int(x1 / stride_ratio))
            y1_f = max(0, int(y1 / stride_ratio))
            x2_f = min(W_f, int(torch.ceil(x2 / stride_ratio)))
            y2_f = min(H_f, int(torch.ceil(y2 / stride_ratio)))
            
            w_f = x2_f - x1_f
            h_f = y2_f - y1_f
            
            # 无效 RoI 越界或太小，跳过
            if w_f <= 0 or h_f <= 0:
                continue
                
            # 3. 动态插值：将固定的 Patch size 缩放回现实特征图中的真实占地大小
            # patch shape: [1, C, H_p, W_p]
            patch = patches[i:i+1] 
            # patch_resized shape: [1, C, h_f, w_f]
            patch_resized = F.interpolate(patch, size=(h_f, w_f), mode='bilinear', align_corners=False)
            
            # 4. 回填处理：多视角中可能存在重叠的 bbox (也可能是单个目标有多个 anchor bbox)
            # 使用 torch.maximum 自动保留置信度/响应最强的高频特征
            out_feat[b_idx:b_idx+1, :, y1_f:y2_f, x1_f:x2_f] = torch.maximum(
                out_feat[b_idx:b_idx+1, :, y1_f:y2_f, x1_f:x2_f],
                patch_resized
            )
            
        return out_feat

    def forward(self, x, prev_rois=None):
        """
        Args:
            x (Tensor): [B, C_in, H, W] 当前 $t$ 时刻的原输入图像
            prev_rois (Tensor | None): [num_rois, 5] t-1 时刻保留的带有 Batch Idx 的 RoIs
        Returns:
            tuple: (x3, x4) 等按照 out_indices 返回的特征元组
        """
        # --- 边界情况：第一帧退化为全计算（或者未检测到目标的保守策略） ---
        if prev_rois is None or prev_rois.size(0) == 0:
            return super().forward(x)

        # ---------------- 浅层正常提取：Stem -> layer1 -> layer2 ----------------
        # 结果下采样率: Stem & layer1 (x4), layer2 (x8)
        outs = []
        if self.deep_stem:
            x_feat = self.stem(x)
        else:
            x_feat = self.conv1(x)
            x_feat = self.norm1(x_feat)
            x_feat = self.relu(x_feat)
        x_feat = self.maxpool(x_feat)

        # 遍历前两个 Stage (全球感知)
        for i in range(2):
            layer_name = self.res_layers[i]
            res_layer = getattr(self, layer_name)
            x_feat = res_layer(x_feat)
            if i in self.out_indices:
                outs.append(x_feat)

        # ---------------- 中间阶段：根据 RoI 执行特征块萃取 ----------------
        # 此时 x_feat 是 layer2 的输出, spatial stride = 8
        spatial_scale = 1.0 / 8.0 
        
        # 裁剪出 Patch 特征池
        # patched_feat shape: [num_rois, C_layer2, roi_size[0], roi_size[1]]
        patched_feat = roi_align(
            x_feat, prev_rois, 
            output_size=self.roi_size, 
            spatial_scale=spatial_scale, 
            aligned=True
        )

        # 记录全图的 Shape (用于后期 Scatter)
        B, _, H_2, W_2 = x_feat.shape
        D_3 = self.layer3[-1].conv3.out_channels if hasattr(self.layer3[-1], 'conv3') else self.layer3[-1].conv2.out_channels
        D_4 = self.layer4[-1].conv3.out_channels if hasattr(self.layer4[-1], 'conv3') else self.layer4[-1].conv2.out_channels
        target_shape_l3 = (B, D_3, H_2 // 2, W_2 // 2) # layer3 经过 stride=2 收缩下采样
        target_shape_l4 = (B, D_4, H_2 // 4, W_2 // 4) # layer4 经过 stride=2 收缩下采样
        
        # ---------------- 深层局部加速计算与回填：layer3 -> layer4 ----------------
        
        # --- Stage 3 (layer3) ---
        patched_feat_l3 = self.layer3(patched_feat)
        # patched_feat_l3 shape: [num_rois, C_layer3, roi_size[0]/2, roi_size[1]/2]
        
        if 2 in self.out_indices:
            feat_l3 = self.scatter_back(patched_feat_l3, prev_rois, target_shape_l3, stride_ratio=16)
            outs.append(feat_l3)
            
        # --- Stage 4 (layer4) ---
        patched_feat_l4 = self.layer4(patched_feat_l3)
        # patched_feat_l4 shape: [num_rois, C_layer4, roi_size[0]/4, roi_size[1]/4]
        
        if 3 in self.out_indices:
            feat_l4 = self.scatter_back(patched_feat_l4, prev_rois, target_shape_l4, stride_ratio=32)
            outs.append(feat_l4)

        return tuple(outs)
