import torch
import torch.nn as torch_nn
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES
from mmdet.models.backbones.resnet import ResNet


@BACKBONES.register_module()
class DynamicResNet(ResNet):
    """tgGBC Temporal routing dynamic ResNet backbone.
    
    It extracts global contextual features in shallow layers (stem -> layer1 -> layer2),
    then uses tgGBC-Guided Camera Routing to dynamically drop computation for entire background 
    cameras in deep layers (layer3 -> layer4).
    """

    def forward(self, x, active_cams=None):
        """
        Args:
            x (Tensor): [B_N, C_in, H, W] 当前 $t$ 时刻的原输入图像 (B_N通常是相机数 6)
            active_cams (Tensor | None): [num_active] t-1 时刻保留下的激活性相机的索引
        Returns:
            tuple: (x3, x4) 等按照 out_indices 返回的特征元组
        """
        # --- 边界情况：第一帧退化为全计算（或者未丢弃任何相机） ---
        if self.training or active_cams is None or len(active_cams) == x.size(0):
            return super().forward(x)

        # ---------------- 浅层正常提取：Stem -> layer1 -> layer2 ----------------
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

        # ---------------- 中间阶段：根据 active_cams 挑选存活的相机视角下潜 ----------------
        # x_feat shape: [B_N, C_layer2, H_2, W_2]  (stride=8)
        x2_active = x_feat[active_cams]
        
        # ---------------- 深层局部加速计算与回填：layer3 -> layer4 ----------------
        
        # --- Stage 3 (layer3) ---
        x3_active = self.layer3(x2_active)
        # x3_active shape: [num_active, C_layer3, H_3, W_3]
        
        if 2 in self.out_indices:
            # 还原补零
            B_N = x.size(0)
            x3_full = torch.zeros(B_N, x3_active.size(1), x3_active.size(2), x3_active.size(3), 
                                  dtype=x3_active.dtype, device=x3_active.device)
            x3_full[active_cams] = x3_active
            outs.append(x3_full)
            
        # --- Stage 4 (layer4) ---
        x4_active = self.layer4(x3_active)
        # x4_active shape: [num_active, C_layer4, H_4, W_4]
        
        if 3 in self.out_indices:
            # 还原补零
            B_N = x.size(0)
            x4_full = torch.zeros(B_N, x4_active.size(1), x4_active.size(2), x4_active.size(3), 
                                  dtype=x4_active.dtype, device=x4_active.device)
            x4_full[active_cams] = x4_active
            outs.append(x4_full)

        return tuple(outs)
