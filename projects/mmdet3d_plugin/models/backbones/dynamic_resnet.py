import torch
import torch.nn as torch_nn
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES
from mmdet.models.backbones.resnet import ResNet

@BACKBONES.register_module()
class DynamicResNet(ResNet):
    """tgGBC Temporal routing dynamic ResNet backbone with Spatial Compensation.
    
    It extracts global contextual features in shallow layers (stem -> layer1 -> layer2),
    then uses tgGBC-Guided Camera Routing to dynamically drop computation for entire background 
    cameras in deep layers (layer3 -> layer4), while applying spatial pooling to inactive
    cameras to preserve 3D geometric anchors.
    """

    def __init__(self, **kwargs):
        super(DynamicResNet, self).__init__(**kwargs)

    def forward(self, x, active_cams=None):
        """
        Args:
            x (Tensor): [B_N, C_in, H, W] 当前 $t$ 时刻的原输入图像
            active_cams (Tensor | None): [num_active] 需要进行深层语义计算的相机索引
        """
        # --- 降级到全量计算 (训练、无信号或全激活时) ---
        if self.training or active_cams is None or len(active_cams) == x.size(0):
            return super().forward(x)

        # ---------------- 浅层正常提取：Stem -> layer1 -> layer2 ----------------
        if self.deep_stem:
            x_feat = self.stem(x)
        else:
            x_feat = self.conv1(x)
            x_feat = self.norm1(x_feat)
            x_feat = self.relu(x_feat)
        x_feat = self.maxpool(x_feat)

        for i in range(2):
            layer_name = self.res_layers[i]
            res_layer = getattr(self, layer_name)
            x_feat = res_layer(x_feat)

        # ---------------- 空间切片：区分高优与低优相机 ----------------
        B_N = x.size(0)
        active_list = active_cams.tolist()
        inactive_list = [i for i in range(B_N) if i not in active_list]
        inactive_cams = torch.tensor(inactive_list, device=x.device, dtype=torch.long)
        
        x2_active = x_feat[active_cams]
        x2_inactive = x_feat[inactive_cams]
        
        # ---------------- 深层局部加速计算：layer3 -> layer4 ----------------
        x3_active = self.layer3(x2_active)
        x4_active = self.layer4(x3_active)
        
        outs = []
        
        # ---------------- 空间代偿与张量缝合 ----------------
        if 2 in self.out_indices:
            # 创建物理 6 视角的底图
            x3_full = torch.empty((B_N, x3_active.size(1), x3_active.size(2), x3_active.size(3)), 
                                  dtype=x3_active.dtype, device=x3_active.device)
            # 填入高优相机的真实深层特征
            x3_full[active_cams] = x3_active
            
            # 为被阻断计算的背景相机生成代理特征 (池化对齐长宽，通道末尾补零)
            if len(inactive_cams) > 0:
                x3_proxy = F.adaptive_avg_pool2d(x2_inactive, x3_active.shape[2:])
                c_diff = x3_active.size(1) - x2_inactive.size(1)
                # F.pad 的顺序是从最后向前：(W_left, W_right, H_top, H_bottom, C_front, C_back)
                x3_proxy = F.pad(x3_proxy, (0, 0, 0, 0, 0, c_diff))
                x3_full[inactive_cams] = x3_proxy
                
            outs.append(x3_full)
            
        if 3 in self.out_indices:
            x4_full = torch.empty((B_N, x4_active.size(1), x4_active.size(2), x4_active.size(3)), 
                                  dtype=x4_active.dtype, device=x4_active.device)
            x4_full[active_cams] = x4_active
            
            if len(inactive_cams) > 0:
                x4_proxy = F.adaptive_avg_pool2d(x2_inactive, x4_active.shape[2:])
                c_diff = x4_active.size(1) - x2_inactive.size(1)
                x4_proxy = F.pad(x4_proxy, (0, 0, 0, 0, 0, c_diff))
                x4_full[inactive_cams] = x4_proxy
                
            outs.append(x4_full)

        return tuple(outs)