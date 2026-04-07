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

    def __init__(self, **kwargs):
        super(DynamicResNet, self).__init__(**kwargs)
        self.feat_cache = None  # 初始化时序特征缓存状态

    def forward(self, x, active_cams=None):
        """
        Args:
            x (Tensor): [B_N, C_in, H, W] 当前 $t$ 时刻的原输入图像 (B_N通常是相机数 6)
            active_cams (Tensor | None): [num_active] t-1 时刻保留下的激活性相机的索引
        Returns:
            tuple: (x3, x4) 等按照 out_indices 返回的特征元组
        """
        # --- 退回全图计算（训练、首次调用、或者全部相机高优） ---
        if self.training or active_cams is None or len(active_cams) == x.size(0) or self.feat_cache is None:
            outs = super().forward(x)
            # 全图计算完毕后，脱离计算图缓存最新的深层特征 (0 FLOPs 基础)
            self.feat_cache = tuple(out.detach().clone() for out in outs)
            return outs

        # ---------------- 浅层正常提取：Stem -> layer1 -> layer2 ----------------
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

        # ---------------- 中间阶段：根据 active_cams 挑选存活的相机视角下潜 ----------------
        # x_feat shape: [B_N, C_layer2, H_2, W_2]  (stride=8)
        x2_active = x_feat[active_cams]
        
        # ---------------- 深层局部加速计算与复用：layer3 -> layer4 ----------------
        # --- 增量计算激活的特征 ---
        x3_active = self.layer3(x2_active)
        x4_active = self.layer4(x3_active)
        
        outs = []
        new_cache = []
        cache_idx = 0
        
        if 2 in self.out_indices:
            # 调出上一帧的全部时序特征作为底图，彻底防止自车漂移丢失结构
            x3_full = self.feat_cache[cache_idx].clone()
            # 局部刷新真实的新算力目标
            x3_full[active_cams] = x3_active
            outs.append(x3_full)
            new_cache.append(x3_full.detach().clone())
            cache_idx += 1
            
        if 3 in self.out_indices:
            x4_full = self.feat_cache[cache_idx].clone()
            x4_full[active_cams] = x4_active
            outs.append(x4_full)
            new_cache.append(x4_full.detach().clone())
            cache_idx += 1

        # 同步更新下一帧的缓存状态
        self.feat_cache = tuple(new_cache)
        return tuple(outs)
