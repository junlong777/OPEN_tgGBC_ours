# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
#  Modified by Jinghua Hou
# ------------------------------------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations

@DETECTORS.register_module()
class OPEN(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=16,
                 position_level=0,
                 single_test=False,
                 pretrained=None):
        super(OPEN, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.prev_active_cams = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level

    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            prev_cams = getattr(self, 'prev_active_cams', None)
            
            # --- 核心修改：Pre-Backbone 绝对切片 & FPN 后置重构 ---
            if prev_cams is not None and not self.training and not training_mode:
                torch.backends.cudnn.benchmark = False 
                
                # 1. 物理拦截：仅高优相机送入普通的 Backbone 和 FPN
                active_img = img[prev_cams].contiguous() 
                img_feats = self.img_backbone(active_img) # 调用标准 Backbone 即可
                
                if isinstance(img_feats, dict):
                    img_feats = list(img_feats.values())
                    
                if self.with_img_neck:
                    img_feats = self.img_neck(img_feats)
                    
                # 2. 干净的数学重构：在 FPN 之后补零，绝不污染 FPN 内部的卷积均值
                reconstructed_feats = []
                num_total_cams = img.size(0) # 保持 6
                for feat in img_feats:
                    C_f, H_f, W_f = feat.shape[1:]
                    full_feat = torch.zeros((num_total_cams, C_f, H_f, W_f), dtype=feat.dtype, device=feat.device)
                    full_feat[prev_cams] = feat
                    reconstructed_feats.append(full_feat)
                img_feats = reconstructed_feats
            else:
                # 首帧或训练时走全量
                img_feats = self.img_backbone(img)
                if isinstance(img_feats, dict):
                    img_feats = list(img_feats.values())
                if self.with_img_neck:
                    img_feats = self.img_neck(img_feats)
        else:
            return None

        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN/B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/len_queue), C, H, W)

        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        img_feats = self.extract_img_feat(img, T, training_mode)
        return img_feats

    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_bboxes=None,
                            gt_labels=None,
                            img_metas=None,
                            centers2d=None,
                            depths=None,
                            gt_bboxes_ignore=None,
                            **data):
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                data_t[key] = data[key][:, i] 

            data_t['img_feats'] = data_t['img_feats']
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                        gt_labels_3d[i], gt_bboxes[i],
                                        gt_labels[i], img_metas[i], centers2d[i], depths[i], requires_grad=requires_grad,return_losses=return_losses,**data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value
        return losses

    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, img_metas, **data):
        outs_roi = self.img_roi_head(img_metas, **data)
        return outs_roi

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        location = self.prepare_location(img_metas, **data)

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(location, img_metas, None, **data)
            self.train()
        else:
            outs_roi = self.forward_roi_head(img_metas, **data)
            topk_indexes = outs_roi['topk_indexes']
            outs = self.pts_bbox_head(outs_roi, img_metas, topk_indexes, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d) 
            return losses
        else:
            return None

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        T = data['img'].size(1)
        prev_img = data['img'][:, :-self.num_frame_backbone_grads]
        rec_img = data['img'][:, -self.num_frame_backbone_grads:]
        rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

        if T-self.num_frame_backbone_grads > 0:
            self.eval()
            with torch.no_grad():
                prev_img_feats = self.extract_feat(prev_img, T-self.num_frame_backbone_grads, True)
            self.train()
            data['img_feats'] = torch.cat([prev_img_feats, rec_img_feats], dim=1)
        else:
            data['img_feats'] = rec_img_feats

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, gt_bboxes,
                        gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)
        return losses
  
    def forward_test(self, img_metas, rescale, **data):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))
        for key in data:
            if key in ['gt_bboxes', 'gt_labels', 'centers2d', 'depths']:
                data[key] = data[key][0]
            elif key == 'gt_bboxes_3d':
                data[key] = data[key][0][0]
            elif key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.img_roi_head.reset_memory()
            self.pts_bbox_head.reset_memory()
            self.prev_active_cams = None
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        outs_roi = self.forward_roi_head(img_metas, **data)
        topk_indexes = outs_roi['topk_indexes']
        outs = self.pts_bbox_head(outs_roi, img_metas, topk_indexes, **data)

        # --- Temporal Routing: Zero-Sync Saliency-Relative (小目标保护版) ---
        scores = getattr(torch, 'tgGBC_latest_scores', None)
        if scores is not None and scores.dim() == 2:  
            scores_mean = scores.mean(dim=0)  
            
            # [6, H*W]
            scores_per_cam = scores_mean.view(6, -1) 

            # --- 核心革新：Top-K 显著性评估 (打破面积霸权，拯救远距离小目标) ---
            # 不再求全图总和，而是提取每个视角最强的 xx 个响应点
            k_core = min(30, scores_per_cam.size(1))
            cam_saliency = scores_per_cam.topk(k_core, dim=1).values.sum(dim=1)

            W = data['img_feats'].size(-1)
            pts_per_cam = scores_per_cam.size(1)
            x_coords = torch.arange(pts_per_cam, device=scores.device) % W
            
            left_mask = x_coords < (W * 0.2)
            right_mask = x_coords >= (W * 0.8)
            zero_pad = torch.tensor(0.0, device=scores.device)
            
            left_scores = torch.where(left_mask.unsqueeze(0), scores_per_cam, zero_pad)
            right_scores = torch.where(right_mask.unsqueeze(0), scores_per_cam, zero_pad)
            
            # 边缘截断区域同样采用 Top-K 提取关键信号，避免被大量边缘空像素稀释
            k_edge = min(10, scores_per_cam.size(1))
            left_saliency = left_scores.topk(k_edge, dim=1).values.sum(dim=1)
            right_saliency = right_scores.topk(k_edge, dim=1).values.sum(dim=1)

            # --- 显著性相对阈值过滤 ---
            mean_saliency = cam_saliency.mean() + 1e-6
            
            # 由于大目标的优势被削弱，均值变得更加真实。
            core_thresh = mean_saliency * 0.35
            core_mask = cam_saliency > core_thresh
            core_cams = torch.nonzero(core_mask).squeeze(1)
            
            # 严守 3D 几何张角的生命线
            if len(core_cams) < 3:
                _, core_cams = torch.topk(cam_saliency, 3)

            # --- 边缘唤醒 ---
            edge_thresh = 0.12  # 边缘区最强的 xx 个点达到均值的 xx% 即可触发唤醒
            left_trigger = left_saliency > (mean_saliency * edge_thresh)
            right_trigger = right_saliency > (mean_saliency * edge_thresh)

            left_adj_tensor = torch.tensor([2, 0, 4, 5, 3, 1], device=scores.device)
            right_adj_tensor = torch.tensor([1, 5, 0, 4, 2, 3], device=scores.device)

            # 仅唤醒活跃相机的物理邻居，阻断噪声相机的链式蔓延
            triggered_left = left_adj_tensor[core_cams][left_trigger[core_cams]]
            triggered_right = right_adj_tensor[core_cams][right_trigger[core_cams]]

            active_cams_tensor = torch.unique(torch.cat([core_cams, triggered_left, triggered_right]))
            self.prev_active_cams = active_cams_tensor.sort().values
        else:
            self.prev_active_cams = None

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]
        return bbox_results
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_img_feat(data['img'], 1)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list