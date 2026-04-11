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

            # 只给支持时序路由的 Backbone (DynamicResNet) 传递 active_cams
            prev_cams = getattr(self, 'prev_active_cams', None)
            try:
                img_feats = self.img_backbone(img, active_cams=prev_cams)
            except TypeError:
                img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN/B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/len_queue), C, H, W)
            
            # --- 恢复幽灵清洗 (Ghost Masking) ---
            if getattr(self, 'prev_active_cams', None) is not None:
                mask = torch.zeros(int(BN/B/len_queue), dtype=img_feats_reshaped.dtype, device=img_feats_reshaped.device)
                mask[self.prev_active_cams] = 1.0
                # mask shape: [6, 1, 1, 1] 广播到 [B, 6, C, H, W]
                mask = mask.view(1, -1, 1, 1, 1)
                img_feats_reshaped = img_feats_reshaped * mask

        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images and points."""
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
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
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
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
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
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
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
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
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

        # --- Temporal Routing: TGC-Routing (Nucleus / Top-p 架构) ---
        scores = getattr(torch, 'tgGBC_latest_scores', None)
        if scores is not None and scores.dim() == 2:  
            scores_mean = scores.mean(dim=0)  # Shape: [Nk]
            scores_per_cam = scores_mean.chunk(6, dim=0)
            
            # --- 升级 1：能量聚合函数 ---
            # 使用 sum() 代替 mean()，真实反映该区域累积的 3D 注意力质量，防小目标被兑水
            def get_energy(tensor, k=50):
                if tensor.numel() == 0: return torch.tensor(0.0, device=tensor.device)
                actual_k = min(k, tensor.numel())
                return tensor.topk(actual_k).values.sum()

            # 计算每个相机的总能量，并转化为全局概率分布
            cam_energies = torch.stack([get_energy(cam_s) for cam_s in scores_per_cam])
            total_energy = cam_energies.sum() + 1e-6
            cam_probs = cam_energies / total_energy  # Shape: [6], 概率和约等于 1.0

            # --- 升级 2：Nucleus (Top-p) 动态路由 ---
            p_target = 0.92  # 核心超参：保证涵盖当前帧 92% 的注意力能量
            min_k = 2        # 保底视野

            sorted_probs, sorted_indices = torch.sort(cam_probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=0)
            
            # 找到刚刚满足或超过 p_target 的相机数量
            keep_num = (cumsum_probs < p_target).sum().item() + 1
            keep_num = max(keep_num, min_k)
            
            active_cams_list = sorted_indices[:keep_num].tolist()

            # --- 升级 3：基于概率的精准边缘唤醒 ---
            left_adj = {0: 2, 1: 0, 5: 1, 3: 5, 4: 3, 2: 4}
            right_adj = {0: 1, 1: 5, 5: 3, 3: 4, 4: 2, 2: 0}
            
            W = data['img_feats'].size(-1)
            pts_per_cam = scores_per_cam[0].size(0)
            x_coords = torch.arange(pts_per_cam, device=scores.device) % W
            left_mask = x_coords < (W * 0.2)
            right_mask = x_coords >= (W * 0.8)

            # 如果某个相机的边缘区域，独立占据了全局 4% 以上的注意力，强制唤醒相邻相机
            edge_prob_thresh = 0.04  

            for c in range(6):
                cam_scores = scores_per_cam[c]
                left_prob = get_energy(cam_scores[left_mask], k=10) / total_energy
                right_prob = get_energy(cam_scores[right_mask], k=10) / total_energy
                
                if left_prob > edge_prob_thresh:
                    active_cams_list.append(left_adj[c])
                if right_prob > edge_prob_thresh:
                    active_cams_list.append(right_adj[c])

            # 去重、排序并保存
            active_cams_tensor = torch.unique(torch.tensor(active_cams_list, dtype=torch.long, device=scores.device))
            self.prev_active_cams = active_cams_tensor.sort().values
            
            # 临时测速与分布监控仪 (请保留此 print)
            # print(f"🔥 Cams: {self.prev_active_cams.numel()} | Probs: {[round(p.item(), 3) for p in cam_probs]}")
        else:
            self.prev_active_cams = None
        # --------------------------------------------------------------------

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        data['img_feats'] = self.extract_img_feat(data['img'], 1)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    