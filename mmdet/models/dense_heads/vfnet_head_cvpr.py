# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob
from mmcv.ops import DeformConv2d
from mmcv.runner import force_fp32

from mmdet.core import (bbox2distance, bbox_overlaps, build_anchor_generator,
                        build_assigner, build_sampler, distance2bbox,
                        multi_apply, multiclass_nms, reduce_mean)
from ..builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import ATSSHead
from mmdet.models.dense_heads.atss_head_cvpr import ATSSHead_cvpr
from .fcos_head import FCOSHead

from mmcv.ops import CornerPool, batched_nms
from ..utils import gaussian_radius, gen_gaussian_target
from math import ceil
# from mmcv.utils import deprecated_api_warning, ext_loader
import sys
from torch.nn import functional as F

INF = 1e8

class BiCornerPool(nn.Module):
    """Bidirectional Corner Pooling Module (TopLeft, BottomRight, etc.)

    Args:
        in_channels (int): Input channels of module.
        out_channels (int): Output channels of module.
        feat_channels (int): Feature channels of module.
        directions (list[str]): Directions of two CornerPools.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 in_channels,
                 directions,
                 feat_channels=256,
                 out_channels=256,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(BiCornerPool, self).__init__()
        self.direction1_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        self.direction2_conv = ConvModule(
            in_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)

        self.aftpool_conv = ConvModule(
            feat_channels,
            out_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.conv1 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.conv2 = ConvModule(
            in_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg)

        self.direction1_pool = CornerPool(directions[0])
        self.direction2_pool = CornerPool(directions[1])
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        """Forward features from the upstream network.

        Args:
            x (tensor): Input feature of BiCornerPool.

        Returns:
            conv2 (tensor): Output feature of BiCornerPool.
        """
        direction1_conv = self.direction1_conv(x)
        direction2_conv = self.direction2_conv(x)
        direction1_feat = self.direction1_pool(direction1_conv)
        direction2_feat = self.direction2_pool(direction2_conv)
        aftpool_conv = self.aftpool_conv(direction1_feat + direction2_feat)
        conv1 = self.conv1(x)
        relu = self.relu(aftpool_conv + conv1)
        conv2 = self.conv2(relu)

        return conv2

@HEADS.register_module()
class VFNetHead_cvpr(ATSSHead, FCOSHead):
    """Head of `VarifocalNet (VFNet): An IoU-aware Dense Object
    Detector.<https://arxiv.org/abs/2008.13367>`_.

    The VFNet predicts IoU-aware classification scores which mix the
    object presence confidence and object localization accuracy as the
    detection score. It is built on the FCOS architecture and uses ATSS
    for defining positive/negative training examples. The VFNet is trained
    with Varifocal Loss and empolys star-shaped deformable convolution to
    extract features for a bbox.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        sync_num_pos (bool): If true, synchronize the number of positive
            examples across GPUs. Default: True
        gradient_mul (float): The multiplier to gradients from bbox refinement
            and recognition. Default: 0.1.
        bbox_norm_type (str): The bbox normalization type, 'reg_denom' or
            'stride'. Default: reg_denom
        loss_cls_fl (dict): Config of focal loss.
        use_vfl (bool): If true, use varifocal loss for training.
            Default: True.
        loss_cls (dict): Config of varifocal loss.
        loss_bbox (dict): Config of localization loss, GIoU Loss.
        loss_bbox (dict): Config of localization refinement loss, GIoU Loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32,
            requires_grad=True).
        use_atss (bool): If true, use ATSS to define positive/negative
            examples. Default: True.
        anchor_generator (dict): Config of anchor generator for ATSS.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = VFNetHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, bbox_pred_refine= self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 sync_num_pos=True,
                 gradient_mul=0.1,
                 bbox_norm_type='reg_denom',
                 loss_cls_fl=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 use_vfl=True,
                 loss_cls=dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0,
                     iou_weighted=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
                 loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 use_atss=True,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     ratios=[1.0],
                     octave_base_scale=8,
                     scales_per_octave=1,
                     center_offset=0.0,
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='vfnet_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        # dcn base offsets, adapted from reppoints_head.py
        self.num_dconv_points = 9
        self.dcn_kernel = int(np.sqrt(self.num_dconv_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        super(FCOSHead, self).__init__(
            num_classes,
            in_channels,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.regress_ranges = regress_ranges
        self.reg_denoms = [
            regress_range[-1] for regress_range in regress_ranges
        ]
        self.reg_denoms[-1] = self.reg_denoms[-2] * 2
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.sync_num_pos = sync_num_pos
        self.bbox_norm_type = bbox_norm_type
        self.gradient_mul = gradient_mul
        self.use_vfl = use_vfl
        if self.use_vfl:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = build_loss(loss_cls_fl)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)

        # for getting ATSS targets
        self.use_atss = use_atss
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.anchor_center_offset = anchor_generator['center_offset']
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.loss_heatmap = build_loss(dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1))

    def _init_layers(self):
        """Initialize layers of the head."""
        super(FCOSHead, self)._init_cls_convs()
        super(FCOSHead, self)._init_reg_convs()
        self.relu = nn.ReLU(inplace=True)
        self.vfnet_reg_conv = ConvModule(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.conv_bias)
        self.vfnet_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_reg_refine_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_reg_refine = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales_refine = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_cls_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

        self.conv1 = ConvModule(
            5*self.feat_channels, self.feat_channels, 1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=None)

        self._init_corner_kpt_layers()

    def _init_corner_kpt_layers(self):
        """Initialize corner keypoint layers.

        Including corner heatmap branch and corner offset branch. Each branch
        has two parts: prefix `tl_` for top-left and `br_` for bottom-right.
        """
        self.tl_pool = BiCornerPool(
            self.in_channels, ['top', 'left'],
            out_channels=self.in_channels)
        self.br_pool= BiCornerPool(
                self.in_channels, ['bottom', 'right'],
                out_channels=self.in_channels)
        self.tl_heat = self._make_layers(
                out_channels=self.num_classes,
                in_channels=self.in_channels)
        self.br_heat = self._make_layers(
                out_channels=self.num_classes,
                in_channels=self.in_channels)

        bias_init = bias_init_with_prob(0.1)
        self.tl_heat[-1].conv.reset_parameters()
        self.tl_heat[-1].conv.bias.data.fill_(bias_init)
        self.br_heat[-1].conv.reset_parameters()
        self.br_heat[-1].conv.bias.data.fill_(bias_init)

    def _make_layers(self, out_channels, in_channels=256, feat_channels=256):
        """Initialize conv sequential for CornerHead."""
        return nn.Sequential(
            ConvModule(in_channels, feat_channels, 3, norm_cfg=dict(type='BN', requires_grad=True), padding=1),
            ConvModule(
                feat_channels, out_channels, 1, norm_cfg=None, act_cfg=None))

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box iou-aware scores for each scale
                    level, each is a 4D-tensor, the channel number is
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box offsets for each
                    scale level, each is a 4D-tensor, the channel number is
                    num_points * 4.
                bbox_preds_refine (list[Tensor]): Refined Box offsets for
                    each scale level, each is a 4D-tensor, the channel
                    number is num_points * 4.
        """
        w, h = feats[0].size(2), feats[0].size(3)
        img_feat = []
        # img_feat = 0
        for feat in feats:
            img_feat.append(F.interpolate(feat, size=(w, h), mode='bilinear'))
            # img_feat += F.interpolate(feat, size=(w, h), mode='bilinear')
        img_feat = torch.cat(img_feat, dim=1)

        img_feat = self.conv1(img_feat)
        tl_heat = self.tl_pool(img_feat)
        tl_heat = self.tl_heat(tl_heat)
        br_heat = self.br_pool(img_feat)
        br_heat = self.br_heat(br_heat)
        return multi_apply(self.forward_single, feats, self.scales,
                           self.scales_refine, self.strides, self.reg_denoms)\
               +([tl_heat for i in range(len(feats))], [br_heat for j in range(len(feats))])

    def forward_single(self, x, scale, scale_refine, stride, reg_denom):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            scale_refine (:obj: `mmcv.cnn.Scale`): Learnable scale module to
                resize the refined bbox prediction.
            stride (int): The corresponding stride for feature maps,
                used to normalize the bbox prediction when
                bbox_norm_type = 'stride'.
            reg_denom (int): The corresponding regression range for feature
                maps, only used to normalize the bbox prediction when
                bbox_norm_type = 'reg_denom'.

        Returns:
            tuple: iou-aware cls scores for each box, bbox predictions and
                refined bbox predictions of input feature maps.
        """
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        # predict the bbox_pred of different level
        reg_feat_init = self.vfnet_reg_conv(reg_feat)
        if self.bbox_norm_type == 'reg_denom':
            bbox_pred = scale(
                self.vfnet_reg(reg_feat_init)).float().exp() * reg_denom
        elif self.bbox_norm_type == 'stride':
            bbox_pred = scale(
                self.vfnet_reg(reg_feat_init)).float().exp() * stride
        else:
            raise NotImplementedError

        # compute star deformable convolution offsets
        # converting dcn_offset to reg_feat.dtype thus VFNet can be
        # trained with FP16
        dcn_offset = self.star_dcn_offset(bbox_pred, self.gradient_mul,
                                          stride).to(reg_feat.dtype)

        # refine the bbox_pred
        reg_feat = self.relu(self.vfnet_reg_refine_dconv(reg_feat, dcn_offset))
        bbox_pred_refine = scale_refine(
            self.vfnet_reg_refine(reg_feat)).float().exp()
        bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()

        # predict the iou-aware cls score
        cls_feat = self.relu(self.vfnet_cls_dconv(cls_feat, dcn_offset))
        cls_score = self.vfnet_cls(cls_feat)

        return cls_score, bbox_pred, bbox_pred_refine

    def star_dcn_offset(self, bbox_pred, gradient_mul, stride):
        """Compute the star deformable conv offsets.

        Args:
            bbox_pred (Tensor): Predicted bbox distance offsets (l, r, t, b).
            gradient_mul (float): Gradient multiplier.
            stride (int): The corresponding stride for feature maps,
                used to project the bbox onto the feature map.

        Returns:
            dcn_offsets (Tensor): The offsets for deformable convolution.
        """
        dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred)
        bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + \
            gradient_mul * bbox_pred
        # map to the feature map scale
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, H, W = bbox_pred.size()

        x1 = bbox_pred_grad_mul[:, 0, :, :]
        y1 = bbox_pred_grad_mul[:, 1, :, :]
        x2 = bbox_pred_grad_mul[:, 2, :, :]
        y2 = bbox_pred_grad_mul[:, 3, :, :]
        bbox_pred_grad_mul_offset = bbox_pred.new_zeros(
            N, 2 * self.num_dconv_points, H, W)
        bbox_pred_grad_mul_offset[:, 0, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 1, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 2, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 4, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 5, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 7, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 11, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 12, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 13, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 14, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 16, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 17, :, :] = x2  # x2
        dcn_offset = bbox_pred_grad_mul_offset - dcn_base_offset

        return dcn_offset

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine', 'tl_heat', 'br_heat'))
    def loss(self,
             cls_scores,
             bbox_preds,
             bbox_preds_refine,
             tl_heat,
             br_heat,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box offsets for each
                scale level, each is a 4D-tensor, the channel number is
                num_points * 4.
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level, each is a 4D-tensor, the channel
                number is num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, label_weights, bbox_targets, bbox_weights, br_heatmap_all, tl_heatmap_all  = self.get_targets(
            cls_scores, all_level_points, gt_bboxes, gt_labels, img_metas,
            gt_bboxes_ignore)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and bbox_preds_refine
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3,
                              1).reshape(-1,
                                         self.cls_out_channels).contiguous()
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds_refine = [
            bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
            for bbox_pred_refine in bbox_preds_refine
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds_refine)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes - 1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.where(
            ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)) > 0)[0]
        num_pos = len(pos_inds)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds]
        pos_labels = flatten_labels[pos_inds]

        # sync num_pos across all gpus
        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                pos_inds.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]

        pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
        pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)
        iou_targets_ini = bbox_overlaps(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_ini = iou_targets_ini.clone().detach()
        bbox_avg_factor_ini = reduce_mean(
            bbox_weights_ini.sum()).clamp_(min=1).item()

        pos_decoded_bbox_preds_refine = \
            distance2bbox(pos_points, pos_bbox_preds_refine)
        iou_targets_rf = bbox_overlaps(
            pos_decoded_bbox_preds_refine,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_rf = iou_targets_rf.clone().detach()
        bbox_avg_factor_rf = reduce_mean(
            bbox_weights_rf.sum()).clamp_(min=1).item()

        new_gt_br_heatmap = []
        new_gt_tl_heatmap = []
        for br_heatmap in zip(br_heatmap_all[0],br_heatmap_all[1]):
            br_heatmap_temp = list(br_heatmap)
            br_heatmap_temp = torch.stack(br_heatmap_temp,dim=0)
            new_gt_br_heatmap.append(br_heatmap_temp)

        for tl_heatmap in zip(tl_heatmap_all[0],tl_heatmap_all[1]):
            tl_heatmap_temp = list(tl_heatmap)
            tl_heatmap_temp = torch.stack(tl_heatmap_temp,dim=0)
            new_gt_tl_heatmap.append(tl_heatmap_temp)


        if num_pos > 0:
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_ini,
                avg_factor=bbox_avg_factor_ini)

            loss_bbox_refine = self.loss_bbox_refine(
                pos_decoded_bbox_preds_refine,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_rf,
                avg_factor=bbox_avg_factor_rf)

            # build IoU-aware cls_score targets
            if self.use_vfl:
                pos_ious = iou_targets_rf.clone().detach()
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                cls_iou_targets[pos_inds, pos_labels] = pos_ious

            # Detection loss
            tl_det_loss = self.loss_heatmap(
                tl_heat[0].sigmoid(),
                new_gt_tl_heatmap[0],
                avg_factor=max(1,
                               new_gt_tl_heatmap[0].eq(1).sum()))
            br_det_loss = self.loss_heatmap(
                tl_heat[0].sigmoid(),
                new_gt_br_heatmap[0],
                avg_factor=max(1,
                               new_gt_br_heatmap[0].eq(1).sum()))

            det_loss = 0.1 * (tl_det_loss + br_det_loss) / 2.0

        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            loss_bbox_refine = pos_bbox_preds_refine.sum() * 0
            if self.use_vfl:
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
            det_loss = torch.tensor(0).to(loss_bbox)

        if self.use_vfl:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                cls_iou_targets,
                avg_factor=num_pos_avg_per_gpu)
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels,
                weight=label_weights,
                avg_factor=num_pos_avg_per_gpu)


        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_rf=loss_bbox_refine,
            loss_det=det_loss)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'bbox_preds_refine','cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   bbox_preds_refine,
                   tl_heat,
                   br_heat,
                   img_metas,
                   cfg=None,
                   rescale=None,
                   with_nms=True):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box offsets for each scale
                level with shape (N, num_points * 4, H, W).
            bbox_preds_refine (list[Tensor]): Refined Box offsets for
                each scale level with shape (N, num_points * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before returning boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            tl_heat_list = [
                tl_heat[i][img_id].detach() for i in range(num_levels)
            ]
            br_heat_list = [
                br_heat[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list, mlvl_points,
                                                 tl_heat_list,
                                                 br_heat_list,
                                                 img_shape, scale_factor, cfg,
                                                 rescale, with_nms)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_points,
                           tl_heat_list,
                           br_heat_list,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for a single scale
                level with shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box offsets for a single scale
                level with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before returning boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_tl_heat = []
        mlvl_br_heat = []
        for cls_score, bbox_pred, points in zip(cls_scores, bbox_preds,
                                                mlvl_points):

            tl_heat = tl_heat_list[0]
            br_heat = br_heat_list[0]
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).contiguous().sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4).contiguous()

            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)

            stride_bboxes = torch.zeros_like(bboxes)
            stride_bboxes[:, [0, 2]] = torch.clamp((tl_heat.size(2) * bboxes[:, [0, 2]] / img_shape[1]), min=0, max=tl_heat.size(2)-1)
            stride_bboxes[:, [1, 3]] = torch.clamp(tl_heat.size(1) * bboxes[:, [1, 3]] / img_shape[0], min=0, max=tl_heat.size(1)-1)

            tl_heat_bboxes = tl_heat[:, stride_bboxes[:, 1].long(), stride_bboxes[:, 0].long()].transpose(1, 0)
            br_heat_bboxes = br_heat[:, stride_bboxes[:, 3].long(), stride_bboxes[:, 2].long()].transpose(1, 0)

            mlvl_tl_heat.append(tl_heat_bboxes)
            mlvl_br_heat.append(br_heat_bboxes)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_tl_heat = torch.cat(mlvl_tl_heat).sigmoid()
        mlvl_br_heat = torch.cat(mlvl_br_heat).sigmoid()
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if with_nms:
            det_bboxes, det_labels = self.multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    mlvl_tl_heat, mlvl_br_heat,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        # to be compatible with anchor points in ATSS
        if self.use_atss:
            points = torch.stack(
                (x.reshape(-1), y.reshape(-1)), dim=-1) + \
                     stride * self.anchor_center_offset
        else:
            points = torch.stack(
                (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def get_targets(self, cls_scores, mlvl_points, gt_bboxes, gt_labels,
                    img_metas, gt_bboxes_ignore):
        """A wrapper for computing ATSS and FCOS targets for points in multiple
        images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor/None): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor/None): Bbox weights of all levels.
        """
        if self.use_atss:
            return self.get_atss_targets(cls_scores, mlvl_points, gt_bboxes,
                                         gt_labels, img_metas,
                                         gt_bboxes_ignore)
        else:
            self.norm_on_bbox = False
            return self.get_fcos_targets(mlvl_points, gt_bboxes, gt_labels)

    def _get_target_single(self, *args, **kwargs):
        """Avoid ambiguity in multiple inheritance."""
        if self.use_atss:
            return ATSSHead_cvpr._get_target_single(self, *args, **kwargs)
        else:
            return FCOSHead._get_target_single(self, *args, **kwargs)

    def get_fcos_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute FCOS regression and classification targets for points in
        multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                labels (list[Tensor]): Labels of each level.
                label_weights: None, to be compatible with ATSS targets.
                bbox_targets (list[Tensor]): BBox targets of each level.
                bbox_weights: None, to be compatible with ATSS targets.
        """
        labels, bbox_targets = FCOSHead.get_targets(self, points,
                                                    gt_bboxes_list,
                                                    gt_labels_list)
        label_weights = None
        bbox_weights = None
        return labels, label_weights, bbox_targets, bbox_weights

    def get_atss_targets(self,
                         cls_scores,
                         mlvl_points,
                         gt_bboxes,
                         gt_labels,
                         img_metas,
                         gt_bboxes_ignore=None):
        """A wrapper for computing ATSS targets for points in multiple images.

        Args:
            cls_scores (list[Tensor]): Box iou-aware scores for each scale
                level with shape (N, num_points * num_classes, H, W).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            gt_bboxes (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4). Default: None.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level.
                label_weights (Tensor): Label weights of all levels.
                bbox_targets_list (list[Tensor]): Regression targets of each
                    level, (l, t, r, b).
                bbox_weights (Tensor): Bbox weights of all levels.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = ATSSHead_cvpr.get_targets(
            self,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            featmap_sizes,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            unmap_outputs=True)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg, br_heatmap_all, tl_heatmap_all) = cls_reg_targets

        bbox_targets_list = [
            bbox_targets.reshape(-1, 4) for bbox_targets in bbox_targets_list
        ]

        num_imgs = len(img_metas)
        # transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format
        bbox_targets_list = self.transform_bbox_targets(
            bbox_targets_list, mlvl_points, num_imgs)

        labels_list = [labels.reshape(-1) for labels in labels_list]
        label_weights_list = [
            label_weights.reshape(-1) for label_weights in label_weights_list
        ]
        bbox_weights_list = [
            bbox_weights.reshape(-1) for bbox_weights in bbox_weights_list
        ]
        label_weights = torch.cat(label_weights_list)
        bbox_weights = torch.cat(bbox_weights_list)
        return labels_list, label_weights, bbox_targets_list, bbox_weights, br_heatmap_all, tl_heatmap_all

    def transform_bbox_targets(self, decoded_bboxes, mlvl_points, num_imgs):
        """Transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format.

        Args:
            decoded_bboxes (list[Tensor]): Regression targets of each level,
                in the form of (x1, y1, x2, y2).
            mlvl_points (list[Tensor]): Points of each fpn level, each has
                shape (num_points, 2).
            num_imgs (int): the number of images in a batch.

        Returns:
            bbox_targets (list[Tensor]): Regression targets of each level in
                the form of (l, t, r, b).
        """
        # TODO: Re-implemented in Class PointCoder
        assert len(decoded_bboxes) == len(mlvl_points)
        num_levels = len(decoded_bboxes)
        mlvl_points = [points.repeat(num_imgs, 1) for points in mlvl_points]
        bbox_targets = []
        for i in range(num_levels):
            bbox_target = bbox2distance(mlvl_points[i], decoded_bboxes[i])
            bbox_targets.append(bbox_target)

        return bbox_targets

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Override the method in the parent class to avoid changing para's
        name."""
        pass

    def multiclass_nms(self, multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                    tl_heat,
                    br_heat,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
        """NMS for multi-class bboxes.

        Args:
            multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
            multi_scores (Tensor): shape (n, #class), where the last column
                contains scores of the background class, but this will be ignored.
            score_thr (float): bbox threshold, bboxes with scores lower than it
                will not be considered.
            nms_thr (float): NMS IoU threshold
            max_num (int, optional): if there are more than max_num bboxes after
                NMS, only top max_num will be kept. Default to -1.
            score_factors (Tensor, optional): The factors multiplied to scores
                before applying NMS. Default to None.
            return_inds (bool, optional): Whether return the indices of kept
                bboxes. Default to False.

        Returns:
            tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
                (k), and (k). Dets are boxes with scores. Labels are 0-based.
        """
        num_classes = multi_scores.size(1) - 1
        # exclude background category
        if multi_bboxes.shape[1] > 4:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 4)

        scores = multi_scores[:, :-1]

        labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
        labels = labels.view(1, -1).expand_as(scores)

        bboxes = bboxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        tl_heat = tl_heat.reshape(-1)
        br_heat = br_heat.reshape(-1)

        if not torch.onnx.is_in_onnx_export():
            # NonZero not supported  in TensorRT
            # remove low scoring boxes
            valid_mask = scores > score_thr
        # multiply score_factor after threshold to preserve more bboxes, improve
        # mAP by 1% for YOLOv3
        if score_factors is not None:
            # expand the shape to match original shape of score
            score_factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes)
            score_factors = score_factors.reshape(-1)
            scores = scores * score_factors

        if not torch.onnx.is_in_onnx_export():
            # NonZero not supported  in TensorRT
            inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
            bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
            br_heat, tl_heat = br_heat[inds], tl_heat[inds]
            try:
                # avg = torch.max(torch.stack([br_heat,tl_heat],dim=1),dim=1)[0]
                avg = (br_heat + tl_heat)/2
            except:
                avg = torch.tensor([]).to(br_heat)
        else:
            # TensorRT NMS plugin has invalid output filled with -1
            # add dummy data to make detection output correct.
            bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
            scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
            labels = torch.cat([labels, labels.new_zeros(1)], dim=0)


        # with open('/home/data/lyl/project/work/mmdetection_cvpr/'+'br_heat.txt', "ab") as f:
        #     np.savetxt(f, np.c_[br_heat.cpu().numpy()], fmt='%f', delimiter='\t')
        # with open('/home/data/lyl/project/work/mmdetection_cvpr/'+'tl_heat.txt', "ab") as f:
        #     np.savetxt(f, np.c_[tl_heat.cpu().numpy()], fmt='%f', delimiter='\t')
        # with open('/home/data/lyl/project/work/mmdetection_cvpr/' + 'bbox.txt', "ab") as f:
        #     np.savetxt(f, np.c_[bboxes.cpu().numpy()], fmt='%f', delimiter='\t')
        # with open('/home/data/lyl/project/work/mmdetection_cvpr/'+'scores.txt', "ab") as f:
        #     np.savetxt(f, np.c_[scores.cpu().numpy()], fmt='%f', delimiter='\t')

        if bboxes.numel() == 0:
            if torch.onnx.is_in_onnx_export():
                raise RuntimeError('[ONNX Error] Can not record NMS '
                                   'as it has not been executed this time')
            dets = torch.cat([bboxes, scores[:, None]], -1)
            if return_inds:
                return dets, labels, inds
            else:
                return dets, labels

        dets, keep = self.batched_nms(bboxes, torch.exp(avg) * scores, labels, nms_cfg, br_heat, tl_heat)#torch.exp(avg)

        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]

        if return_inds:
            return dets, labels[keep], inds[keep]
        else:
            return dets, labels[keep]

    def batched_nms(self, boxes, scores, idxs, nms_cfg, br_heat, tl_heat, class_agnostic=False):
        """Performs non-maximum suppression in a batched fashion.

        Modified from https://github.com/pytorch/vision/blob
        /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
        In order to perform NMS independently per class, we add an offset to all
        the boxes. The offset is dependent only on the class idx, and is large
        enough so that boxes from different classes do not overlap.

        Arguments:
            boxes (torch.Tensor): boxes in shape (N, 4).
            scores (torch.Tensor): scores in shape (N, ).
            idxs (torch.Tensor): each index value correspond to a bbox cluster,
                and NMS will not be applied between elements of different idxs,
                shape (N, ).
            nms_cfg (dict): specify nms type and other parameters like iou_thr.
                Possible keys includes the following.

                - iou_thr (float): IoU threshold used for NMS.
                - split_thr (float): threshold number of boxes. In some cases the
                    number of boxes is large (e.g., 200k). To avoid OOM during
                    training, the users could set `split_thr` to a small value.
                    If the number of boxes is greater than the threshold, it will
                    perform NMS on each group of boxes separately and sequentially.
                    Defaults to 10000.
            class_agnostic (bool): if true, nms is class agnostic,
                i.e. IoU thresholding happens over all boxes,
                regardless of the predicted class.

        Returns:
            tuple: kept dets and indice.
        """
        nms_cfg_ = nms_cfg.copy()
        class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
        if class_agnostic:
            boxes_for_nms = boxes
        else:
            max_coordinate = boxes.max()
            offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
            boxes_for_nms = boxes + offsets[:, None]

        nms_type = nms_cfg_.pop('type', 'nms')
        nms_op = eval(nms_type)

        split_thr = nms_cfg_.pop('split_thr', 10000)
        # Won't split to multiple nms nodes when exporting to onnx
        if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
            dets, keep = nms_op(boxes_for_nms, scores, br_heat, tl_heat, **nms_cfg_)
            boxes = boxes[keep]
            # -1 indexing works abnormal in TensorRT
            # This assumes `dets` has 5 dimensions where
            # the last dimension is score.
            # TODO: more elegant way to handle the dimension issue.
            # Some type of nms would reweight the score, such as SoftNMS
            scores = dets[:, 4]
        else:
            max_num = nms_cfg_.pop('max_num', -1)
            total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
            # Some type of nms would reweight the score, such as SoftNMS
            scores_after_nms = scores.new_zeros(scores.size())
            for id in torch.unique(idxs):
                mask = (idxs == id).nonzero(as_tuple=False).view(-1)
                dets, keep = nms_op(boxes_for_nms[mask], scores[mask], br_heat, tl_heat, **nms_cfg_)
                total_mask[mask[keep]] = True
                scores_after_nms[mask[keep]] = dets[:, -1]
            keep = total_mask.nonzero(as_tuple=False).view(-1)

            scores, inds = scores_after_nms[keep].sort(descending=True)
            keep = keep[inds]
            boxes = boxes[keep]

            if max_num > 0:
                keep = keep[:max_num]
                boxes = boxes[:max_num]
                scores = scores[:max_num]

        return torch.cat([boxes, scores[:, None]], -1), keep



    # def multiclass_nms(self,
    #                    multi_bboxes,
    #                    multi_scores,
    #                    score_thr,
    #                    nms_cfg,
    #                    tl_heat,
    #                    br_heat,
    #                    max_num=-1,
    #                    score_factors=None,
    #                    return_inds=False):
    #     """NMS for multi-class bboxes.
    #
    #     Args:
    #         multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
    #         multi_scores (Tensor): shape (n, #class), where the last column
    #             contains scores of the background class, but this will be ignored.
    #         score_thr (float): bbox threshold, bboxes with scores lower than it
    #             will not be considered.
    #         nms_thr (float): NMS IoU threshold
    #         max_num (int, optional): if there are more than max_num bboxes after
    #             NMS, only top max_num will be kept. Default to -1.
    #         score_factors (Tensor, optional): The factors multiplied to scores
    #             before applying NMS. Default to None.
    #         return_inds (bool, optional): Whether return the indices of kept
    #             bboxes. Default to False.
    #
    #     Returns:
    #         tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
    #             (k), and (k). Labels are 0-based.
    #     """
    #     num_classes = multi_scores.size(1) - 1
    #     # exclude background category
    #     if multi_bboxes.shape[1] > 4:
    #         bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    #     else:
    #         bboxes = multi_bboxes[:, None].expand(
    #             multi_scores.size(0), num_classes, 4)
    #
    #     scores = multi_scores[:, :-1]
    #
    #     labels = torch.arange(num_classes, dtype=torch.long)
    #     labels = labels.view(1, -1).expand_as(scores)
    #
    #     bboxes = bboxes.reshape(-1, 4)
    #     scores = scores.reshape(-1)
    #     labels = labels.reshape(-1)
    #     tl_heat = tl_heat.reshape(-1)
    #     br_heat = br_heat.reshape(-1)
    #     # remove low scoring boxes
    #     valid_mask = scores > score_thr
    #     # multiply score_factor after threshold to preserve more bboxes, improve
    #     # mAP by 1% for YOLOv3
    #     if score_factors is not None:
    #         # expand the shape to match original shape of score
    #         score_factors = score_factors.view(-1, 1).expand(
    #             multi_scores.size(0), num_classes)
    #         score_factors = score_factors.reshape(-1)
    #         scores = scores * score_factors
    #     inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    #     bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    #     br_heat, tl_heat = br_heat[inds], tl_heat[inds]
    #     try:
    #         #     # avg = torch.max(torch.stack([br_heat,tl_heat],dim=1),dim=1)[0]
    #         avg = (br_heat + tl_heat) / 2
    #     except:
    #         avg = torch.tensor([]).to(br_heat)
    #     if inds.numel() == 0:
    #         if torch.onnx.is_in_onnx_export():
    #             raise RuntimeError('[ONNX Error] Can not record NMS '
    #                                'as it has not been executed this time')
    #         if return_inds:
    #             return bboxes, labels, inds
    #         else:
    #             return bboxes, labels
    #
    #     # TODO: add size check before feed into batched_nms
    #     dets, keep = self.batched_nms(bboxes, torch.exp(avg) * scores, labels, nms_cfg, br_heat, tl_heat)
    #
    #     if max_num > 0:
    #         dets = dets[:max_num]
    #         keep = keep[:max_num]
    #
    #     if return_inds:
    #         return dets, keep
    #     else:
    #         return dets, keep
    #
    #     #
    #
    #
    # def batched_nms(self, boxes, scores, idxs, nms_cfg, br_heat, tl_heat, class_agnostic=False):
    #         """Performs non-maximum suppression in a batched fashion.
    #
    #         Modified from https://github.com/pytorch/vision/blob
    #         /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    #         In order to perform NMS independently per class, we add an offset to all
    #         the boxes. The offset is dependent only on the class idx, and is large
    #         enough so that boxes from different classes do not overlap.
    #
    #         Arguments:
    #             boxes (torch.Tensor): boxes in shape (N, 4).
    #             scores (torch.Tensor): scores in shape (N, ).
    #             idxs (torch.Tensor): each index value correspond to a bbox cluster,
    #                 and NMS will not be applied between elements of different idxs,
    #                 shape (N, ).
    #             nms_cfg (dict): specify nms type and other parameters like iou_thr.
    #                 Possible keys includes the following.
    #
    #                 - iou_thr (float): IoU threshold used for NMS.
    #                 - split_thr (float): threshold number of boxes. In some cases the
    #                     number of boxes is large (e.g., 200k). To avoid OOM during
    #                     training, the users could set `split_thr` to a small value.
    #                     If the number of boxes is greater than the threshold, it will
    #                     perform NMS on each group of boxes separately and sequentially.
    #                     Defaults to 10000.
    #             class_agnostic (bool): if true, nms is class agnostic,
    #                 i.e. IoU thresholding happens over all boxes,
    #                 regardless of the predicted class.
    #
    #         Returns:
    #             tuple: kept dets and indice.
    #         """
    #         nms_cfg_ = nms_cfg.copy()
    #         class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    #         if class_agnostic:
    #             boxes_for_nms = boxes
    #         else:
    #             max_coordinate = boxes.max()
    #             offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    #             boxes_for_nms = boxes + offsets[:, None]
    #
    #         nms_type = nms_cfg_.pop('type', 'nms')
    #         nms_op = eval(nms_type)
    #
    #         split_thr = nms_cfg_.pop('split_thr', 10000)
    #         # Won't split to multiple nms nodes when exporting to onnx
    #         if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
    #             dets, keep = self.nms(boxes_for_nms, scores.unsqueeze(1), idxs, br_heat, tl_heat,
    #                                   nms_cfg_['iou_threshold'])
    #             # a = torch.tensor(keep).to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    #             # b = dets[:,:4]
    #             try:
    #                 dets[:, :4] = dets[:, :4] - (torch.tensor(keep).to(boxes) * (
    #                             max_coordinate + torch.tensor(1).to(boxes)))[:, None]
    #                 boxes = dets
    #                 scores = dets[:, -1]
    #             except:
    #                 return torch.tensor([]).to(boxes), torch.tensor([]).to(boxes)
    #         else:
    #             # total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    #             # for id in torch.unique(idxs):
    #             #     mask = (idxs == id).nonzero(as_tuple=False).view(-1)
    #             #     # dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
    #             #     dets, keep = self.nms(boxes_for_nms[mask], scores[mask].unsqueeze(1), idxs[mask], br_heat[mask],
    #             #                           tl_heat[mask], nms_cfg_['iou_threshold'])
    #             #     total_mask[mask[keep]] = True
    #             #
    #             # keep = total_mask.nonzero(as_tuple=False).view(-1)
    #             # keep = keep[scores[keep].argsort(descending=True)]
    #             # boxes = dets
    #             # scores = scores[keep]
    #
    #             dets, keep = self.nms(boxes_for_nms, scores.unsqueeze(1), idxs, br_heat, tl_heat,
    #                                   nms_cfg_['iou_threshold'])
    #             # a = torch.tensor(keep).to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    #             # b = dets[:,:4]
    #             try:
    #                 dets[:, :4] = dets[:, :4] - (torch.tensor(keep).to(boxes) * (
    #                             max_coordinate + torch.tensor(1).to(boxes)))[:, None]
    #                 boxes = dets
    #                 scores = dets[:, -1]
    #             except:
    #                 return torch.tensor([]).to(boxes), torch.tensor([]).to(boxes)
    #
    #         return boxes, torch.tensor(keep).to(boxes)
    #
    # def nms(self, boxes, scores, idxs, br_heat, tl_heat, overlap=0.5, top_k=200):
    #         """
    #         输入:
    #             boxes: 存储一个图片的所有预测框。[num_positive,4].
    #             scores:置信度。如果为多分类则需要将nms函数套在一个循环内。[num_positive].
    #             overlap: nms抑制时iou的阈值.
    #             top_k: 先选取置信度前top_k个框再进行nms.
    #         返回:
    #             nms后剩余预测框的索引.
    #         """
    #         det = []
    #         cls = []
    #         for cls_id in range(self.num_classes):
    #             mask = idxs == cls_id
    #             score = scores[mask]
    #             box = boxes[mask]
    #             tl_heat_ = tl_heat[mask]
    #             br_heat_ = br_heat[mask]
    #             keep = score.new(score.size(0)).zero_().long()
    #             # 保存留下来的box的索引 [num_positive]
    #             # 函数new(): 构建一个有相同数据类型的tensor
    #
    #             # 如果输入box为空则返回空Tensor
    #             if box.numel() == 0:
    #                 continue
    #
    #             x1 = box[:, 0]  # x1 坐标
    #             y1 = box[:, 1]
    #             x2 = box[:, 2]
    #             y2 = box[:, 3]
    #             area = torch.mul(x2 - x1, y2 - y1)  # 并行化计算所有框的面积
    #             v, idx = score.sort(0)  # 升序排序
    #             idx = idx.squeeze(1)
    #             # idx = idx[-top_k:].squeeze(1)  # 前top-k的索引，从小到大
    #             xx1 = box.new()
    #             yy1 = box.new()
    #             xx2 = box.new()  # new() 无参数，创建 相同类型的空值；
    #             yy2 = box.new()
    #             w = box.new()
    #             h = box.new()
    #
    #             count = 0
    #             while idx.numel() > 0:
    #                 i = idx[-1]
    #                 cla_score = score[i]  # 目前最大score对应的索引  # 选取得分最大的框索引；
    #                 keep[count] = i  # 存储在keep中
    #                 count += 1
    #                 if idx.size(0) == 1:  # 跳出循环条件：box被筛选完了
    #                     break
    #                 idx = idx[:-1]  # 去掉最后一个
    #
    #                 # 剩下boxes的信息存储在xx，yy中
    #                 torch.index_select(x1, 0, idx, out=xx1)  # 从x1中再维度0选取索引为idx 数据 输出到xx1中；
    #                 torch.index_select(y1, 0, idx, out=yy1)  # torch.index_select（） # 从tensor中按指定维度和索引 取值；
    #                 torch.index_select(x2, 0, idx, out=xx2)
    #                 torch.index_select(y2, 0, idx, out=yy2)
    #
    #                 # 计算当前最大置信框与其他剩余框的交集，不知道clamp的同学确实容易被误导
    #                 xx1 = torch.clamp(xx1, min=x1[i])  # max(x1,xx1)  # x1 y1 的最大值
    #                 yy1 = torch.clamp(yy1, min=y1[i])  # max(y1,yy1)
    #                 xx2 = torch.clamp(xx2, max=x2[i])  # min(x2,xx2)  # x2 x3 最小值；
    #                 yy2 = torch.clamp(yy2, max=y2[i])  # min(y2,yy2)
    #                 w.resize_as_(xx2)
    #                 h.resize_as_(yy2)
    #                 w = xx2 - xx1  # w=min(x2,xx2)−max(x1,xx1)
    #                 h = yy2 - yy1  # h=min(y2,yy2)−max(y1,yy1)
    #                 w = torch.clamp(w, min=0.0)  # max(w,0)
    #                 h = torch.clamp(h, min=0.0)  # max(h,0)
    #                 inter = w * h
    #
    #                 # 计算当前最大置信框与其他剩余框的IOU
    #                 # IoU = i / (area(a) + area(b) - i)
    #                 rem_areas = torch.index_select(area, 0, idx)  # 剩余的框的面积
    #                 union = rem_areas + area[i] - inter  # 并集
    #                 IoU = inter / union  # 计算iou
    #
    #                 try:
    #                     # mask1 = IoU.gt(overlap) & (score[idx]>(cla_score)).squeeze(1)
    #                     # mask1 = IoU.gt(overlap)
    #                     # mask1[-10:-1]=False
    #                     # mask1 = (score[idx]>=(score[idx].max()-torch.std(score[idx][mask1]))).squeeze(1)
    #                     # tl_x1 = x1[idx][mask1][torch.argmax(tl_heat_[idx][mask1])].unsqueeze(0)
    #                     # tl_y1 = y1[idx][mask1][torch.argmax(tl_heat_[idx][mask1])].unsqueeze(0)
    #                     # br_x2 = x2[idx][mask1][torch.argmax(br_heat_[idx][mask1])].unsqueeze(0)
    #                     # br_y2 = y2[idx][mask1][torch.argmax(br_heat_[idx][mask1])].unsqueeze(0)
    #                     # if cla_score > 0.5 :
    #                     #
    #                     #     a = torch.cat([x1[idx][mask1],x1[i].unsqueeze(0)],dim=0)
    #                     #     with open('/home/ubuntu/lyl/project/cvpr/cvpr/'+'x1.txt', "ab") as f:
    #                     #         np.savetxt(f, np.c_[a.cpu().numpy()], fmt='%f', delimiter='\t')
    #                     #
    #                     #     b = torch.cat([y1[idx][mask1],y1[i].unsqueeze(0)],dim=0)
    #                     #     with open('/home/ubuntu/lyl/project/cvpr/cvpr/'+'y1.txt', "ab") as f:
    #                     #         np.savetxt(f, np.c_[b.cpu().numpy()], fmt='%f', delimiter='\t')
    #                     #
    #                     #     c = torch.cat([x2[idx][mask1],x2[i].unsqueeze(0)],dim=0)
    #                     #     with open('/home/ubuntu/lyl/project/cvpr/cvpr/'+'x2.txt', "ab") as f:
    #                     #         np.savetxt(f, np.c_[c.cpu().numpy()], fmt='%f', delimiter='\t')
    #                     #
    #                     #     d = torch.cat([y2[idx][mask1],y2[i].unsqueeze(0)],dim=0)
    #                     #     with open('/home/ubuntu/lyl/project/cvpr/cvpr/'+'y2.txt', "ab") as f:
    #                     #         np.savetxt(f, np.c_[d.cpu().numpy()], fmt='%f', delimiter='\t')
    #                     #
    #                     #     aa = torch.cat([tl_heat_[idx][mask1],tl_heat_[i].unsqueeze(0)],dim=0)
    #                     #     with open('/home/ubuntu/lyl/project/cvpr/cvpr/'+'tl.txt', "ab") as f:
    #                     #         np.savetxt(f, np.c_[aa.cpu().numpy()], fmt='%f', delimiter='\t')
    #                     #
    #                     #     cc = torch.cat([br_heat_[idx][mask1],br_heat_[i].unsqueeze(0)],dim=0)
    #                     #     with open('/home/ubuntu/lyl/project/cvpr/cvpr/'+'br.txt', "ab") as f:
    #                     #         np.savetxt(f, np.c_[cc.cpu().numpy()], fmt='%f', delimiter='\t')
    #                     # mask1 = IoU.gt(overlap) & (tl_heat_[idx] >= tl_heat_[i])
    #                     # mask2 = IoU.gt(overlap) & (br_heat_[idx] >= br_heat_[i])
    #                     # a = (tl_heat_[idx]>=torch.topk(tl_heat_[idx], k=5)[0][-1])
    #                     # d = (tl_heat_[idx]>=torch.topk(tl_heat_[idx], k=5)[0])
    #                     # b = tl_heat_[idx]
    #                     # c = (IoU.gt(overlap) & (tl_heat_[idx] >= tl_heat_[i]))
    #                     # if ((IoU.gt(overlap) & (score[idx]>=(cla_score-0.2)).squeeze(1)) & (tl_heat_[idx] >= tl_heat_[i])).sum()>50:
    #                     #     mask1 = IoU.gt(overlap) & (score[idx]>=(cla_score-0.2)).squeeze(1) & (tl_heat_[idx]>=torch.topk(tl_heat_[idx], k=5)[0][-1])
    #                     #
    #                     #     tl_x1 = x1[idx][mask1].mean().unsqueeze(0)
    #                     #     tl_y1 = y1[idx][mask1].mean().unsqueeze(0)
    #                     #
    #                     # else:
    #                     #     mask1 = (IoU.gt(overlap) & (score[idx]>=(cla_score-0.2)).squeeze(1)) & (tl_heat_[idx] >= tl_heat_[i])
    #                     #     tl_x1 = torch.cat([x1[idx][mask1],x1[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
    #                     #     tl_y1 = torch.cat([y1[idx][mask1],y1[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
    #                     #
    #                     # if ((IoU.gt(overlap) & (score[idx]>=(cla_score-0.2)).squeeze(1)) & (br_heat_[idx] >= br_heat_[i])).sum() > 50:
    #                     #     # a = br_heat_[idx] >= torch.topk(br_heat_[idx], k=5)
    #                     #     mask2 = IoU.gt(overlap) & (score[idx]>=(cla_score-0.2)).squeeze(1) & (br_heat_[idx] >= torch.topk(br_heat_[idx], k=5)[0][-1])
    #                     #     br_x2 = x2[idx][mask2].mean().unsqueeze(0)
    #                     #     br_y2 = y2[idx][mask2].mean().unsqueeze(0)
    #                     # else:
    #                     #     mask2 = IoU.gt(overlap) & (score[idx]>=(cla_score-0.2)).squeeze(1) & (br_heat_[idx] >= br_heat_[i])
    #                     #     # mask1 = IoU.gt(overlap) & (tl_heat_[idx] >= tl_heat_[idx].mean() - torch.std(tl_heat_[idx], dim=0))
    #                     #     # mask2 = IoU.gt(overlap) & (br_heat_[idx] >= br_heat_[idx].mean() - torch.std(br_heat_[idx], dim=0))
    #                     #     br_x2 = torch.cat([x2[idx][mask2], x2[i].unsqueeze(0)], dim=0).mean().unsqueeze(0)
    #                     #     br_y2 = torch.cat([y2[idx][mask2], y2[i].unsqueeze(0)], dim=0).mean().unsqueeze(0)
    #
    #                     # mask1 = IoU.gt(overlap) & (tl_heat_[idx] >= (tl_heat_[idx].mean() + tl_heat_[
    #                     #     idx].std()))  # & (score[idx]>=(score[idx].mean()+score[idx].std())).squeeze(1)
    #                     # mask2 = IoU.gt(overlap) & (br_heat_[idx] >= (br_heat_[idx].mean() + br_heat_[
    #                     #     idx].std()))  # & (score[idx]>=(score[idx].mean()+score[idx].std())).squeeze(1)
    #                     # tl_x1 = torch.cat([x1[idx][mask1], x1[i].unsqueeze(0)], dim=0).mean().unsqueeze(0)
    #                     # tl_y1 = torch.cat([y1[idx][mask1], y1[i].unsqueeze(0)], dim=0).mean().unsqueeze(0)
    #                     # br_x2 = torch.cat([x2[idx][mask2], x2[i].unsqueeze(0)], dim=0).mean().unsqueeze(0)
    #                     # br_y2 = torch.cat([y2[idx][mask2], y2[i].unsqueeze(0)], dim=0).mean().unsqueeze(0)
    #
    #                     mask1 = IoU.gt(overlap)  & (tl_heat_[idx] >= tl_heat_[i])& (score[idx]>=(score[idx].mean()+score[idx].std())).squeeze(1)
    #                     mask2 = IoU.gt(overlap)  & (br_heat_[idx] >= br_heat_[i])& (score[idx]>=(score[idx].mean()+score[idx].std())).squeeze(1)
    #                     # a = x1[idx][mask1]
    #                     tl_x1 = torch.cat([x1[idx][mask1],x1[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
    #                     tl_y1 = torch.cat([y1[idx][mask1],y1[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
    #                     br_x2 = torch.cat([x2[idx][mask2],x2[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
    #                     br_y2 = torch.cat([y2[idx][mask2],y2[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
    #
    #                     # a = x1[i]
    #                     # b = y1[i]
    #                     # c = x2[i]
    #                     # d = y2[i]
    #
    #                     # tl_x1 = torch.cat([x1[idx][mask1],x1[i].unsqueeze(0)],dim=0)[torch.argmax(torch.cat([tl_heat_[idx][mask1],tl_heat_[i].unsqueeze(0)],dim=0))].unsqueeze(0)
    #                     # tl_y1 = torch.cat([y1[idx][mask1],y1[i].unsqueeze(0)],dim=0)[torch.argmax(torch.cat([tl_heat_[idx][mask1],tl_heat_[i].unsqueeze(0)],dim=0))].unsqueeze(0)
    #                     # br_x2 = torch.cat([x2[idx][mask1],x2[i].unsqueeze(0)],dim=0)[torch.argmax(torch.cat([br_heat_[idx][mask1],br_heat_[i].unsqueeze(0)],dim=0))].unsqueeze(0)
    #                     # br_y2 = torch.cat([y2[idx][mask1],y2[i].unsqueeze(0)],dim=0)[torch.argmax(torch.cat([br_heat_[idx][mask1],br_heat_[i].unsqueeze(0)],dim=0))].unsqueeze(0)
    #                     # tl_x1 = torch.cat([x1[idx][mask1],x1[i].unsqueeze(0)],dim=0)[torch.topk(torch.cat([tl_heat_[idx][mask1],tl_heat_[i].unsqueeze(0)],dim=0),k=3)[1]].mean().unsqueeze(0)
    #                     # tl_y1 = torch.cat([y1[idx][mask1],y1[i].unsqueeze(0)],dim=0)[torch.topk(torch.cat([tl_heat_[idx][mask1],tl_heat_[i].unsqueeze(0)],dim=0),k=3)[1]].mean().unsqueeze(0)
    #                     # br_x2 = torch.cat([x2[idx][mask1],x2[i].unsqueeze(0)],dim=0)[torch.topk(torch.cat([br_heat_[idx][mask1],br_heat_[i].unsqueeze(0)],dim=0),k=3)[1]].mean().unsqueeze(0)
    #                     # br_y2 = torch.cat([y2[idx][mask1],y2[i].unsqueeze(0)],dim=0)[torch.topk(torch.cat([br_heat_[idx][mask1],br_heat_[i].unsqueeze(0)],dim=0),k=3)[1]].mean().unsqueeze(0)
    #                     # a = boxes[i]
    #                     # b = torch.stack([tl_x1,tl_y1,br_x2,br_y2,cla_score],dim=1)
    #                     # det.append(torch.stack([tl_x1, tl_y1, br_x2, br_y2, cla_score], dim=1))
    #                     det.append(torch.stack([x1[i].unsqueeze(0), y1[i].unsqueeze(0), x2[i].unsqueeze(0), y2[i].unsqueeze(0), cla_score], dim=1))
    #                     cls.append(cls_id)
    #                 except:
    #                     continue
    #
    #                 # 选出IoU <= overlap的boxes(注意le函数的使用)
    #                 idx = idx[IoU.le(overlap)]  # le: 小于等于 返回的bool ， 去除大于overlap的值；
    #         if len(det) == 0:
    #             return torch.tensor([]), []
    #         det = torch.cat(det, dim=0)
    #         return det, cls

# def nms(boxes, scores, overlap=0.7, top_k=200):
#     """
#     输入:
#         boxes: 存储一个图片的所有预测框。[num_positive,4].
#         scores:置信度。如果为多分类则需要将nms函数套在一个循环内。[num_positive].
#         overlap: nms抑制时iou的阈值.
#         top_k: 先选取置信度前top_k个框再进行nms.
#     返回:
#         nms后剩余预测框的索引.
#     """
#
#     keep = scores.new(scores.size(0)).zero_().long()
#     # 保存留下来的box的索引 [num_positive]
#     # 函数new(): 构建一个有相同数据类型的tensor
#
#     # 如果输入box为空则返回空Tensor
#     if boxes.numel() == 0:
#         return keep
#
#     x1 = boxes[:, 0]  # x1 坐标
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#     area = torch.mul(x2 - x1, y2 - y1)  # 并行化计算所有框的面积
#     v, idx = scores.sort(0)  # 升序排序
#     idx = idx[-top_k:]  # 前top-k的索引，从小到大
#     xx1 = boxes.new()
#     yy1 = boxes.new()
#     xx2 = boxes.new()  # new() 无参数，创建 相同类型的空值；
#     yy2 = boxes.new()
#     w = boxes.new()
#     h = boxes.new()
#
#     count = 0
#     while idx.numel() > 0:
#         i = idx[-1]  # 目前最大score对应的索引  # 选取得分最大的框索引；
#         keep[count] = i  # 存储在keep中
#         count += 1
#         if idx.size(0) == 1:  # 跳出循环条件：box被筛选完了
#             break
#         idx = idx[:-1]  # 去掉最后一个
#
#         # 剩下boxes的信息存储在xx，yy中
#         torch.index_select(x1, 0, idx, out=xx1)  # 从x1中再维度0选取索引为idx 数据 输出到xx1中；
#         torch.index_select(y1, 0, idx, out=yy1)  # torch.index_select（） # 从tensor中按指定维度和索引 取值；
#         torch.index_select(x2, 0, idx, out=xx2)
#         torch.index_select(y2, 0, idx, out=yy2)
#
#         # 计算当前最大置信框与其他剩余框的交集，不知道clamp的同学确实容易被误导
#         xx1 = torch.clamp(xx1, min=x1[i])  # max(x1,xx1)  # x1 y1 的最大值
#         yy1 = torch.clamp(yy1, min=y1[i])  # max(y1,yy1)
#         xx2 = torch.clamp(xx2, max=x2[i])  # min(x2,xx2)  # x2 x3 最小值；
#         yy2 = torch.clamp(yy2, max=y2[i])  # min(y2,yy2)
#         w.resize_as_(xx2)
#         h.resize_as_(yy2)
#         w = xx2 - xx1  # w=min(x2,xx2)−max(x1,xx1)
#         h = yy2 - yy1  # h=min(y2,yy2)−max(y1,yy1)
#         w = torch.clamp(w, min=0.0)  # max(w,0)
#         h = torch.clamp(h, min=0.0)  # max(h,0)
#         inter = w * h
#
#         # 计算当前最大置信框与其他剩余框的IOU
#         # IoU = i / (area(a) + area(b) - i)
#         rem_areas = torch.index_select(area, 0, idx)  # 剩余的框的面积
#         union = rem_areas + area[i] - inter  # 并集
#         IoU = inter / union  # 计算iou
#
#         # 选出IoU <= overlap的boxes(注意le函数的使用)
#         idx = idx[IoU.le(overlap)]  # le: 小于等于 返回的bool ， 去除大于overlap的值；
#     return keep, count

def nms(boxes, scores, br_heat, tl_heat, iou_threshold=0.5, top_k=200):
        """
        输入:
            boxes: 存储一个图片的所有预测框。[num_positive,4].
            scores:置信度。如果为多分类则需要将nms函数套在一个循环内。[num_positive].
            overlap: nms抑制时iou的阈值.
            top_k: 先选取置信度前top_k个框再进行nms.
        返回:
            nms后剩余预测框的索引.
        """
        det = []
        keep = []
        # keep = scores.new(scores.size(0)).zero_().long()
        # 保存留下来的box的索引 [num_positive]
        # 函数new(): 构建一个有相同数据类型的tensor

        # 如果输入box为空则返回空Tensor
        if boxes.numel() == 0:
            return torch.tensor(keep).to(boxes)

        x1 = boxes[:, 0]  # x1 坐标
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)  # 并行化计算所有框的面积
        v, idx = scores.sort(0)  # 升序排序
        # idx = idx[-top_k:]  # 前top-k的索引，从小到大
        xx1 = boxes.new()
        yy1 = boxes.new()
        xx2 = boxes.new()  # new() 无参数，创建 相同类型的空值；
        yy2 = boxes.new()
        w = boxes.new()
        h = boxes.new()

        count = 0
        while idx.numel() > 0:
            i = idx[-1]
            cla_score = scores[i]# 目前最大score对应的索引  # 选取得分最大的框索引；
            # keep[count] = i  # 存储在keep中
            keep.append(i)
            count += 1
            if idx.size(0) == 1:  # 跳出循环条件：box被筛选完了
                # det.append(torch.stack(
                #     [x1[i], y1[i], x2[i], y2[i], cla_score], dim=0))
                break
            idx = idx[:-1]  # 去掉最后一个

            # 剩下boxes的信息存储在xx，yy中
            torch.index_select(x1, 0, idx, out=xx1)  # 从x1中再维度0选取索引为idx 数据 输出到xx1中；
            torch.index_select(y1, 0, idx, out=yy1)  # torch.index_select（） # 从tensor中按指定维度和索引 取值；
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)

            # 计算当前最大置信框与其他剩余框的交集，不知道clamp的同学确实容易被误导
            xx1 = torch.clamp(xx1, min=x1[i])  # max(x1,xx1)  # x1 y1 的最大值
            yy1 = torch.clamp(yy1, min=y1[i])  # max(y1,yy1)
            xx2 = torch.clamp(xx2, max=x2[i])  # min(x2,xx2)  # x2 x3 最小值；
            yy2 = torch.clamp(yy2, max=y2[i])  # min(y2,yy2)
            w.resize_as_(xx2)
            h.resize_as_(yy2)
            w = xx2 - xx1  # w=min(x2,xx2)−max(x1,xx1)
            h = yy2 - yy1  # h=min(y2,yy2)−max(y1,yy1)
            w = torch.clamp(w, min=0.0)  # max(w,0)
            h = torch.clamp(h, min=0.0)  # max(h,0)
            inter = w * h

            # 计算当前最大置信框与其他剩余框的IOU
            # IoU = i / (area(a) + area(b) - i)
            rem_areas = torch.index_select(area, 0, idx)  # 剩余的框的面积
            union = rem_areas + area[i] - inter  # 并集
            IoU = inter / union  # 计算iou

            # mask1 = IoU.gt(iou_threshold) & (tl_heat[idx] >= (tl_heat[idx].mean() + tl_heat[idx].std()))
            # mask2 = IoU.gt(iou_threshold) & (br_heat[idx] >= (br_heat[idx].mean() + br_heat[idx].std()))  # & (score[idx]>=(score[idx].mean()+score[idx].std())).squeeze(1)
            # tl_x1 = torch.cat([x1[idx][mask1], x1[i].unsqueeze(0)], dim=0).mean().unsqueeze(0)
            # tl_y1 = torch.cat([y1[idx][mask1], y1[i].unsqueeze(0)], dim=0).mean().unsqueeze(0)
            # br_x2 = torch.cat([x2[idx][mask2], x2[i].unsqueeze(0)], dim=0).mean().unsqueeze(0)
            # br_y2 = torch.cat([y2[idx][mask2], y2[i].unsqueeze(0)], dim=0).mean().unsqueeze(0)

            # mask1 = IoU.gt(iou_threshold)  & (tl_heat[idx] >= tl_heat[i]) & (scores[idx]>=(scores[idx].mean()+scores[idx].std()))
            # mask2 = IoU.gt(iou_threshold)  & (br_heat[idx] >= br_heat[i]) & (scores[idx]>=(scores[idx].mean()+scores[idx].std()))
            # a = x1[idx][mask1]
            # tl_x1 = torch.cat([x1[idx][mask1],x1[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
            # tl_y1 = torch.cat([y1[idx][mask1],y1[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
            # br_x2 = torch.cat([x2[idx][mask2],x2[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
            # br_y2 = torch.cat([y2[idx][mask2],y2[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
            #
            #
            # det.append(torch.cat([tl_x1, tl_y1, br_x2, br_y2, cla_score.unsqueeze(0)], dim=0))

            # 选出IoU <= overlap的boxes(注意le函数的使用)
            idx = idx[IoU.le(iou_threshold)]  # le: 小于等于 返回的bool ， 去除大于overlap的值；
        keep = torch.stack(keep, dim=0).to(boxes).long()
        det = torch.cat([boxes, scores.unsqueeze(1)], dim=1)[keep]

        # det = torch.stack(det, dim=0).to(boxes)
        return det, keep
