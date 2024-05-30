import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from ..builder import HEADS
from .anchor_head import AnchorHead
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS, build_loss
from mmcv.ops import CornerPool, batched_nms
import torch
from mmcv.runner import force_fp32
from ..utils import gaussian_radius, gen_gaussian_target
from math import ceil
# from mmcv.utils import deprecated_api_warning, ext_loader
import sys
import numpy as np
from torch.nn import functional as F

# ext_module = ext_loader.load_ext(
#     '_ext', ['nms', 'softnms', 'nms_match', 'nms_rotated'])

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
class RetinaHead_aid(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.num_feat_levels = 5
        super(RetinaHead_aid, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            **kwargs)
        self.loss_heatmap = build_loss(
            loss_heatmap)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)

        self.conv1 = ConvModule(
            5*self.feat_channels, self.feat_channels, 1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=None)

        self._init_corner_kpt_layers()


    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

        bias_init = bias_init_with_prob(0.1)
        self.tl_heat[-1].conv.reset_parameters()
        self.tl_heat[-1].conv.bias.data.fill_(bias_init)
        self.br_heat[-1].conv.reset_parameters()
        self.br_heat[-1].conv.bias.data.fill_(bias_init)

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
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        w, h = feats[0].size(2), feats[0].size(3)
        img_feat = []
        for feat in feats:
            img_feat.append(F.interpolate(feat, size=(w, h), mode='bilinear'))
        img_feat = torch.cat(img_feat, dim=1)

        img_feat = self.conv1(img_feat)
        tl_heat = self.tl_pool(img_feat)
        tl_heat = self.tl_heat(tl_heat)
        br_heat = self.br_pool(img_feat)
        br_heat = self.br_heat(br_heat)
        return multi_apply(self.forward_single, feats)+([tl_heat for i in range(len(feats))], [br_heat for j in range(len(feats))])

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        return cls_score, bbox_pred

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    featmap_sizes,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        featmap_sizes_list = [featmap_sizes for i in range(len(concat_anchor_list))]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            featmap_sizes_list,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list, br_heatmap_all, tl_heatmap_all) = results[:9]
        rest_results = list(results[9:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg, br_heatmap_all, tl_heatmap_all)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)


    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            featmap_size_list,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        tl_heatmap_all = []
        br_heatmap_all = []
        for featmap_size in featmap_size_list:
            height, width = featmap_size
            img_h, img_w = img_meta['img_shape'][:2]

            width_ratio = float(width / img_w)
            height_ratio = float(height / img_h)

            gt_tl_heatmap = gt_bboxes[-1].new_zeros(
                [self.num_classes, height, width])
            gt_br_heatmap = gt_bboxes[-1].new_zeros(
                [self.num_classes, height, width])

            for box_id in range(len(gt_labels)):
                left, top, right, bottom = gt_bboxes[box_id]
                label = gt_labels[box_id]

                # Use coords in the feature level to generate ground truth
                scale_left = left * width_ratio
                scale_right = right * width_ratio
                scale_top = top * height_ratio
                scale_bottom = bottom * height_ratio

                # Int coords on feature map/ground truth tensor
                left_idx = int(min(scale_left, width - 1))
                right_idx = int(min(scale_right, width - 1))
                top_idx = int(min(scale_top, height - 1))
                bottom_idx = int(min(scale_bottom, height - 1))

                # Generate gaussian heatmap
                scale_box_width = ceil(scale_right - scale_left)
                scale_box_height = ceil(scale_bottom - scale_top)
                radius = gaussian_radius((scale_box_height, scale_box_width),
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                gt_tl_heatmap[label] = gen_gaussian_target(
                    gt_tl_heatmap[label], [left_idx, top_idx],
                    radius)
                gt_br_heatmap[label] = gen_gaussian_target(
                    gt_br_heatmap[label], [right_idx, bottom_idx],
                    radius)
            br_heatmap_all.append(gt_br_heatmap)
            tl_heatmap_all.append(gt_tl_heatmap)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, br_heatmap_all, tl_heatmap_all)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             tl_heat,
             br_heat,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            featmap_sizes,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, gt_br_heatmap, gt_tl_heatmap) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        new_gt_br_heatmap = []
        new_gt_tl_heatmap = []
        for br_heatmap in zip(gt_br_heatmap[0],gt_br_heatmap[1]):
            br_heatmap_temp = list(br_heatmap)
            br_heatmap_temp = torch.stack(br_heatmap_temp,dim=0)
            new_gt_br_heatmap.append(br_heatmap_temp)

        for tl_heatmap in zip(gt_tl_heatmap[0],gt_tl_heatmap[1]):
            tl_heatmap_temp = list(tl_heatmap)
            tl_heatmap_temp = torch.stack(tl_heatmap_temp,dim=0)
            new_gt_tl_heatmap.append(tl_heatmap_temp)

        losses_cls, losses_bbox, det_loss = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            tl_heat,
            br_heat,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            new_gt_br_heatmap,
            new_gt_tl_heatmap,
            num_total_samples=num_total_samples)

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_heat= det_loss)

    def loss_single(self, cls_score, bbox_pred, tl_hmp, br_hmp, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, gt_br_hmp, gt_tl_hmp, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)

        # Detection loss
        if tl_hmp.size(2) == gt_tl_hmp.size(2):
            tl_det_loss = self.loss_heatmap(
                tl_hmp.sigmoid(),
                gt_tl_hmp,
                avg_factor=max(1,
                               gt_tl_hmp.eq(1).sum()))
            br_det_loss = self.loss_heatmap(
                br_hmp.sigmoid(),
                gt_br_hmp,
                avg_factor=max(1,
                               gt_br_hmp.eq(1).sum()))
        else:
            tl_det_loss = torch.tensor(0).to(loss_bbox)
            br_det_loss = torch.tensor(0).to(loss_bbox)
        det_loss = (tl_det_loss + br_det_loss) / 2.0

        return loss_cls, loss_bbox, 0.1 * det_loss

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   tl_heat,
                   br_heat,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class labelof the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        #
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            tl_heat_list = [
                tl_heat[i][img_id].detach() for i in range(num_levels)
            ]
            br_heat_list = [
                br_heat[i][img_id].detach() for i in range(num_levels)
            ]

        img_shape = img_metas[img_id]['img_shape']
        scale_factor = img_metas[img_id]['scale_factor']
        if with_nms:
            # some heads don't support with_nms argument
            proposals = self._get_bboxes_single(cls_score_list,
                                                bbox_pred_list,
                                                tl_heat_list,
                                                br_heat_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale)
        else:
            proposals = self._get_bboxes_single(cls_score_list,
                                                bbox_pred_list,
                                                tl_heat_list,
                                                br_heat_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale,
                                                with_nms)
        result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           tl_heat_list,
                           br_heat_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_tl_heat = []
        mlvl_br_heat = []
        for cls_score, bbox_pred, anchors  in zip(cls_score_list, bbox_pred_list,
                                                                              mlvl_anchors):
            tl_heat = tl_heat_list[0]
            br_heat = br_heat_list[0]
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)

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
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_tl_heat = torch.cat(mlvl_tl_heat).sigmoid()
        mlvl_br_heat = torch.cat(mlvl_br_heat).sigmoid()
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = self.multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    mlvl_tl_heat, mlvl_br_heat,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

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
                avg = (br_heat + tl_heat) / 2
            except:
                avg = torch.tensor([]).to(br_heat)
        else:
            # TensorRT NMS plugin has invalid output filled with -1
            # add dummy data to make detection output correct.
            bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
            scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
            labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

        if bboxes.numel() == 0:
            if torch.onnx.is_in_onnx_export():
                raise RuntimeError('[ONNX Error] Can not record NMS '
                                   'as it has not been executed this time')
            dets = torch.cat([bboxes, scores[:, None]], -1)
            if return_inds:
                return dets, labels, inds
            else:
                return dets, labels

        dets, keep = self.batched_nms(bboxes, torch.exp(avg) * scores, labels, nms_cfg, br_heat, tl_heat)

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

def nms(boxes, scores, br_heat, tl_heat, iou_threshold=0.5, top_k=200):
        det = []
        keep = []
        if boxes.numel() == 0:
            return torch.tensor(keep).to(boxes)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = torch.mul(x2 - x1, y2 - y1)
        v, idx = scores.sort(0)
        xx1 = boxes.new()
        yy1 = boxes.new()
        xx2 = boxes.new()
        yy2 = boxes.new()
        w = boxes.new()
        h = boxes.new()

        count = 0
        while idx.numel() > 0:
            i = idx[-1]
            cls_score = scores[i]
            keep.append(i)
            count += 1
            if idx.size(0) == 1:
                det.append(torch.stack(
                    [x1[i], y1[i], x2[i], y2[i], cls_score], dim=0))
                break
            idx = idx[:-1]
            torch.index_select(x1, 0, idx, out=xx1)
            torch.index_select(y1, 0, idx, out=yy1)
            torch.index_select(x2, 0, idx, out=xx2)
            torch.index_select(y2, 0, idx, out=yy2)

            xx1 = torch.clamp(xx1, min=x1[i])
            yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i])
            yy2 = torch.clamp(yy2, max=y2[i])
            w.resize_as_(xx2)
            h.resize_as_(yy2)
            w = xx2 - xx1
            h = yy2 - yy1
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w * h
            rem_areas = torch.index_select(area, 0, idx)
            union = rem_areas + area[i] - inter
            IoU = inter / union


            mask1 = IoU.gt(iou_threshold)  & (tl_heat[idx] >= tl_heat[i])
            mask2 = IoU.gt(iou_threshold)  & (br_heat[idx] >= br_heat[i])
            tl_x1 = torch.cat([x1[idx][mask1],x1[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
            tl_y1 = torch.cat([y1[idx][mask1],y1[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
            br_x2 = torch.cat([x2[idx][mask2],x2[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)
            br_y2 = torch.cat([y2[idx][mask2],y2[i].unsqueeze(0)],dim=0).mean().unsqueeze(0)

            det.append(torch.cat([tl_x1, tl_y1, br_x2, br_y2, cls_score.unsqueeze(0)], dim=0))
            idx = idx[IoU.le(iou_threshold)]  # le: 小于等于 返回的bool ， 去除大于overlap的值；
        keep = torch.stack(keep, dim=0).to(boxes).long()
        det = torch.stack(det, dim=0).to(boxes)
        return det, keep
