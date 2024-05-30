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
from mmcv.utils import deprecated_api_warning, ext_loader
import sys
import numpy as np

ext_module = ext_loader.load_ext(
    '_ext', ['nms', 'softnms', 'nms_match', 'nms_rotated'])

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
                 feat_channels=128,
                 out_channels=128,
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

        # # relu = self.relu(aftpool_conv)
        # conv2 = self.conv2(aftpool_conv)
        return conv2

@HEADS.register_module()
class RetinaHead(AnchorHead):
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
        super(RetinaHead, self).__init__(
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
        for i in range(self.num_feat_levels):
            # The initialization of parameters are different between nn.Conv2d
            # and ConvModule. Our experiments show that using the original
            # initialization of nn.Conv2d increases the final mAP by about 0.2%
            self.tl_heat[i][-1].conv.reset_parameters()
            self.tl_heat[i][-1].conv.bias.data.fill_(bias_init)
            self.br_heat[i][-1].conv.reset_parameters()
            self.br_heat[i][-1].conv.bias.data.fill_(bias_init)


    def _init_corner_kpt_layers(self):
        """Initialize corner keypoint layers.

        Including corner heatmap branch and corner offset branch. Each branch
        has two parts: prefix `tl_` for top-left and `br_` for bottom-right.
        """
        self.tl_pool, self.br_pool = nn.ModuleList(), nn.ModuleList()
        self.tl_heat, self.br_heat = nn.ModuleList(), nn.ModuleList()

        for _ in range(self.num_feat_levels):
            self.tl_pool.append(
                BiCornerPool(
                    self.in_channels, ['top', 'left'],
                    out_channels=self.in_channels))
            self.br_pool.append(
                BiCornerPool(
                    self.in_channels, ['bottom', 'right'],
                    out_channels=self.in_channels))

            self.tl_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))
            self.br_heat.append(
                self._make_layers(
                    out_channels=self.num_classes,
                    in_channels=self.in_channels))

    def _make_layers(self, out_channels, in_channels=256, feat_channels=256):
        """Initialize conv sequential for CornerHead."""
        return nn.Sequential(
            ConvModule(in_channels, feat_channels, 3, padding=1),
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
        lvl_ind = list(range(self.num_feat_levels))
        return multi_apply(self.forward_single, feats, lvl_ind)

    def forward_single(self, x, lvl_ind):
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
            cls_feat1 = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat1 = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat1)
        bbox_pred = self.retina_reg(reg_feat1)

        tl_heat = self.tl_pool[lvl_ind](cls_feat1)
        tl_heat = self.tl_heat[lvl_ind](tl_heat)
        br_heat = self.br_pool[lvl_ind](cls_feat1)
        br_heat = self.br_heat[lvl_ind](br_heat)

        return cls_score, bbox_pred, tl_heat, br_heat

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
        # br_heatmap_all = torch.stack(br_heatmap_all, dim=0)
        # tl_heatmap_all = torch.stack(tl_heatmap_all, dim=0)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result, br_heatmap_all, tl_heatmap_all)#, br_heatmap_all, tl_heatmap_all

    # def get_targets(self,
    #                 gt_bboxes,
    #                 gt_labels,
    #                 feat_shape,
    #                 img_shape):
    #     """Generate corner targets.
    #
    #     Including corner heatmap, corner offset.
    #
    #     Optional: corner embedding, corner guiding shift, centripetal shift.
    #
    #     For CornerNet, we generate corner heatmap, corner offset and corner
    #     embedding from this function.
    #
    #     For CentripetalNet, we generate corner heatmap, corner offset, guiding
    #     shift and centripetal shift from this function.
    #
    #     Args:
    #         gt_bboxes (list[Tensor]): Ground truth bboxes of each image, each
    #             has shape (num_gt, 4).
    #         gt_labels (list[Tensor]): Ground truth labels of each box, each has
    #             shape (num_gt,).
    #         feat_shape (list[int]): Shape of output feature,
    #             [batch, channel, height, width].
    #         img_shape (list[int]): Shape of input image,
    #             [height, width, channel].
    #         with_corner_emb (bool): Generate corner embedding target or not.
    #             Default: False.
    #         with_guiding_shift (bool): Generate guiding shift target or not.
    #             Default: False.
    #         with_centripetal_shift (bool): Generate centripetal shift target or
    #             not. Default: False.
    #
    #     Returns:
    #         dict: Ground truth of corner heatmap, corner offset, corner
    #         embedding, guiding shift and centripetal shift. Containing the
    #         following keys:
    #
    #             - topleft_heatmap (Tensor): Ground truth top-left corner
    #               heatmap.
    #             - bottomright_heatmap (Tensor): Ground truth bottom-right
    #               corner heatmap.
    #             - topleft_offset (Tensor): Ground truth top-left corner offset.
    #             - bottomright_offset (Tensor): Ground truth bottom-right corner
    #               offset.
    #             - corner_embedding (list[list[list[int]]]): Ground truth corner
    #               embedding. Not must have.
    #             - topleft_guiding_shift (Tensor): Ground truth top-left corner
    #               guiding shift. Not must have.
    #             - bottomright_guiding_shift (Tensor): Ground truth bottom-right
    #               corner guiding shift. Not must have.
    #             - topleft_centripetal_shift (Tensor): Ground truth top-left
    #               corner centripetal shift. Not must have.
    #             - bottomright_centripetal_shift (Tensor): Ground truth
    #               bottom-right corner centripetal shift. Not must have.
    #     """
    #     batch_size, _, height, width = feat_shape
    #     img_h, img_w = img_shape[:2]
    #
    #     width_ratio = float(width / img_w)
    #     height_ratio = float(height / img_h)
    #
    #     gt_tl_heatmap = gt_bboxes[-1].new_zeros(
    #         [batch_size, self.num_classes, height, width])
    #     gt_br_heatmap = gt_bboxes[-1].new_zeros(
    #         [batch_size, self.num_classes, height, width])
    #     gt_tl_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
    #     gt_br_offset = gt_bboxes[-1].new_zeros([batch_size, 2, height, width])
    #
    #     for batch_id in range(batch_size):
    #         # Ground truth of corner embedding per image is a list of coord set
    #         corner_match = []
    #         for box_id in range(len(gt_labels[batch_id])):
    #             left, top, right, bottom = gt_bboxes[batch_id][box_id]
    #             center_x = (left + right) / 2.0
    #             center_y = (top + bottom) / 2.0
    #             label = gt_labels[batch_id][box_id]
    #
    #             # Use coords in the feature level to generate ground truth
    #             scale_left = left * width_ratio
    #             scale_right = right * width_ratio
    #             scale_top = top * height_ratio
    #             scale_bottom = bottom * height_ratio
    #             scale_center_x = center_x * width_ratio
    #             scale_center_y = center_y * height_ratio
    #
    #             # Int coords on feature map/ground truth tensor
    #             left_idx = int(min(scale_left, width - 1))
    #             right_idx = int(min(scale_right, width - 1))
    #             top_idx = int(min(scale_top, height - 1))
    #             bottom_idx = int(min(scale_bottom, height - 1))
    #
    #             # Generate gaussian heatmap
    #             scale_box_width = ceil(scale_right - scale_left)
    #             scale_box_height = ceil(scale_bottom - scale_top)
    #             radius = gaussian_radius((scale_box_height, scale_box_width),
    #                                      min_overlap=0.3)
    #             radius = max(0, int(radius))
    #             gt_tl_heatmap[batch_id, label] = gen_gaussian_target(
    #                 gt_tl_heatmap[batch_id, label], [left_idx, top_idx],
    #                 radius)
    #             gt_br_heatmap[batch_id, label] = gen_gaussian_target(
    #                 gt_br_heatmap[batch_id, label], [right_idx, bottom_idx],
    #                 radius)
    #
    #             # Generate corner offset
    #             left_offset = scale_left - left_idx
    #             top_offset = scale_top - top_idx
    #             right_offset = scale_right - right_idx
    #             bottom_offset = scale_bottom - bottom_idx
    #             gt_tl_offset[batch_id, 0, top_idx, left_idx] = left_offset
    #             gt_tl_offset[batch_id, 1, top_idx, left_idx] = top_offset
    #             gt_br_offset[batch_id, 0, bottom_idx, right_idx] = right_offset
    #             gt_br_offset[batch_id, 1, bottom_idx,
    #                          right_idx] = bottom_offset
    #
    #             # Generate corner embedding
    #             if with_corner_emb:
    #                 corner_match.append([[top_idx, left_idx],
    #                                      [bottom_idx, right_idx]])
    #             # Generate guiding shift
    #             if with_guiding_shift:
    #                 gt_tl_guiding_shift[batch_id, 0, top_idx,
    #                                     left_idx] = scale_center_x - left_idx
    #                 gt_tl_guiding_shift[batch_id, 1, top_idx,
    #                                     left_idx] = scale_center_y - top_idx
    #                 gt_br_guiding_shift[batch_id, 0, bottom_idx,
    #                                     right_idx] = right_idx - scale_center_x
    #                 gt_br_guiding_shift[
    #                     batch_id, 1, bottom_idx,
    #                     right_idx] = bottom_idx - scale_center_y
    #             # Generate centripetal shift
    #             if with_centripetal_shift:
    #                 gt_tl_centripetal_shift[batch_id, 0, top_idx,
    #                                         left_idx] = log(scale_center_x -
    #                                                         scale_left)
    #                 gt_tl_centripetal_shift[batch_id, 1, top_idx,
    #                                         left_idx] = log(scale_center_y -
    #                                                         scale_top)
    #                 gt_br_centripetal_shift[batch_id, 0, bottom_idx,
    #                                         right_idx] = log(scale_right -
    #                                                          scale_center_x)
    #                 gt_br_centripetal_shift[batch_id, 1, bottom_idx,
    #                                         right_idx] = log(scale_bottom -
    #                                                          scale_center_y)
    #
    #         if with_corner_emb:
    #             match.append(corner_match)
    #
    #     target_result = dict(
    #         topleft_heatmap=gt_tl_heatmap,
    #         topleft_offset=gt_tl_offset,
    #         bottomright_heatmap=gt_br_heatmap,
    #         bottomright_offset=gt_br_offset)
    #
    #     if with_corner_emb:
    #         target_result.update(corner_embedding=match)
    #     if with_guiding_shift:
    #         target_result.update(
    #             topleft_guiding_shift=gt_tl_guiding_shift,
    #             bottomright_guiding_shift=gt_br_guiding_shift)
    #     if with_centripetal_shift:
    #         target_result.update(
    #             topleft_centripetal_shift=gt_tl_centripetal_shift,
    #             bottomright_centripetal_shift=gt_br_centripetal_shift)
    #
    #     return target_result

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

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_heat=det_loss)

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
        det_loss = (tl_det_loss + br_det_loss) / 2.0

        return loss_cls, loss_bbox, det_loss

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    # def get_bboxes(self,
    #                cls_scores,
    #                bbox_preds,
    #                tl_heat,
    #                br_heat,
    #                img_metas,
    #                cfg=None,
    #                rescale=False,
    #                with_nms=True):
    #     """Transform network output for a batch into bbox predictions.
    #
    #     Args:
    #         cls_scores (list[Tensor]): Box scores for each scale level
    #             Has shape (N, num_anchors * num_classes, H, W)
    #         bbox_preds (list[Tensor]): Box energies / deltas for each scale
    #             level with shape (N, num_anchors * 4, H, W)
    #         img_metas (list[dict]): Meta information of each image, e.g.,
    #             image size, scaling factor, etc.
    #         cfg (mmcv.Config | None): Test / postprocessing configuration,
    #             if None, test_cfg would be used
    #         rescale (bool): If True, return boxes in original image space.
    #             Default: False.
    #         with_nms (bool): If True, do nms before return boxes.
    #             Default: True.
    #
    #     Returns:
    #         list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
    #             The first item is an (n, 5) tensor, where the first 4 columns
    #             are bounding box positions (tl_x, tl_y, br_x, br_y) and the
    #             5-th column is a score between 0 and 1. The second item is a
    #             (n,) tensor where each item is the predicted class labelof the
    #             corresponding box.
    #
    #     Example:
    #         >>> import mmcv
    #         >>> self = AnchorHead(
    #         >>>     num_classes=9,
    #         >>>     in_channels=1,
    #         >>>     anchor_generator=dict(
    #         >>>         type='AnchorGenerator',
    #         >>>         scales=[8],
    #         >>>         ratios=[0.5, 1.0, 2.0],
    #         >>>         strides=[4,]))
    #         >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
    #         >>> cfg = mmcv.Config(dict(
    #         >>>     score_thr=0.00,
    #         >>>     nms=dict(type='nms', iou_thr=1.0),
    #         >>>     max_per_img=10))
    #         >>> feat = torch.rand(1, 1, 3, 3)
    #         >>> cls_score, bbox_pred = self.forward_single(feat)
    #         >>> # note the input lists are over different levels, not images
    #         >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
    #         >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
    #         >>>                               img_metas, cfg)
    #         >>> det_bboxes, det_labels = result_list[0]
    #         >>> assert len(result_list) == 1
    #         >>> assert det_bboxes.shape[1] == 5
    #         >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
    #     """
    #     assert len(cls_scores) == len(bbox_preds)
    #     num_levels = len(cls_scores)
    #
    #     device = cls_scores[0].device
    #     featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    #     mlvl_anchors = self.anchor_generator.grid_anchors(
    #         featmap_sizes, device=device)
    #
    #     result_list = []
    #     for img_id in range(len(img_metas)):
    #         cls_score_list = [
    #             cls_scores[i][img_id].detach() for i in range(num_levels)
    #         ]
    #         bbox_pred_list = [
    #             bbox_preds[i][img_id].detach() for i in range(num_levels)
    #         ]
    #         img_shape = img_metas[img_id]['img_shape']
    #         scale_factor = img_metas[img_id]['scale_factor']
    #         if with_nms:
    #             # some heads don't support with_nms argument
    #             proposals = self._get_bboxes_single(cls_score_list,
    #                                                 bbox_pred_list,
    #                                                 mlvl_anchors, img_shape,
    #                                                 scale_factor, cfg, rescale)
    #         else:
    #             proposals = self._get_bboxes_single(cls_score_list,
    #                                                 bbox_pred_list,
    #                                                 mlvl_anchors, img_shape,
    #                                                 scale_factor, cfg, rescale,
    #                                                 with_nms)
    #         result_list.append(proposals)
    #     return result_list

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
        for cls_score, bbox_pred, anchors, tl_heat, br_heat in zip(cls_score_list, bbox_pred_list,
                                                                             mlvl_anchors, tl_heat_list, br_heat_list):
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


    def multiclass_nms(self,
                       multi_bboxes,
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
            tuple: (bboxes, labels, indices (optional)), tensors of shape (k, 5),
                (k), and (k). Labels are 0-based.
        """
        num_classes = multi_scores.size(1) - 1
        # exclude background category
        if multi_bboxes.shape[1] > 4:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 4)

        scores = multi_scores[:, :-1]

        labels = torch.arange(num_classes, dtype=torch.long)
        labels = labels.view(1, -1).expand_as(scores)

        bboxes = bboxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        tl_heat = tl_heat.reshape(-1)
        br_heat = br_heat.reshape(-1)
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
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
        br_heat, tl_heat = br_heat[inds], tl_heat[inds]
        if inds.numel() == 0:
            if torch.onnx.is_in_onnx_export():
                raise RuntimeError('[ONNX Error] Can not record NMS '
                                   'as it has not been executed this time')
            if return_inds:
                return bboxes, labels, inds
            else:
                return bboxes, labels

        # TODO: add size check before feed into batched_nms
        dets, keep = self.batched_nms(bboxes, scores, labels, nms_cfg, br_heat, tl_heat)

        if max_num > 0:
            dets = dets[:max_num]
            keep = keep[:max_num]

        if return_inds:
            return dets, keep
        else:
            return dets, keep

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
            dets, keep = self.nms(boxes_for_nms, scores.unsqueeze(1), idxs, br_heat, tl_heat, nms_cfg_['iou_threshold'])
            boxes = dets
            scores = dets[:, -1]

        else:
            total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
            for id in torch.unique(idxs):
                mask = (idxs == id).nonzero(as_tuple=False).view(-1)
                # dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
                dets, keep = self.nms(boxes_for_nms[mask], scores[mask].unsqueeze(1), idxs[mask], br_heat[mask], tl_heat[mask], nms_cfg_['iou_threshold'])
                total_mask[mask[keep]] = True

            keep = total_mask.nonzero(as_tuple=False).view(-1)
            keep = keep[scores[keep].argsort(descending=True)]
            boxes = dets
            scores = scores[keep]


        return boxes, torch.tensor(keep).to(boxes.device)

    def nms(self, boxes, scores, idxs, br_heat, tl_heat, overlap=0.7, top_k=200):
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
        cls = []
        for cls_id in range(self.num_classes):
            mask = idxs==cls_id
            score = scores[mask]
            box = boxes[mask]
            tl_heat_ = tl_heat[mask]
            br_heat_ = br_heat[mask]
            keep = score.new(score.size(0)).zero_().long()
            # 保存留下来的box的索引 [num_positive]
            # 函数new(): 构建一个有相同数据类型的tensor

            # 如果输入box为空则返回空Tensor
            if box.numel() == 0:
                return keep

            x1 = box[:, 0]  # x1 坐标
            y1 = box[:, 1]
            x2 = box[:, 2]
            y2 = box[:, 3]
            area = torch.mul(x2 - x1, y2 - y1)  # 并行化计算所有框的面积
            v, idx = score.sort(0)  # 升序排序
            idx = idx[-top_k:].squeeze(1)  # 前top-k的索引，从小到大
            xx1 = box.new()
            yy1 = box.new()
            xx2 = box.new()  # new() 无参数，创建 相同类型的空值；
            yy2 = box.new()
            w = box.new()
            h = box.new()

            count = 0
            while idx.numel() > 0:
                i = idx[-1]
                cla_score = score[i]# 目前最大score对应的索引  # 选取得分最大的框索引；
                keep[count] = i  # 存储在keep中
                count += 1
                if idx.size(0) == 1:  # 跳出循环条件：box被筛选完了
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

                try:
                    # a = score[idx]
                    # mask1 = IoU.gt(overlap) & (score[idx]>=(cla_score-0.2)).squeeze(1)
                    mask1 = IoU.gt(overlap)
                    # mask1 = (score[idx]>=(score[idx].max()-torch.std(score[idx][mask1]))).squeeze(1)
                    # tl_x1 = x1[idx][mask1][torch.argmax(tl_heat_[idx][mask1])].unsqueeze(0)
                    # tl_y1 = y1[idx][mask1][torch.argmax(tl_heat_[idx][mask1])].unsqueeze(0)
                    # br_x2 = x2[idx][mask1][torch.argmax(br_heat_[idx][mask1])].unsqueeze(0)
                    # br_y2 = y2[idx][mask1][torch.argmax(br_heat_[idx][mask1])].unsqueeze(0)
                    # a = tl_heat_[idx][mask1]
                    # b = tl_heat_[i].unsqueeze(0)
                    # c = x1[idx][mask1]
                    # d = x1[i]
                    tl_x1 = torch.cat([x1[idx][mask1],x1[i].unsqueeze(0)],dim=0)[torch.argmax(torch.cat([tl_heat_[idx][mask1],tl_heat_[i].unsqueeze(0)],dim=0))].unsqueeze(0)
                    tl_y1 = torch.cat([y1[idx][mask1],y1[i].unsqueeze(0)],dim=0)[torch.argmax(torch.cat([tl_heat_[idx][mask1],tl_heat_[i].unsqueeze(0)],dim=0))].unsqueeze(0)
                    br_x2 = torch.cat([x2[idx][mask1],x2[i].unsqueeze(0)],dim=0)[torch.argmax(torch.cat([br_heat_[idx][mask1],br_heat_[i].unsqueeze(0)],dim=0))].unsqueeze(0)
                    br_y2 = torch.cat([y2[idx][mask1],y2[i].unsqueeze(0)],dim=0)[torch.argmax(torch.cat([br_heat_[idx][mask1],br_heat_[i].unsqueeze(0)],dim=0))].unsqueeze(0)
                    # tl_x1 = torch.cat([x1[idx][mask1],x1[i].unsqueeze(0)],dim=0)[torch.topk(torch.cat([tl_heat_[idx][mask1],tl_heat_[i].unsqueeze(0)],dim=0),k=3)[1]].mean().unsqueeze(0)
                    # tl_y1 = torch.cat([y1[idx][mask1],y1[i].unsqueeze(0)],dim=0)[torch.topk(torch.cat([tl_heat_[idx][mask1],tl_heat_[i].unsqueeze(0)],dim=0),k=3)[1]].mean().unsqueeze(0)
                    # br_x2 = torch.cat([x2[idx][mask1],x2[i].unsqueeze(0)],dim=0)[torch.topk(torch.cat([br_heat_[idx][mask1],br_heat_[i].unsqueeze(0)],dim=0),k=3)[1]].mean().unsqueeze(0)
                    # br_y2 = torch.cat([y2[idx][mask1],y2[i].unsqueeze(0)],dim=0)[torch.topk(torch.cat([br_heat_[idx][mask1],br_heat_[i].unsqueeze(0)],dim=0),k=3)[1]].mean().unsqueeze(0)
                    # a = boxes[i]
                    # b = torch.stack([tl_x1,tl_y1,br_x2,br_y2,cla_score],dim=1)
                    det.append(torch.stack([tl_x1, tl_y1, br_x2, br_y2, cla_score],dim=1))
                    cls.append(cls_id)
                except:
                    continue

                # 选出IoU <= overlap的boxes(注意le函数的使用)
                idx = idx[IoU.le(overlap)]  # le: 小于等于 返回的bool ， 去除大于overlap的值；
        det = torch.cat(det,dim=0)
        return det, cls


class NMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, bboxes, scores, iou_threshold, offset):
        inds = ext_module.nms(
            bboxes, scores, iou_threshold=float(iou_threshold), offset=offset)
        return inds

    @staticmethod
    def symbolic(g, bboxes, scores, iou_threshold, offset):
        from mmcv.onnx import is_custom_op_loaded
        has_custom_op = is_custom_op_loaded()
        if has_custom_op:
            return g.op(
                'mmcv::NonMaxSuppression',
                bboxes,
                scores,
                iou_threshold_f=float(iou_threshold),
                offset_i=int(offset))
        else:
            from torch.onnx.symbolic_opset9 import select, squeeze, unsqueeze
            boxes = unsqueeze(g, bboxes, 0)
            scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
            max_output_per_class = g.op(
                'Constant',
                value_t=torch.tensor([sys.maxsize], dtype=torch.long))
            iou_threshold = g.op(
                'Constant',
                value_t=torch.tensor([iou_threshold], dtype=torch.float))
            nms_out = g.op('NonMaxSuppression', boxes, scores,
                           max_output_per_class, iou_threshold)
            return squeeze(
                g,
                select(
                    g, nms_out, 1,
                    g.op(
                        'Constant',
                        value_t=torch.tensor([2], dtype=torch.long))), 1)

def non_max_suppress(predicts_dict, scores, idxs, threshold=0.2):
    """
    implement non-maximum supression on predict bounding boxes.
    Args:
        predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
        threshhold: iou threshold
    Return:
        predicts_dict processed by non-maximum suppression
    """
    for bbox in predicts_dict:  # 对每一个类别的目标分别进行NMS
        bbox_array = np.array(bbox, dtype=np.float)

        ## 获取当前目标类别下所有矩形框（bounding box,下面简称bbx）的坐标和confidence,并计算所有bbx的面积
        x1, y1, x2, y2, scores = bbox_array[:, 0], bbox_array[:, 1], bbox_array[:, 2], bbox_array[:, 3], bbox_array[:,
                                                                                                         4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # print("areas shape = ", areas.shape)

        ## 对当前类别下所有的bbx的confidence进行从高到低排序（order保存索引信息）
        order = scores.argsort()[::-1]
        print("order = ", order)
        keep = []  # 用来存放最终保留的bbx的索引信息

        ## 依次从按confidence从高到低遍历bbx，移除所有与该矩形框的IOU值大于threshold的矩形框
        while order.size > 0:
            i = order[0]
            keep.append(i)  # 保留当前最大confidence对应的bbx索引

            ## 获取所有与当前bbx的交集对应的左上角和右下角坐标，并计算IOU（注意这里是同时计算一个bbx与其他所有bbx的IOU）
            xx1 = np.maximum(x1[i], x1[order[1:]])# 当order.size=1时，下面的计算结果都为np.array([]),不影响最终结果
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            print("iou =", iou)

            print(np.where(iou <= threshold))  # 输出没有被移除的bbx索引（相对于iou向量的索引）
            indexs = np.where(iou <= threshold)[0] + 1  # 获取保留下来的索引(因为没有计算与自身的IOU，所以索引相差１，需要加上)
            print("indexs = ", type(indexs))
            order = order[indexs]  # 更新保留下来的索引
            print("order = ", order)
        bbox = bbox_array[keep]
        predicts_dict = bbox.tolist()
        predicts_dict = predicts_dict
    return predicts_dict

def nms(boxes, scores, overlap=0.7, top_k=200):
    """
    输入:
        boxes: 存储一个图片的所有预测框。[num_positive,4].
        scores:置信度。如果为多分类则需要将nms函数套在一个循环内。[num_positive].
        overlap: nms抑制时iou的阈值.
        top_k: 先选取置信度前top_k个框再进行nms.
    返回:
        nms后剩余预测框的索引.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    # 保存留下来的box的索引 [num_positive]
    # 函数new(): 构建一个有相同数据类型的tensor

    # 如果输入box为空则返回空Tensor
    if boxes.numel() == 0:
        return keep

    x1 = boxes[:, 0]  # x1 坐标
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)  # 并行化计算所有框的面积
    v, idx = scores.sort(0)  # 升序排序
    idx = idx[-top_k:]  # 前top-k的索引，从小到大
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()  # new() 无参数，创建 相同类型的空值；
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # 目前最大score对应的索引  # 选取得分最大的框索引；
        keep[count] = i  # 存储在keep中
        count += 1
        if idx.size(0) == 1:  # 跳出循环条件：box被筛选完了
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

        # 选出IoU <= overlap的boxes(注意le函数的使用)
        idx = idx[IoU.le(overlap)]  # le: 小于等于 返回的bool ， 去除大于overlap的值；
    return keep, count
