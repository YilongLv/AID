# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
import mmcv

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .kd_one_stage import KnowledgeDistillationSingleStageDetector
from pathlib import Path
from .. import build_detector
from .single_stage import SingleStageDetector
import os

@DETECTORS.register_module()
class Kdpisa_step(SingleStageDetector):
    """Implementation of `LAD <https://arxiv.org/pdf/2108.10520.pdf>`_."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config, (str, Path)):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        self.cpkt_dir = teacher_ckpt
        self.step_size = 1

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self.epoch != 0 :
            epoch_n_ckpt = self.step_size * (self.epoch+1)
            epoch_n_ckpt_path = self.cpkt_dir + 'epoch_{}.pth'.format(epoch_n_ckpt)
            if os.path.exists(epoch_n_ckpt_path):
                load_checkpoint(
                    self.teacher_model, epoch_n_ckpt_path, map_location='cpu')


    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # get label assignment from the teacher
        with torch.no_grad():
            x_teacher = self.teacher_model.extract_feat(img)
            outs_teacher = self.teacher_model.bbox_head(x_teacher)
            teacher_targets = self.bbox_head.r_p(
                    *outs_teacher, gt_bboxes, gt_labels, img_metas,
                    gt_bboxes_ignore)

        # the student use the label assignment from the teacher to learn
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, teacher_targets,
                                              img_metas, gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)
        return losses


    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)