import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from torch import Tensor
import torch.nn as nn

@HEADS.register_module()
class StandardRoIHead_IAN(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    class IAN():
        def __init__(self, spatial_threshold, channel_threshold):
            self.Ts = spatial_threshold
            self.Tc = channel_threshold
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        def __call__(self,
                     features: Tensor,
                     gradients: Tensor):
            N, C, H, W = features.size()
            feat = features.clone().detach()  # [N,C,H,W]
            grad = gradients.clone().detach()  # [N,C,H,W]
            pooled_grad = self.pool(grad)  # [N,C,1,1]
            attention_map = pooled_grad * feat  # [N,C,H,W]
            # spatial Thresh
            # inverted_attention_map = torch.ones_like(attention_map)  # [[1,1..1]]
            # inverted_attention_map[attention_map > self.Ts] = 0
            # # channel Thresh
            # boolen_grad = (pooled_grad <= self.Tc).repeat(1, 1, H, W)
            # inverted_attention_map[boolen_grad] = 1

            # spatial Thresh
            inverted_attention_map = torch.zeros_like(attention_map)  # [[1,1..1]]
            inverted_attention_map[attention_map > self.Ts] = 1
            # channel Thresh
            boolen_grad = (pooled_grad >= self.Tc).repeat(1, 1, H, W)
            inverted_attention_map[boolen_grad] = 1

            return inverted_attention_map.requires_grad_(True)

    def save_gradients(self, grad):
        self.gradients = grad

    def _bbox_forward1(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        feat = x
        if self.training:
            self.gradients: Tensor = None
            self.invertedAttNet = self.IAN(
                spatial_threshold=1e-7,
                channel_threshold=1e-6, )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            reg_set = []
            cls_set = []
            reg_subset = []
            cls_subset = []
            for j in range(len(x[0])):
                clone_set = []
                for i in range(len(x)):
                    clone = feat[i][j].unsqueeze(0).detach()
                    clone.requires_grad_()
                    clone.register_hook(self.save_gradients)
                    clone = [clone]
                    # locals()['clone'+str(i)] = feat[i][j].unsqueeze(0).detach()
                    # locals()['clone' + str(i)].requires_grad_()
                    # locals()['clone' + str(i)].register_hook(self.save_gradients)
                    # for p in range(5):
                    #     clone_set.append(locals()['clone' + str(i)] )
                    # clone_set = tuple(clone_set)
                # for k in range(len(x)):
                    bbox_feats = self.bbox_roi_extractor(
                        tuple(clone), rois)
                    if self.with_shared_head:
                        bbox_feats = self.shared_head(bbox_feats)
                    cls, reg = self.bbox_head(bbox_feats)

                    total_cls = torch.sum(cls) / (cls.size(0) * cls.size(1))  # sum(tensor:[N,])/N
                    total_cls.backward(retain_graph=True)
                    cls_att = self.gradients
                    # self.zero_grad()
                    if self.with_shared_head:
                        self.shared_head.zero_grad()
                    self.bbox_roi_extractor.zero_grad()
                    # with torch.no_grad():
                    #     self.bbox_head.zero_grad()

                    total_reg = torch.sum(reg) / (reg.size(0) * reg.size(1))  # sum(tensor:[N,])/N
                    total_reg.backward(retain_graph=True)
                    reg_att = self.gradients
                    # self.zero_grad()
                    if self.with_shared_head:
                        self.shared_head.zero_grad()
                    self.bbox_roi_extractor.zero_grad()
                    # with torch.no_grad():
                    #     self.bbox_head.zero_grad()

                    cls_attention_map = self.invertedAttNet(locals()['clone'+str(k)], cls_att)  # + reg_att cls_att
                    reg_attention_map = self.invertedAttNet(locals()['clone'+str(k)], reg_att) # + reg_att cls_att

                    reg_subset.append(locals()['clone'+str(k)] * cls_attention_map + locals()['clone'+str(k)])
                    cls_subset.append(locals()['clone'+str(k)] * reg_attention_map + locals()['clone'+str(k)])
                reg_set.append(reg_subset)
                cls_set.append(cls_subset)
            reg_ = []
            for bs1, bs2 in zip(reg_set[0],reg_set[1]):
                reg_.append(torch.cat([bs1,bs2],dim=0))
            reg_set = tuple(reg_)

            cls_ = []
            for bs1, bs2 in zip(cls_set[0],cls_set[1]):
                cls_.append(torch.cat([bs1,bs2],dim=0))
            cls_set = tuple(cls_)

        else:
            cls_set = x
            reg_set = x

        cls_feats = self.bbox_roi_extractor(
            cls_set[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            cls_feats = self.shared_head(cls_feats)
        cls_score, _ = self.bbox_head(cls_feats)

        bbox_feats = self.bbox_roi_extractor(
            reg_set[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        _, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)

        feat = bbox_feats
        if self.training:
            self.gradients: Tensor = None
            self.invertedAttNet = self.IAN(
                spatial_threshold=1e-7,
                channel_threshold=1e-6, )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            reg_set = []
            cls_set = []
            for i in range(len(bbox_feats)):
                clone = feat[i].unsqueeze(0).detach()
                clone.requires_grad_()
                clone.register_hook(self.save_gradients)

                if self.with_shared_head:
                    clone = self.shared_head(clone)
                cls, reg = self.bbox_head(clone)

                total_cls = torch.sum(cls) / (cls.size(0) * cls.size(1))  # sum(tensor:[N,])/N
                total_cls.backward(retain_graph=True)
                cls_att = self.gradients
                self.zero_grad()

                total_reg = torch.sum(reg) / (reg.size(0) * reg.size(1))  # sum(tensor:[N,])/N
                total_reg.backward(retain_graph=True)
                reg_att = self.gradients
                self.zero_grad()

                cls_attention_map = self.invertedAttNet(feat[0].unsqueeze(0), cls_att).squeeze(0)  # + reg_att cls_att
                reg_attention_map = self.invertedAttNet(feat[0].unsqueeze(0), reg_att).squeeze(0)  # + reg_att cls_att

                reg_set.append(feat[0] * cls_attention_map + feat[0])
                cls_set.append(feat[0] * reg_attention_map + feat[0])
            reg_set = torch.stack(reg_set, dim=0)
            cls_set = torch.stack(cls_set, dim=0)
        else:
            cls_set = bbox_feats
            reg_set = bbox_feats

        if self.with_shared_head:
            reg_set = self.shared_head(reg_set)
        _, bbox_pred = self.bbox_head(reg_set)

        if self.with_shared_head:
            cls_set = self.shared_head(cls_set)
        cls_score, _ = self.bbox_head(cls_set)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        if torch.onnx.is_in_onnx_export():
            if self.with_mask:
                segm_results = self.simple_test_mask(
                    x, img_metas, det_bboxes, det_labels, rescale=rescale)
                return det_bboxes, det_labels, segm_results
            else:
                return det_bboxes, det_labels

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]
