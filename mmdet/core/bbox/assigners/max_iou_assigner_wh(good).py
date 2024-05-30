import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class MaxIoUAssigner_WH(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def calc_wh(self,a, b):
        # aw = torch.from_numpy(a[:, 2] - a[:, 0])
        # ah = torch.from_numpy(a[:, 3] - a[:, 1])
        # bw = torch.from_numpy(b[:, 2] - b[:, 0])
        # bh = torch.from_numpy(b[:, 3] - b[:, 1])

        aw = a[:, 2] - a[:, 0]
        ah = a[:, 3] - a[:, 1]
        bw = b[:, 2] - b[:, 0]
        bh = b[:, 3] - b[:, 1]

        w_min = torch.min(aw[:, None], bw[None, :])  # [batch_size,h*w]
        w_max = torch.max(aw[:, None], bw[None, :])
        h_min = torch.min(ah[:, None], bh[None, :])
        h_max = torch.max(ah[:, None], bh[None, :])
        wh_targets = ((w_min * h_min) / (w_max * h_max + 1e-10)).sqrt().unsqueeze(
            dim=-1)
        return wh_targets.squeeze(2)

    def calc_cen(self, b, a):
        x = (a[:, 2] + a[:, 0]) / 2
        y = (a[:, 3] + a[:, 1]) / 2
        l_off = x[:, None] - b[:, 0][None, :]
        b_off = y[:, None] - b[:, 1][None, :]
        r_off = b[:, 2][None, :] - x[:, None]
        t_off = b[:, 3][None, :] - y[:, None]
        ltrb_off = torch.stack([l_off, b_off, r_off, t_off], dim=-1)
        try:
            off_min = torch.min(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]
        except:
            off_min = torch.zeros_like(ltrb_off)
        mask_in_gtboxes = off_min > 0

        # gt_x = (b[:, 2] - b[:, 0]) / 2
        # gt_y = (b[:, 3] - b[:, 1]) / 2
        # c_l_off = x[:, None] - gt_x[None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        # c_t_off = y[:, None] - gt_y[None, :]
        # c_r_off = gt_x[None, :] - x[:, None]
        # c_b_off = gt_y[None, :] - y[:, None]
        # c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)  # [batch_size,h*w,m,4]
        # c_off_max = torch.max(c_ltrb_off, dim=-1)[0]
        # mask_center = c_off_max < (2.5 * stride.unsqueeze(1))

        mask = mask_in_gtboxes  # mask_center&

        ltrb_off[~mask] = 0

        left_right_min = torch.min(ltrb_off[..., 0], ltrb_off[..., 2])  # [batch_size,h*w]
        left_right_max = torch.max(ltrb_off[..., 0], ltrb_off[..., 2])
        top_bottom_min = torch.min(ltrb_off[..., 1], ltrb_off[..., 3])
        top_bottom_max = torch.max(ltrb_off[..., 1], ltrb_off[..., 3])

        cnt_targets = ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(
            dim=-1)

        return cnt_targets.transpose(1,0).squeeze(2)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes)
        WH = self.calc_wh(gt_bboxes,bboxes)
        cen = self.calc_cen(gt_bboxes,bboxes)
        overlaps = (1-cen) * overlaps + cen * WH

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # 3. assign positive: above positive IoU threshold
        gt_assignment_pos = list()
        if overlaps.shape[0] > 1:
            for i in range(overlaps.shape[0]):
                # mean = iou[:,i].mean()
                std = overlaps[i, :].std()
                thre = self.pos_iou_thr + 1 * std

                iou_i_pos = torch.zeros(overlaps.shape[1]).to(overlaps.device)
                # thre = mean + 3 * std
                iou_i_pos[overlaps[i, :] > thre] = 1
                # a = (iou_i_pos>0).sum()
                # iou_i_pos[torch.topk(torch.tensor(iou[:,i]),5)[1]] = 1
                iou_i_pos = iou_i_pos * torch.tensor(overlaps[i, :])
                gt_assignment_pos.append(iou_i_pos.unsqueeze(0))
            overlaps_pos = torch.cat(gt_assignment_pos, dim=0)
        else:
            # mean = iou[:, 0].mean()
            std = overlaps[0].std()
            # thre = mean + 3 * std
            thre = self.pos_iou_thr + 1 * std
            iou_i_pos = torch.zeros(overlaps.shape[1]).to(overlaps.device)
            iou_i_pos[overlaps[0] > thre] = 1
            # a = (iou_i_pos > 0).sum()
            # iou_i_pos[torch.topk(torch.tensor(iou[:, 0]), 5)[1]] = 1
            iou_i_pos = iou_i_pos * torch.tensor(overlaps[0])
            overlaps_pos = iou_i_pos.unsqueeze(0)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps_pos.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps_pos.max(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        pos_inds = max_overlaps > 0
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if self.match_low_quality:
            # Low-quality matching will overwirte the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox B.
            # This might be the reason that it is not used in ROI Heads.
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0


        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
