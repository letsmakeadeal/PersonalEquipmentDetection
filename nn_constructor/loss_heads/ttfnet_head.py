import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


from mmcv.cnn import build_norm_layer, ConvModule, build_activation_layer


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def py_modified_sigmoid_focal_loss(pred, target, gamma=2.0):
    pred = torch.clamp(pred.sigmoid_(), min=1e-4, max=1 - 1e-4)
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)  # reduce punishment
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred,
                                                gamma) * neg_inds * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return neg_loss
    return (pos_loss + neg_loss) / num_pos


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _is_enumerable(obj):
    return isinstance(obj, list) or isinstance(obj, tuple)


class UpsamplingLayers(nn.Sequential):

    def __init__(self, in_channels, out_channels,
                 norm_cfg=dict(type='BN'), activation_cfg=dict(type='ReLU')):
        mdcn = nn.Conv2d(in_channels, out_channels, 3,
                             stride=1, padding=1, dilation=1, bias=norm_cfg is None)
        up = nn.UpsamplingNearest2d(scale_factor=2)

        layers = []
        layers.append(mdcn)
        if norm_cfg is not None:
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        layers.append(build_activation_layer(activation_cfg))
        layers.append(up)
        super(UpsamplingLayers, self).__init__(*layers)


class ShortcutConnection(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes,
                 norm_cfg=dict(type='BN'), activation_cfg=dict(type='ReLU')):
        super(ShortcutConnection, self).__init__()
        layers = []
        for i, kernel_size in enumerate(kernel_sizes):
            inc = in_channels if i == 0 else out_channels
            padding = (kernel_size - 1) // 2
            layers.append(
                nn.Conv2d(inc, out_channels, kernel_size, padding=padding, bias=norm_cfg is None))
            if norm_cfg is not None:
                layers.append(build_norm_layer(norm_cfg, out_channels)[1])
            if i < len(kernel_sizes) - 1:
                layers.append(build_activation_layer(activation_cfg))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TTFHead(nn.Module):
    """Training-Time-Friendly Network for Real-Time Object Detection
    https://arxiv.org/abs/1909.00700
    """

    def __init__(self,
                 build_down_top=True,
                 inplanes=(64, 128, 256, 512),
                 planes=(256, 128, 64),
                 down_ratio=4,
                 hm_head_channels=64,
                 wh_head_channels=64,
                 hm_head_conv_num=2,
                 wh_head_conv_num=2,
                 num_classes=80,
                 shortcut_cfg=(1, 2, 3),
                 wh_scale_factor=16.,
                 alpha=0.54,
                 beta=0.54,
                 max_objs=128,
                 hm_weight=1.,
                 loss_bbox=dict(type='GIoULoss', loss_weight=5.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 activation_cfg=dict(type='ReLU'),
                 wh_agnostic=True):
        super(TTFHead, self).__init__()
        # TODO(i.rodin): Circular dependency issue
        from utilities.builders import build_loss_from_cfg
        if build_down_top:
            assert len(planes) in [2, 3, 4]
        else:
            assert len(planes) == 1

        self.inplanes = inplanes
        self.planes = planes
        self.down_ratio = down_ratio
        self.hm_head_channels = hm_head_channels
        self.wh_head_channels = wh_head_channels
        self.hm_head_conv_num = hm_head_conv_num
        self.wh_head_conv_num = wh_head_conv_num
        self.num_classes = num_classes
        self.shortcut_cfg = shortcut_cfg
        self.wh_scale_factor = wh_scale_factor
        self.alpha = alpha
        self.beta = beta
        self.hm_weight = hm_weight
        self.max_objs = max_objs
        self.loss_bbox = build_loss_from_cfg(loss_bbox)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation_cfg = activation_cfg
        self.base_loc = None
        self.fp16_enabled = False
        self.wh_agnostic = wh_agnostic
        self.wh_planes = 4 if wh_agnostic else 4 * self.num_classes
        self.build_down_top = build_down_top

        self._init_layers()

    def _init_layers(self):
        # upsample 3 times (1/32 => 1/4).
        if self.build_down_top:
            self.deconv_layers = nn.ModuleList()
            for i in range(len(self.planes)):
                in_channels = self.inplanes[-1] if i == 0 else self.planes[i - 1]
                self.deconv_layers.append(
                    UpsamplingLayers(
                        in_channels, self.planes[i],
                        norm_cfg=self.norm_cfg, activation_cfg=self.activation_cfg))

            shortcut_num = min(len(self.inplanes) - 1, len(self.planes))
            assert shortcut_num == len(self.shortcut_cfg)
            self.shortcut_layers = nn.ModuleList()
            for (inp, outp, layer_num) in zip(self.inplanes[:-1][::-1][:shortcut_num],
                                              self.planes[:shortcut_num], self.shortcut_cfg):
                assert layer_num > 0, "Shortcut connection must be included."
                self.shortcut_layers.append(ShortcutConnection(inp, outp, [3] * layer_num,
                                                               norm_cfg=self.norm_cfg,
                                                               activation_cfg=self.activation_cfg))

        # heads
        wh_layers, hm_layers = [], []
        inp = self.planes[-1]
        for i in range(self.wh_head_conv_num):
            wh_layers.append(
                ConvModule(
                    inp,
                    self.wh_head_channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    act_cfg=self.activation_cfg))
            inp = self.wh_head_channels
        wh_layers.append(nn.Conv2d(self.wh_head_channels, self.wh_planes, 1))

        inp = self.planes[-1]
        for i in range(self.hm_head_conv_num):
            hm_layers.append(
                ConvModule(
                    inp,
                    self.hm_head_channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    act_cfg=self.activation_cfg))
            inp = self.hm_head_channels
        hm_layers.append(nn.Conv2d(inp, self.num_classes, 1))

        self.wh = nn.Sequential(*wh_layers)
        self.hm = nn.Sequential(*hm_layers)

    def init_weights(self):
        if self.build_down_top:
            for m in self.deconv_layers.modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            for m in self.shortcut_layers.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)

        for m in self.hm.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.hm[-1], std=0.01, bias=bias_cls)

        for m in self.wh.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)

    def forward(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm: tensor, (batch, class_num, h, w).
            wh: tensor, (batch, 4, h, w) or (batch, class_num * 4, h, w).
        """
        x = feats[-1] if _is_enumerable(feats) else feats

        if self.build_down_top:
            for i, upsampling_layer in enumerate(self.deconv_layers):
                x = upsampling_layer(x)
                if i < len(self.shortcut_layers):
                    shortcut = self.shortcut_layers[i](feats[-i - 2])
                    x = x + shortcut

        hm = self.hm(x)
        wh = F.relu(self.wh(x)) * self.wh_scale_factor

        return hm, wh

    def forward_export(self, feats):
        """

        Args:
            feats: list(tensor).

        Returns:
            hm_wh: tensor, (batch, class_num + 4, h, w) or (batch, class_num + 4 * class_num, h, w).
        """

        pred_heatmap, pred_wh = self.forward(feats)

        bs, _, height, width = map(int, pred_heatmap.size())
        pred_heatmap = pred_heatmap.sigmoid()

        heat = self.simple_nms(pred_heatmap)

        base_step = self.down_ratio
        shifts_x = np.arange(0, width * base_step, base_step, dtype=np.float32)
        shifts_y = np.arange(0, height * base_step, base_step, dtype=np.float32)
        shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
        base_loc = torch.tensor(np.stack((shift_x, shift_y), axis=0)).to(pred_wh.device)  # (2, h, w)

        if not self.wh_agnostic:
            pred_wh = pred_wh.view(bs, self.num_classes, 4, height, width)
            pred_boxes = torch.cat((base_loc - pred_wh[:, :, [0, 1]],
                                    base_loc + pred_wh[:, :, [2, 3]]), dim=2)
            pred_boxes = pred_boxes.view(bs, self.num_classes * 4, height, width)
        else:
            pred_boxes = torch.cat((base_loc - pred_wh[:, [0, 1]],
                                    base_loc + pred_wh[:, [2, 3]]), dim=1)

        res = torch.cat([heat, pred_boxes], 1)
        return res

    def get_bboxes(self, predicted, img_metas):
        pred_hm, pred_wh = predicted
        batch, cat, height, width = pred_hm.size()
        pred_hm = pred_hm.detach().sigmoid_()
        wh = pred_wh.detach()

        # used maxpool to filter the max score
        heat = self.simple_nms(pred_hm)

        topk = 100  # TODO(i.rodin) Move to config?
        # (batch, topk)
        scores, inds, clses, ys, xs = self._topk(heat, topk=topk)
        xs = xs.view(batch, topk, 1) * self.down_ratio
        ys = ys.view(batch, topk, 1) * self.down_ratio

        wh = wh.permute(0, 2, 3, 1).contiguous()
        wh = wh.view(wh.size(0), -1, wh.size(3))
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), wh.size(2))
        wh = wh.gather(1, inds)

        if not self.wh_agnostic:
            wh = wh.view(-1, topk, self.num_classes, 4)
            wh = torch.gather(wh, 2, clses[..., None, None].expand(
                clses.size(0), clses.size(1), 1, 4).long())

        wh = wh.view(batch, topk, 4)
        clses = clses.view(batch, topk, 1).float()
        scores = scores.view(batch, topk, 1)

        bboxes = torch.cat([
            xs - wh[..., [0]], ys - wh[..., [1]],
            xs + wh[..., [2]], ys + wh[..., [3]]
        ], dim=2)

        result_list = []
        score_thr = 0.01  # TODO(i.rodin) Move to cfg
        for batch_i in range(bboxes.shape[0]):
            scores_per_img = scores[batch_i]
            scores_keep = (scores_per_img > score_thr).squeeze(-1)

            scores_per_img = scores_per_img[scores_keep]
            bboxes_per_img = bboxes[batch_i][scores_keep]
            labels_per_img = clses[batch_i][scores_keep]
            img_shape = (img_metas[batch_i]['image_height'],
                         img_metas[batch_i]['image_width'])
            bboxes_per_img[:, 0::2] = bboxes_per_img[:, 0::2].clamp(min=0, max=img_shape[1])
            bboxes_per_img[:, 1::2] = bboxes_per_img[:, 1::2].clamp(min=0, max=img_shape[0])

            bboxes_per_img = torch.cat([bboxes_per_img, scores_per_img], dim=1)
            labels_per_img = labels_per_img.squeeze(-1)
            # TODO (i.rodin) Rethink it
            bboxes_with_labels = torch.cat([bboxes_per_img, labels_per_img.unsqueeze(-1)], dim=-1)
            dict_per_image = dict(image_width=img_metas[batch_i]['image_width'],
                                  image_height=img_metas[batch_i]['image_height'],
                                  image_name=img_metas[batch_i]['image_name'],
                                  bboxes=bboxes_with_labels)
            result_list.append(dict_per_image)

        return result_list

    def loss(self,
             pred_hm,
             pred_wh,
             gt_info):
        heatmap, box_target, wh_weight = self.ttf_target(gt_info)

        H, W = pred_hm.shape[2:]

        loss_cls = py_modified_sigmoid_focal_loss(pred_hm, heatmap) * self.hm_weight

        if self.base_loc is None or H != self.base_loc.shape[1] or W != \
                self.base_loc.shape[2]:
            base_step = self.down_ratio
            shifts_x = torch.arange(
                0, (W - 1) * base_step + 1,
                base_step,
                dtype=torch.float32,
                device=heatmap.device)
            shifts_y = torch.arange(
                0, (H - 1) * base_step + 1,
                base_step,
                dtype=torch.float32,
                device=heatmap.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            self.base_loc = torch.stack((shift_x, shift_y), dim=0)  # (2, h, w)

        # (batch, h, w, 4)
        if not self.wh_agnostic:
            box_target = box_target.view(-1, 4, H, W)
            pred_wh = pred_wh.view(-1, 4, H, W)
        pred_boxes = torch.cat((self.base_loc - pred_wh[:, [0, 1]],
                                self.base_loc + pred_wh[:, [2, 3]]),
                               dim=1).permute(0, 2, 3, 1)
        boxes = box_target.permute(0, 2, 3, 1)

        mask = wh_weight.view(-1, H, W)
        pos_mask_idx = mask > 0

        mask = mask[pos_mask_idx].float()
        pred_boxes = pred_boxes[pos_mask_idx].view(-1, 4)
        boxes = boxes[pos_mask_idx].view(-1, 4)
        loss_bbox = self.loss_bbox(
            pred_boxes, boxes, mask)

        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def _topk(self, scores, topk):
        batch, cat, height, width = scores.size()

        # (batch, 80, topk)
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), topk)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # (batch, topk). select topk from 80*topk
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), topk)
        topk_clses = (topk_ind // topk).int()
        topk_ind = topk_ind.unsqueeze(2)
        topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)
        topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind).view(batch, topk)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def gaussian_2d(self, shape, sigma_x=1, sigma_y=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius, k=1):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6

        gaussian = self.gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
        gaussian = heatmap.new_tensor(gaussian)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def ttf_target_single(self, gt_info, feat_shape):
        """

        Args:
            gt_info consists of:
            feat_shape: tuple.

        Returns:
            heatmap: tensor, tensor <=> img, (80, h, w).
            box_target: tensor, tensor <=> img, (4, h, w).
            reg_weight: tensor, same as box_target
        """
        output_h, output_w = feat_shape
        heatmap = gt_info['bboxes'].new_zeros((self.num_classes, output_h, output_w))
        beta_heatmap = gt_info['bboxes'].new_zeros((self.num_classes, output_h, output_w))
        fake_heatmap = gt_info['bboxes'].new_zeros((output_h, output_w))
        box_target = gt_info['bboxes'].new_ones((self.wh_planes, output_h, output_w)) * -1
        reg_weight = gt_info['bboxes'].new_zeros((self.wh_planes // 4, output_h, output_w))

        bboxes = gt_info['bboxes']
        bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
        bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]

        if len(bboxes) == 0:
            return heatmap, box_target, reg_weight

        boxes_areas_log = self.bbox_areas(bboxes).log()
        boxes_area_topk_log, boxes_ind = torch.topk(boxes_areas_log,
                                                    boxes_areas_log.size(0))

        gt_boxes = bboxes[boxes_ind, :4]
        gt_labels = bboxes[boxes_ind, -1]

        feat_gt_boxes = gt_boxes / self.down_ratio
        feat_gt_boxes[:, [0, 2]] = feat_gt_boxes[:, [0, 2]].clamp(0, output_w)
        feat_gt_boxes[:, [1, 3]] = feat_gt_boxes[:, [1, 3]].clamp(0, output_h)
        feat_hs, feat_ws = (feat_gt_boxes[:, 3] - feat_gt_boxes[:, 1],
                            feat_gt_boxes[:, 2] - feat_gt_boxes[:, 0])

        ct_ints = (torch.stack([(gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
                                (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2],
                               dim=1) / self.down_ratio).to(torch.int)

        h_radiuses_alpha = (feat_hs / 2. * self.alpha).int()
        w_radiuses_alpha = (feat_ws / 2. * self.alpha).int()
        if self.alpha != self.beta:
            h_radiuses_beta = (feat_hs / 2. * self.beta).int()
            w_radiuses_beta = (feat_ws / 2. * self.beta).int()

        # larger boxes have lower priority than small boxes.
        for k in range(boxes_ind.shape[0]):
            cls_id = gt_labels[k].long()

            fake_heatmap = fake_heatmap.zero_()
            self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                        h_radiuses_alpha[k].item(),
                                        w_radiuses_alpha[k].item())

            box_target_inds = fake_heatmap > heatmap[cls_id]
            heatmap[cls_id] = torch.max(heatmap[cls_id], fake_heatmap)

            if self.alpha != self.beta:
                fake_heatmap = fake_heatmap.zero_()
                self.draw_truncate_gaussian(fake_heatmap, ct_ints[k],
                                            h_radiuses_beta[k].item(),
                                            w_radiuses_beta[k].item())
                box_target_inds = fake_heatmap > beta_heatmap[cls_id]
                beta_heatmap[cls_id] = torch.max(beta_heatmap[cls_id], fake_heatmap)

            if self.wh_agnostic:
                box_target[:, box_target_inds] = gt_boxes[k][:, None]
                cls_id = 0
            else:
                box_target[(cls_id * 4):((cls_id + 1) * 4), box_target_inds] = gt_boxes[k][:, None]

            local_heatmap = fake_heatmap[box_target_inds]
            ct_div = local_heatmap.sum()
            local_heatmap *= boxes_area_topk_log[k]
            reg_weight[cls_id, box_target_inds] = local_heatmap / ct_div

        return heatmap, box_target, reg_weight

    def ttf_target(self, gt_info):
        with torch.no_grad():
            feat_shape = (int(gt_info[0]['image_height']) // self.down_ratio,
                          int(gt_info[0]['image_width']) // self.down_ratio)
            heatmap, box_target, reg_weight = multi_apply(
                self.ttf_target_single,
                gt_info,
                feat_shape=feat_shape)

            heatmap, box_target, reg_weight = [
                torch.stack(t, dim=0).detach()
                for t in [heatmap, box_target, reg_weight]
            ]

            return heatmap, box_target, reg_weight

    def simple_nms(self, heat, kernel=3, out_heat=None):
        pad = (kernel - 1) // 2
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = 1. - torch.ceil(hmax - heat)
        out_heat = heat if out_heat is None else out_heat
        return out_heat * keep

    def bbox_areas(self, bboxes, keep_axis=False):
        x_min, y_min, x_max, y_max = bboxes[:, 0], bboxes[:, 1], \
                                     bboxes[:, 2], bboxes[:, 3]
        areas = (y_max - y_min + 1) * (x_max - x_min + 1)
        if keep_axis:
            return areas[:, None]
        return areas
