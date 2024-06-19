import math

import torch
import torch.nn.functional as F


def generate_bbox_mask(bbox_mask, bbox):
    b, h, w = bbox_mask.shape
    for i in range(b):
        bbox_i = bbox[i].cpu().tolist()
        bbox_mask[i, int(bbox_i[1]):int(bbox_i[1] + bbox_i[3] - 1), int(bbox_i[0]):int(bbox_i[0] + bbox_i[2] - 1)] = 1
    return bbox_mask

'''
这段代码是生成用于条件Batch Normalization的mask。
主要逻辑:
1. 获取模板图像大小template_size和特征图缩减步长stride。
2. 根据配置设置选择生成mask的方式:
   - ALL: 不生成mask,为None
   - CTR_POINT: 只保留中心1个点
   - CTR_REC: 保留中心区域
   - GT_BOX: 使用ground truth框生成mask
3. 对于CTR_POINT和CTR_REC,通过索引选取中心点或区域,生成0-1 mask。
4. 对于GT_BOX,先生成目标框mask,再缩放到特征图尺寸。
5. 最终mask大小为 (batch_size, feat_h * feat_w),内容为0或1。
这个mask的作用是选择特征图中只保留目标相关的区域参与Batch Normalization,从而避免背景区域对目标表示的干扰。
通过配置,可以选择不同的保留区域,来达到不同程度上减少背景对目标表征的影响。
整体上,这段代码生成了条件Batch Norm所需的二值化掩码,实现了只对感兴趣区域进行规范化的目的。
'''
def generate_mask_cond(cfg, bs, device, gt_bbox):
    template_size = cfg.DATA.TEMPLATE.SIZE  # 128
    stride = cfg.MODEL.BACKBONE.STRIDE  # 16
    template_feat_size = template_size // stride   # 8

    if cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'ALL':
        box_mask_z = None
    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_POINT':  # check！ 
        if template_feat_size == 8:
            index = slice(3, 4)
        elif template_feat_size == 12:
            index = slice(5, 6)
        elif template_feat_size == 7:
            index = slice(3, 4)
        elif template_feat_size == 14:
            index = slice(6, 7)
        else:
            raise NotImplementedError
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'CTR_REC':
        # use fixed 4x4 region, 3:5 for 8x8
        # use fixed 4x4 region 5:6 for 12x12
        if template_feat_size == 8:
            index = slice(3, 5)
        elif template_feat_size == 12:
            index = slice(5, 7)
        elif template_feat_size == 7:
            index = slice(3, 4)
        else:
            raise NotImplementedError
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)

    elif cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE == 'GT_BOX':
        box_mask_z = torch.zeros([bs, template_size, template_size], device=device)
        # box_mask_z_ori = data['template_seg'][0].view(-1, 1, *data['template_seg'].shape[2:])  # (batch, 1, 128, 128)
        box_mask_z = generate_bbox_mask(box_mask_z, gt_bbox * template_size).unsqueeze(1).to(
            torch.float)  # (batch, 1, 128, 128)
        # box_mask_z_vis = box_mask_z.cpu().numpy()
        box_mask_z = F.interpolate(box_mask_z, scale_factor=1. / cfg.MODEL.BACKBONE.STRIDE, mode='bilinear',
                                   align_corners=False)
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
        # box_mask_z_vis = box_mask_z[:, 0, ...].cpu().numpy()
        # gaussian_maps_vis = generate_heatmap(data['template_anno'], self.cfg.DATA.TEMPLATE.SIZE, self.cfg.MODEL.STRIDE)[0].cpu().numpy()
    else:
        raise NotImplementedError

    return box_mask_z


def adjust_keep_rate(epoch, warmup_epochs, total_epochs, ITERS_PER_EPOCH, base_keep_rate=0.5, max_keep_rate=1, iters=-1):
    if epoch < warmup_epochs:
        return 1
    if epoch >= total_epochs:
        return base_keep_rate
    if iters == -1:
        iters = epoch * ITERS_PER_EPOCH
    total_iters = ITERS_PER_EPOCH * (total_epochs - warmup_epochs)
    iters = iters - ITERS_PER_EPOCH * warmup_epochs
    keep_rate = base_keep_rate + (max_keep_rate - base_keep_rate) \
        * (math.cos(iters / total_iters * math.pi) + 1) * 0.5

    return keep_rate
"""
该函数用于根据当前epoch（轮次）和设定的warmup_epochs（预热轮次）、total_epochs（总轮次）以及ITERS_PER_EPOCH（每轮次的迭代次数）
    来计算当前的保留率（keep_rate）。

如果当前epoch小于warmup_epochs，即处于预热阶段，返回1，即保留所有数据。
如果当前epoch大于等于total_epochs，即已经完成所有轮次的训练，返回base_keep_rate，即基础保留率。
如果未指定iters参数，则将其计算为当前epoch乘以ITERS_PER_EPOCH。
计算总迭代次数total_iters，即从预热阶段结束后到全部轮次结束的总迭代次数。
将当前的iters减去预热阶段的迭代次数。
根据余弦函数的特性，计算当前的保留率keep_rate，其范围在base_keep_rate和max_keep_rate之间。
最后返回计算得到的keep_rate。
"""


if __name__ == '__main__':
    # 设置参数
    epoch = 69
    warmup_epochs = 20
    total_epochs = 70
    ITERS_PER_EPOCH = 1
    base_keep_rate = 0.7
    max_keep_rate = 1

    # 调用函数计算保持率
    keep_rate = adjust_keep_rate(epoch, warmup_epochs, total_epochs, ITERS_PER_EPOCH, base_keep_rate, max_keep_rate)

    print(f"Epoch {epoch}: Keep rate = {keep_rate}")