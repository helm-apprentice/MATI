"""
Basic ProContEXT model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.procontext.vit import vit_base_patch16_224
from lib.models.procontext.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.procontext.vit_cmc import vit_base_patch16_224_cmc
from lib.utils.box_ops import box_xyxy_to_cxcywh


class ProContEXT(nn.Module):
    """ This is the base class for ProContEXT """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer    # 是vit_cmc
        self.box_head = box_head   # head.py

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,   # 双模态的时候就是两个掩码
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        # print(f"return_cmc_attn: {return_last_attn}")
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        #print(f"cat_feature: {cat_feature.shape}")
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        #print(f"self.feat_len_s: {self.feat_len_s}")
        #print(f"enc_opt: {enc_opt.shape}")
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        #print(f"opt: {opt.shape}")
        bs, Nq, C, HW = opt.size()
        #print(f"bs: {bs}, Nq:{Nq}, C:{C}, HW:{HW}")
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map, score = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'score': score} # 多了1个score，用来筛选模版
            return out
        else:
            raise NotImplementedError
        
    def freeze_backbone(self):
        """冻结权重"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.box_head.parameters():
            param.requires_grad = False
    
    def unfreeze_cmc_block(self, has_infrared):
        """解冻CMCBlock权重"""
        for param in self.backbone.cross_modal_block.parameters():
            param.requires_grad = True
        if has_infrared:
            for param in self.backbone.blocks[4].parameters():
                param.requires_grad = True
            for param in self.backbone.blocks[8].parameters():
                param.requires_grad = True
            for param in self.backbone.blocks[12].parameters():
                param.requires_grad = True
            for param in self.backbone.blocks[3].parameters():
                param.requires_grad = True
            for param in self.backbone.blocks[7].parameters():
                param.requires_grad = True
            for param in self.backbone.blocks[11].parameters():
                param.requires_grad = True


"""
'pred_boxes'：这是预测的目标框坐标，表示为(cx, cy, w, h)，其中cx和cy是目标中心的坐标，w和h分别是目标的宽度和高度。这些坐标被标准化为相对于搜索图像大小的比例。
                pred_boxes是模型主要的输出，用于定位搜索图像中目标的位置。

'score_map'：这是目标中心点的预测得分图。对于每个像素点，得分图反映了该点作为目标中心的概率。
            这个得分图通常用于可视化目标的位置，并且在一些后处理步骤中，如非极大值抑制（NMS）中，用于提高目标定位的准确性。

'size_map'：这是预测的目标尺寸图，包含了目标的宽度和高度信息。每个像素点的值反映了如果该点是目标中心，预测目标的尺寸。
            尺寸图有助于理解目标的大小分布，并在结合中心点预测时，提供更完整的目标定位信息。

'offset_map'：这是预测的偏移图，对于每个像素点，提供了从当前位置到真实目标中心的偏移量。这个偏移量有助于校正基于离散像素点的预测，使得目标定位更加精确。

'score'：这是预测的目标得分，通常反映了模型对于其预测置信度的量化。这个得分可以用于排序或选择最终的目标框，尤其是在存在多个候选框时，选择置信度最高的框作为最终输出。

使用场景
目标定位与追踪：在视频监控、自动驾驶等应用中，'pred_boxes' 用于定位和追踪视频帧中的目标对象。
目标识别与分析：'score_map' 和 'score' 可用于分析目标的存在概率，辅助识别和分类任务。
尺寸估计：'size_map' 对于估计目标的实际尺寸非常有用，特别是在需要对目标大小进行量化分析的场景。
位置校正：'offset_map' 有助于从粗略的像素级预测中获得更精细的目标中心位置，对于提高目标检测的准确性至关重要。
"""

def build_procontext(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('ProContEXT' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_cmc':
        backbone = vit_base_patch16_224_cmc(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            cmc_loc=cfg.MODEL.BACKBONE.CMC_LOC,
                                            use_cmc=cfg.TEST.USE_CMC
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = ProContEXT(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )
    if cfg.TRAIN.FINETUNE:
        
        model.freeze_backbone()

        model.unfreeze_cmc_block(has_infrared=cfg.TRAIN.HAS_INFRARED)

    if 'ProContEXT' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

"""
这些输出是通过一系列卷积层和最终的预测层得到的，具体来说，是在模型的最后几层中，通过对特征图的处理和转换来实现的。下面是详细过程：

### `pred_boxes`的计算过程

1. **特征提取**：输入的搜索图像通过模型的前几层进行特征提取。
2. **中心预测层（Center Predictor）**：提取的特征图通过中心预测器，预测目标的中心点。这涉及到一系列的卷积操作，
                                    最终通过一个卷积层（`conv5_ctr`）输出一个单通道的`score_map`，表示目标中心的概率分布。
3. **大小（尺寸）预测层**：并行地，特征图也被送入另一系列卷积层（通过`conv5_size`），输出两个通道的`size_map`，分别代表预测目标的宽度和高度。
4. **偏移预测层**：同时，特征图通过另一组卷积层（通过`conv5_offset`），输出两个通道的`offset_map`，提供从每个像素点到目标中心的偏移量。
5. **综合计算**：利用中心点`score_map`、尺寸`size_map`和偏移量`offset_map`，结合模型的后处理算法（如软非极大值抑制或软argmax等），计算最终的目标框（`pred_boxes`）。

### `score_map`的得到

- 直接通过中心预测层的最终输出得到，代表每个像素点作为目标中心的置信度。

### `size_map`和`offset_map`的得到

- 分别通过大小预测层和偏移预测层的输出得到，它们描述了目标的尺寸和从每个像素点到目标中心的偏移量。

### `score`的计算

- 这是一个衡量模型预测置信度的标量值，可能是通过分析`score_map`得到的最大值，或是采用其他方法综合考虑`score_map`、`size_map`和`offset_map`的信息得到的。

### 计算过程的技术细节

- **卷积层**：各种预测层（中心、大小、偏移）使用了一系列卷积操作，其中包括标准的卷积（`nn.Conv2d`）、批量归一化（`nn.BatchNorm2d`）和激活函数（如`nn.ReLU`）。
            这些卷积层逐步减少通道数，直到最后的输出层，具体的通道数减少策略和卷积核大小等参数依模型具体设计而定。
- **特征图大小和步长**：特征图的大小（`feat_sz`）和模型的步长（`stride`）决定了最终输出的空间分辨率。
                    `feat_sz`影响`score_map`、`size_map`和`offset_map`的维度，而`stride`决定了特征图相对于原始图像的缩放比例。
- **最终输出的处理**：最终的目标框（`pred_boxes`）通过后处理步骤计算得到，可能包括应用softmax函数来规范化`score_map`，
                    并结合`size_map`和`offset_map`来计算每个目标的精确位置和尺寸。这一步可能涉及到软argmax等算法来从离散的输出中估算连续的目标位置。

通过这一系列的操作，模型能够从输入的搜索图像中预测目标的位置、大小和存在的置信度，为后续的目标跟踪或识别任务提供关键信息。
"""