from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class ProContEXTActor(BaseActor):
    """ Actor for training ProContEXT models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        #print(f"self.cfg.TRAIN.FINETUNE: {self.cfg.TRAIN.FINETUNE}")
        return_cmc_attn = True if self.cfg.TRAIN.FINETUNE else False
        # forward pass
        out_dict = self.forward_pass(data ,return_cmc_attn)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status, out_dict

    def forward_pass(self, data, return_cmc_attn=False):
        # 这里data是一个批次里堆叠好的
        # print(f'forward_pass(data): {data}')
        image_types = ['visible', 'infrared'] if data['use_infrared'].all().item() else ['visible']
        # use_infrared = True if data['use_infrared'].all().item() else False
        # 如果至少有一个元素是0，返回False
        # currently only support 1 search region
        assert len(data['search_images']) == 1
        template_list = []
        for img_type in image_types:
            assert len(data['template_images'][img_type]) == 1

            for i in range(len(data['template_images'][img_type])):
                template_img_i = data['template_images'][img_type][i].view(-1,
                                                                *data['template_images'][img_type].shape[2:])  # (batch, 3, 128, 128)
                # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
                template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            if 'template_anno' in data:
                for img_type in image_types:
                    for i in range(len(data['template_images'][img_type])):
                        template_img_i = data['template_images'][img_type][i].view(-1,
                                                                *data['template_images'][img_type].shape[2:])  # (batch, 3, 128, 128)
                        mask_i = generate_mask_cond(self.cfg, template_img_i.shape[0], template_img_i.device,
                                                        data['template_anno'][img_type][i])
                        box_mask_z.append(mask_i)

                    ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
                    ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
                    ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                        total_epochs=ce_start_epoch + ce_warm_epoch,
                                                        ITERS_PER_EPOCH=1,
                                                        base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]
        
        if len(box_mask_z) == 1:
            box_mask_z = box_mask_z[0]

        # net 是 ProContEXT()
        #print(f"return_cmc_attn: {return_cmc_attn}")
        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,   # 双模态的时候就是两个掩码
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=return_cmc_attn,
                            )
        #print(f"return_cmc_attn: {return_cmc_attn}")
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # print('compute losses ...')
        # print(f'pred_dict: {pred_dict.keys()}\ngt_dict: {gt_dict}\n')
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss


"""
损失的计算是模型训练过程中的关键步骤，用于衡量模型的预测与实际标签之间的差异。在你提供的代码中，损失是通过结合几个不同的损失函数来计算的，具体包括GIoU损失、L1损失和位置（定位）损失。
    以下是这些损失的详细解释及其作用：

### GIoU 损失 (Generalized Intersection over Union Loss)
- **定义**：GIoU损失是IoU损失的一个扩展，它解决了IoU在某些情况下（如没有交集时）无法提供有用梯度的问题。
            GIoU在原有IoU的基础上加入了一个衡量边界框和真实框之间相对位置的项，从而即使在没有交集的情况下也能提供有效的梯度。
- **用途**：GIoU损失用于更精确地衡量预测框和真实框之间的差异，特别是它们的形状和位置。通过最小化GIoU损失，模型学习到如何更准确地预测目标的位置和大小。

### L1 损失
- **定义**：L1损失是预测值和真实值差的绝对值，是一种常见的回归损失函数。
- **用途**：在目标检测任务中，L1损失通常用于目标框的坐标预测，帮助模型准确预测目标的具体位置。它对异常值（outliers）更加鲁棒。

### 位置（定位）损失
- **定义**：位置损失是通过比较预测的目标中心点得分图和真实的高斯热图来计算的。
            在这段代码中，使用了焦点损失（Focal Loss）来计算位置损失，焦点损失是交叉熵损失的一个变种，设计用来解决类别不平衡问题。
- **用途**：位置损失用于指导模型学习如何准确地定位目标的中心点，尤其是在目标稀疏和背景复杂的情况下。通过焦点损失，模型能够更关注那些难以分类的样本，提高定位的准确性。

### 损失的综合与权重
在你的代码中，总损失是这些单独损失的加权和，权重由`loss_weight`字典提供。这种加权方法允许模型同时优化多个目标，如框的位置、形状和大小，以及目标的中心定位。

### 损失指标的用途
- **训练优化**：损失指标直接用于训练过程中的反向传播，指导模型参数的更新。
- **性能评估**：损失值和IoU提供了评估模型性能的手段，可以用来监控训练过程，判断模型是否在学习，并且在哪些方面需要改进。
- **调参依据**：通过分析不同损失的大小，可以对模型进行调参，例如调整损失权重，以平衡不同任务的优化目标，或是调整模型结构和学习率等。

综合来看，损失计算不仅是模型训练的核心，也是模型性能优化和评估的基础。通过精心设计和加权不同的损失函数，可以有效地指导模型学习到

更加准确和鲁棒的目标检测能力。
"""