import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
#from lib.models.layers.cust_attn import CustomAttentionLayer, LinearCombinationLayer
from .utils import combine_with_linear_transformation, combine_tokens, recover_tokens, combine_multi_tokens, weighted_average, union_masks
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, CMCBlock, CMCBlock1

import matplotlib.pyplot as plt
import numpy as np

_logger = logging.getLogger(__name__)


class VisionTransformerCMC(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module
        with CrossModalCompensationBlock

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None, cmc_loc=None,  use_cmc=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        # super().__init__()
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(     # image -> tokens
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        #====================================================================================================
        self.cross_modal_block = CMCBlock1(dim=embed_dim, num_heads=num_heads)
        #====================================================================================================

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        #self.ce_loc = ce_loc   # [3,6,9]
        self.use_cmc = use_cmc
        #print(f"vitcmc self.use_cmc: {self.use_cmc}")
        self.cmc_loc = cmc_loc
        ce_loc = [3, 6, 9]
            #depth = 15
        for i in range(depth):  # 0~ 11
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]   # CE_KEEP_RATIO: [0.8, 0.7, 0.7]
                ce_index += 1


            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )
            if ce_loc is not None and i in ce_loc:
                blocks.append(
                    CMCBlock(dim=embed_dim, num_heads=num_heads)
                )


        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False
                         ):
        ce_template_mask_combined = ce_template_mask
        # print(f"ce_template_mask: {ce_template_mask_combined}")
        # true_positions = ce_template_mask_combined.nonzero()
        # print("True values positions:\n", true_positions)

        if isinstance(ce_template_mask, list):
            # print(f"ce_visible_template_mask: {ce_template_mask[0].shape}")  # torch.Size([1, 144])
            # print(f"ce_infrared_template_mask: {ce_template_mask[1].shape}")  # torch.Size([1, 144])
            # #print(f"ce_template_mask[0]: {ce_template_mask[0]}")
            # #print(f"ce_template_mask[1]: {ce_template_mask[1]}")
            ce_template_mask_combined = ce_template_mask[0] | ce_template_mask[1]
            # print(f"ce_template_mask_combined: {ce_template_mask_combined.shape}")  # torch.Size([1, 144])
            # #print(f"ce_template_mask_combined: {ce_template_mask_combined}")
            # true_positions0 = ce_template_mask[0].nonzero()   
            # print("visible True values positions:\n", true_positions0)   # tensor([[ 0, 65]]
            # true_positions1 = ce_template_mask[1].nonzero()
            # print("infrared True values positions:\n", true_positions1)   # tensor([[ 0, 65]]
            # true_positions = ce_template_mask_combined.nonzero()
            # print("combine True values positions:\n", true_positions)  # tensor([[ 0, 65]]


        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        x += self.pos_embed_x   # pos_embed_x在base_backbone的finetune_track里定义
        # print(f"self.pos_embed_x: {self.pos_embed_x.shape}")   # torch.Size([1, 256, 768])
        #print(f"x.shape: {x.shape}")  # torch.Size([1, 256, 768])  搜索图像尺寸为256,patchsize为16，patches为256
        if not isinstance(z, list):
            
            z = self.patch_embed(z)
            z += self.pos_embed_z
            x = combine_tokens(z, x, mode=self.cat_mode)
            attn_cmc = None
        # ========================================================================================
        elif isinstance(z, list): # 多模态处理
            #print(f"z: {z}")
            # for i in range(len(z)):
            #     print(f"z{i}.shape: {z[i].shape}")
            # z_list = [self.patch_embed(zi) + self.pos_embed_z for zi in z]
            visible_features = self.patch_embed(z[0]) + self.pos_embed_z
            infrared_features = self.patch_embed(z[1]) + self.pos_embed_z
            if not return_last_attn:
                z = self.cross_modal_block(visible_features, infrared_features, return_attn=return_last_attn)
                #print("not return_last_attn")
            else:
                z, attn_cmc = self.cross_modal_block(visible_features, infrared_features, return_attn=return_last_attn)
                #print("return_last_attn")
            """ 跨模态补偿，得到增强的z"""
            #print(f"z.shape: {z.shape}")  # torch.Size([1, 144, 768])   模版尺寸为192,patchsize为16，patches为144
            x = combine_tokens(z, x, mode=self.cat_mode)   # torch.Size([1, 400, 768])
            #print(f"x.shape: {x.shape}")
            #print(f"z_list: {z_list}")
            # for i in range(len(z_list)):
            #     print(f"z_list{i}.shape: {z_list[i].shape}")


        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            # print("check")  # 无
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)
        # print(f"self.add_cls_token: {self.add_cls_token}")
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed
        # print(f"self.add_sep_seg: {self.add_sep_seg}")
        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)
        # print(f"x.shape: {x.shape}")  # torch.Size([1, 400, 768])
        lens_x = self.pos_embed_x.shape[1]  # 256
        # print(f"self.pos_embed_x.shape: {self.pos_embed_x.shape}")   # torch.Size([1, 256, 768])
        lens_z = self.pos_embed_z.shape[1]  # 144
        # print(f"self.pos_embed_z.shape: {self.pos_embed_z.shape}")   # torch.Size([1, 144, 768])
        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)
        # print(f"global_index_t: {global_index_t}")   # 0 - 143

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        # print(f"global_index_s: {global_index_s}")   # 0 - 255
        removed_indexes_s = []
        global_indexes_s = []
        attn_list = []
        hook_loc = self.cmc_loc  #[3, 7, 11]
        hook_loc = [3, 7, 11]  # 消除位置：没有中间插入双模块的是369，CMC后插入的时候是3711，前插入的时候是4812
        #print(f"use_cmc: {self.use_cmc}")
        for i, blk in enumerate(self.blocks):
            # print(f"i: {i}   blk: {blk}")
            #print(f"i: {i}")
            if self.use_cmc and blk.name == 'CMCBlock':
                #print("经过CMC")
                lens_t = global_index_t.shape[1]
                template_tokens = x[:, :lens_t]  # 分理出模版特征
                #print(f"{i}_template_tokens: {template_tokens.shape}")
                search_tokens = x[:, lens_t:]   # 分理出搜索特征
                #print(f"{i}_search_tokens: {search_tokens.shape}")
                template_tokens = blk(template_tokens, infrared_features) # 模版再补偿红外特征
                #print(f"{i}_(cmc)template_tokens: {template_tokens.shape}")
                x = torch.cat([template_tokens, search_tokens], dim=1)
                #print(f"{i}_{blk.name}_x.shape: {x.shape}")

            else:
                #print("不经过CMC")
                if blk.name and blk.name == 'CMCBlock': continue
                else:
                    #print(f"ce_keep_rate: {ce_keep_rate}")
                    x, global_index_t, global_index_s, removed_index_s, attn = \
                        blk(x, global_index_t, global_index_s, mask_x, ce_template_mask_combined, ce_keep_rate)
                    #print(f"{i}_{'CEB'}_x.shape: {x.shape}")
                    if return_last_attn:
                        attn_list.append(attn)
                    
                    if hook_loc is not None and i in hook_loc:
                        removed_indexes_s.append(removed_index_s)
                        global_indexes_s.append(global_index_s)
        # print(f"removed_indexes_s: {removed_indexes_s}")
        x = self.norm(x)
        #print(f"x.shape: {x.shape}")   # torch.Size([1, 233, 768])
        lens_x_new = global_index_s.shape[1]   # 89
        #print(f"global_index_s: {global_index_s.shape}")  # torch.Size([1, 89])
        lens_z_new = global_index_t.shape[1]   # 144
        #print(f"global_index_t: {global_index_t.shape}")  # torch.Size([1, 144])

        z = x[:, :lens_z_new]
        #print(f"z.shape: {z.shape}")  # torch.Size([1, 144, 768])
        x = x[:, lens_z_new:]
        #print(f"x1.shape: {x.shape}")  # torch.Size([1, 89, 768])

        if removed_indexes_s and removed_indexes_s[0] is not None:
            """通过添加零填充和重新排列索引的方式，模拟移除x张量中某些元素的效果，并试图恢复这些元素的原始顺序"""
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)  
            #print(f"removed_indexes_cat: {removed_indexes_cat.shape}")  # torch.Size([1, 167])
            pruned_lens_x = lens_x - lens_x_new   
            #print(f"pruned_lens_x: {pruned_lens_x}")    # 167 
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            #print(f"pad_x: {pad_x.shape}")   # torch.Size([1, 167, 768])
            x = torch.cat([x, pad_x], dim=1)
            #print(f"x: {x.shape}")  # torch.Size([1, 256, 768])
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            #print(f"index_all: {index_all.shape}")   # torch.Size([1, 256])
            # recover original token order
            C = x.shape[-1]
            #print(f"C: {C}")   # 768
            #print(f"index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64): {index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64).shape}")
            # index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64): torch.Size([1, 256, 768])
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
        #print(f"x2.shape: {x.shape}")  # torch.Size([1, 256, 768])
        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
        #print(f"x3.shape: {x.shape}")  # torch.Size([1, 256, 768])
        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)  
        #print(f"x4.shape: {x.shape}")  # torch.Size([1, 400, 768])
        # print(f"return_cmc_attn: {return_last_attn}")
        if return_last_attn:
            aux_dict = {
                "attn_cmc": attn_cmc,   # visdom in track
                "attn": attn_list,  # visdom in track
                "removed_indexes_s": removed_indexes_s,  # used for visualization
                "global_indexes_s": global_indexes_s
            }
            # print(f"removed_indexes_s1.size: {removed_indexes_s[0].shape}")  # 76  torch.Size([1, 76])
            # print(f"removed_indexes_s2.size: {removed_indexes_s[1].shape}")  # 54  torch.Size([1, 54])
            # print(f"removed_indexes_s2.size: {removed_indexes_s[2].shape}")  # 37  torch.Size([1, 37])
            # print(f"global_indexes_s[0].size: {global_indexes_s[0].shape}")  # 180  torch.Size([1, 180])
            # print(f"global_indexes_s[1].size: {global_indexes_s[1].shape}")  # 126  torch.Size([1, 126])
            # print(f"global_indexes_s[2].size: {global_indexes_s[2].shape}")  # 89  torch.Size([1, 89])
            """  训练时
            out_dict.attn[0]: torch.Size([32, 12, 400, 400]) 表示 CEBlock 的输出，其中：
            32 可能表示批处理大小（batch size）。
            12 表示注意力头的数量。
            400 x 400 表示序列长度的自注意力权重矩阵（假设序列长度为 400）。

            out_dict.attn_cmc: torch.Size([144, 32, 32]) 表示 CMCBlock 的输出，其中：
            144 可能是由于在跨模态注意力中合并了多个头的输出结果。例如，如果你有 12 个注意力头，每个头对应 12 个不同的特征表示，则可能会有 12 x 12 = 144 个这样的表示。
            32 x 32 表示对于每种特征表示，都有一个 32 x 32 的注意力权重矩阵。
            """
            '''  跟踪时
            attn_cmc.shape: torch.Size([144, 1, 1])
            attn[0].shape: torch.Size([1, 12, 400, 400])
            attn[11].shape: torch.Size([1, 12, 233, 233])
            '''
        else:
            aux_dict = {
                "attn": attn,
                "removed_indexes_s": removed_indexes_s,  # used for visualization
            }
        # print(f"attn.shape: {attn_list[11].shape}")
        # print(f"attn_cmc.shape: {attn_cmc.shape}")
        # attn_weights_sample = attn_cmc[:, 0, :].detach().cpu().numpy()  # 调整索引以匹配权重形状
        # attn_weights_sample = attn_list[11][0, 0, :].detach().cpu().numpy()
        # 使用matplotlib生成热图
        # plt.figure(figsize=(20, 16))
        # plt.imshow(attn_weights_sample, cmap='viridis', aspect='auto')
        # plt.colorbar()
        # plt.xlabel('Source Position')
        # plt.ylabel('Target Position')
        # plt.title('Attention Weights Heatmap')
        # plt.show()
        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):
    # 在ProContEXT——actor里的forward_pass()
        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate, return_last_attn=return_last_attn)
        #print(f"x.size: {x.size()}")
        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCMC(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            # print('checkpoint.keys: ', checkpoint.keys())
            key = "model" if "model" in checkpoint else 'net'
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint[key], strict=False)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224_cmc(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_cmc(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
