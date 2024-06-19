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
from lib.models.layers.cust_attn import CustomAttentionLayer, LinearCombinationLayer
from .utils import combine_with_linear_transformation, combine_tokens, recover_tokens, combine_multi_tokens, weighted_average, union_masks
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock

_logger = logging.getLogger(__name__)


class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=None, ce_keep_ratio=None):
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
        self.custom_attention_layer = CustomAttentionLayer(d_model = embed_dim, nhead = 8)
        self.linear_combination_layer = LinearCombinationLayer(input_dim = embed_dim, output_dim = embed_dim)
        #====================================================================================================

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc   # [3,6,9]
        for i in range(depth):  # i 是 0-11
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]   # CE_KEEP_RATIO: [0.7, 0.7, 0.7]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False
                         ):
        """
        procontext.py/procontext   line40
        """
        '''ce_template_mask是使用可见模版的原始掩码，还是可见和红外融合之后的并集掩码？？？'''
        '''
        def compute_mask_from_features(fused_features, threshold=0.5):
            """基于融合特征计算掩码。
            
            Args:
                fused_features (torch.Tensor): 融合特征张量，形状为[C, H, W]。
                threshold (float): 用于生成掩码的阈值。
            
            Returns:
                torch.Tensor: 从融合特征计算出的掩码。
            """
            # 假设我们通过取通道的平均值来简化特征表示
            avg_features = torch.mean(fused_features, dim=0)
            
            # 应用阈值生成掩码
            mask = avg_features > threshold
            return mask
        '''
        if isinstance(ce_template_mask, list):
            assert len(ce_template_mask) == 2
            ce_template_mask = union_masks(ce_template_mask)
        
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        x += self.pos_embed_x
        if not isinstance(z, list):
            # print(f"z: {z}")
            # # 打印z的类型
            # print("Type of z:", type(z))

            # # 如果z有可能是一个自定义或复杂的对象，尝试打印一些额外的信息
            # if hasattr(z, '__dict__'):
            #     print("Attributes of z:", z.__dict__)
            # if hasattr(z, 'shape'):
            #     print("Shape of z:", z.shape)
            # if hasattr(z, 'dtype'):
            #     print("Data type of z:", z.dtype)

            z = self.patch_embed(z)
            z += self.pos_embed_z
            lens_z = self.pos_embed_z.shape[1]
            x = combine_tokens(z, x, mode=self.cat_mode)
        # ========================================================================================
        elif isinstance(z, list): # 多模态处理
            #print(f"z: {z}")
            for i in range(len(z)):
                print(f"z{i}.shape: {z[i].shape}")
            z_list = [self.patch_embed(zi) + self.pos_embed_z for zi in z]
            #print(f"z_list: {z_list}")
            for i in range(len(z_list)):
                print(f"z_list{i}.shape: {z_list[i].shape}")
        # --------------------------------------------------------------------
            '''  自注意力   '''
            z = torch.cat(z_list, dim=1)
            #print(f"z_cat: {z}")
            print(f"z_cat.shape: {z.shape}")
            z = z.transpose(0, 1)
            print(f"z_cat_t.shape: {z.shape}")
            z, weights = self.custom_attention_layer(z, z, z)
            #print(f"z_layer: {z}")


            print(f"z_layer.shape: {z.shape}")
            #print(f"weights: {weights}")
            print(f"weights.shape: {weights.shape}")
            # z_mean = z.mean(dim=0)
            #print(f"z_mean: {z_mean}")
            # print(f"z_mean.shape: {z_mean.shape}")
            # --------------------------------------------------------------------
            '''   交叉注意力   '''
            visible_features = self.patch_embed(z[0]) + self.pos_embed_z
            infrared_features = self.patch_embed(z[1]) + self.pos_embed_z
            z, weights = self.custom_attention_layer(visible_features, infrared_features, infrared_features)
            # --------------------------------------------------------------------
            z = weighted_average(z, weights)
            lens_z = self.pos_embed_z.shape[1]
            print(f"x shape: {x.shape}")
            x = combine_with_linear_transformation(z, x, self.linear_combination_layer)

            """
在 PyTorch 中,nn.MultiheadAttention 需要的输入形状通常是 [序列长度, 批次大小, 特征维度]

1. `z0.shape` 和 `z1.shape`: 这两个张量代表原始输入图像的形状,每个都是 `[32, 3, 192, 192]`。这表示你有 32 个样本,每个样本有 3 个通道,大小为 192x192。
2. `z_list0.shape` 和 `z_list1.shape`: 在应用了 `self.patch_embed` 和 `self.pos_embed_z` 后,每个输入张量的形状变为 `[32, 144, 768]`。
    这表明每个样本被转换成了一个 144 个元素的序列,每个元素是一个 768 维的嵌入向量。
3. `z_cat.shape`: 使用 `torch.cat(z_list, dim=1)` 后,这个张量的形状是 `[32, 288, 768]`。这意味着你将两个输入的序列(每个长度为 144)拼接起来,形成一个长度为 288 的序列。
4. `z_cat_t.shape`: 在对 `z` 进行转置后,形状变为 `[288, 32, 768]`,这是 `nn.MultiheadAttention` 所期望的输入形状,其中序列长度是 288,批次大小是 32,嵌入维度是 768。
5. `z_layer.shape` 和 `weights.shape`: 经过自定义的注意力层后,输出 `z_layer` 保持形状 `[288, 32, 768]`,而注意力权重 `weights` 的形状是 `[32, 288, 288]`。
    这表明对于每个样本(32个),你有一个 288x288 的注意力矩阵,显示了 288 个序列元素之间的注意力关系。
6. `z_mean.shape`: 对 `z_layer` 在序列维度(dim=0)上求平均后,形状变为 `[32, 768]`。这意味着你为每个样本得到了一个 768 维的向量,这是序列的平均表示。
7. `Features shape` 和 `Weights shape`: 这两个形状分别是 `[32, 768]` 和 `[32, 288, 288]`,现在用于计算加权平均。

    通常 weights 或 attn_output_weights 应该有形状 (batch_size, seq_len, seq_len) 或 (batch_size*num_heads, seq_len, seq_len)。
            """
        # ========================================================================================
        # else:
        #     z_list = []
        #     for zi in z:
        #         z_list.append(self.patch_embed(zi) + self.pos_embed_z)
        #     lens_z = self.pos_embed_z.shape[1] * len(z_list)
        #     x = combine_multi_tokens(z_list, x, mode=self.cat_mode)
        '''
多模板处理:这个版本增加了对多模板输入的支持。它首先检查模板 (z) 是否是一个列表。
如果是,表示有多个模板输入,对每个模板都进行独立的分块嵌入。

特征组合方式:使用 combine_multi_tokens 方法来处理多模板特征的组合。
这种方法更为复杂和灵活,允许多个模板特征与搜索区域特征的有效合并。

注意力掩码处理:与第一个版本类似,但是在处理多模板的情况下,注意力掩码的处理可能更为复杂。

Transformer层: 同样通过一系列Transformer层处理合并后的特征,这些层也可能包含候选消除(CE)模块。

多模板支持:第二个版本相对于第一个版本最大的区别在于它能够处理多个模板输入。
这在处理涉及多个模板(例如,多视角或多时刻的目标特征)的场景中非常有用。

特征组合机制:两个版本在特征组合上采用了不同的方法。
第二个版本的 combine_multi_tokens 提供了更高的灵活性,能够更好地处理多模板情况下的特征融合。
        '''

        # attention mask handling
        # B, H, W
        print(f"mask_z: {mask_z}")
        print(f"mask_x: {mask_x}")
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)
        print(f"self.add_cls_token: {self.add_cls_token}")
        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed
        print(f"self.add_sep_seg: {self.add_sep_seg}")
        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        removed_indexes_s = []
        for i, blk in enumerate(self.blocks):
            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            # x = x.gather(1, index_all.unsqueeze(-1).expand(B, -1, C).argsort(1))
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):
    # 在ProContEXT——actor里的forward_pass()
        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

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


def vit_base_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
