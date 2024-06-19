import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
# from lib.models.layers.cust_attn import CustomAttentionLayer, LinearCombinationLayer
from lib.models.layers.attn import Attention, CustomCrossModalAttention

import matplotlib.pyplot as plt
import numpy as np

"""
这个`candidate_elimination`函数的目的是在注意力机制中消除潜在的背景候选项，以减少计算量和噪声。这个过程通常用于跟踪任务中，通过消除与模板不相关的搜索区域补丁来提高跟踪的准确性。
以下是函数的主要步骤和它们的作用：
1. **计算模板注意力** (`attn_t`): 从完整的注意力矩阵中提取与模板相关的部分。
2. **应用模板掩码** (`box_mask_z`): 如果提供了模板掩码，它会被用来过滤注意力矩阵，只保留与模板相关的部分。
3. **计算平均注意力**: 对过滤后的模板注意力进行平均，以得到每个搜索补丁的总体注意力分数。
4. **排序和选择Top-K**: 对注意力分数进行排序，并选择前`lens_keep`个最高分数的补丁作为保留的候选补丁。
5. **获取保留和移除的索引**: 使用排序后的索引来获取保留的搜索补丁的全局索引和被移除的搜索补丁的全局索引。
6. **分离模板和搜索补丁**: 将模板补丁和搜索补丁分开。
7. **收集保留的搜索补丁**: 使用保留的索引从搜索补丁中收集保留的补丁。
8. **拼接补丁**: 将模板补丁和保留的搜索补丁拼接起来，形成新的补丁序列。
在这个过程中，被移除的搜索补丁是那些平均注意力分数较低的补丁，意味着它们与模板的相关性较低。
这些补丁在新的`tokens_new`张量中被排除，因此它们的相对位置确实发生了变化——它们不再出现在最终的补丁序列中。
需要注意的是，这个函数并没有改变搜索补丁之间的相对位置，而是简单地移除了一些补丁。保留的补丁（即Top-K注意力分数最高的补丁）保持了它们在原始搜索区域中的相对顺序。
这意味着，如果补丁A在原始搜索区域中位于补丁B的前面，并且在注意力分数排序后被保留，那么在`tokens_new`中，补丁A仍然会位于补丁B的前面。
"""
def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    #print(f"attn.shape: {attn.shape}")  # torch.Size([1, 12, 400, 400])->torch.Size([1, 12, 324, 324])->torch.Size([1, 12, 270, 270])->torch.Size([1, 12, 233, 233])
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t:]
    #print(f"attn_t.shape: {attn_t.shape}")  # torch.Size([1, 12, 144, 256])->torch.Size([1, 12, 144, 180])->torch.Size([1, 12, 144, 126])->torch.Size([1, 12, 144, 89])

    if box_mask_z is not None:
        #  print("check")   check!
        # for mask in box_mask_z:
        # box_mask_z_cat = torch.stack(box_mask_z, dim=1)
        # box_mask_z = box_mask_z_cat.flatten(1)
        # print(f"box_mask_z: {box_mask_z.shape}")    # torch.Size([1, 144])
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        #print(f"box_mask_z: {box_mask_z.shape}")   # torch.Size([1, 12, 144, 256])->([1, 12, 144, 180])->([1, 12, 144, 126])
        # attn_t = attn_t[:, :, box_mask_z, :]
        # box_mask_z = torch.cat((box_mask_z, box_mask_z, box_mask_z, box_mask_z), 2)


        #true_positions0 = box_mask_z.nonzero().size(0)  
        #print("visible True values positions:\n", true_positions0)   # 3072->2160->1512
        # 假设 box_mask_z 是一个形状为 [1, 12, 144, 256] 的布尔张量
        # 计算所有值为1的元素的数量
        #num_valid_elements = torch.sum(box_mask_z).item()

        # 打印有效元素的数量
        #print(num_valid_elements)   # # 3072->2160->1512

        attn_t = attn_t[box_mask_z]
        #print(f"attn_t1.shape: {attn_t.shape}")  # torch.Size([3072])->([2160])->([1512])
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        #print(f"attn_t2.shape: {attn_t.shape}")  # torch.Size([1, 12, 1, 256])->([1, 12, 1, 180])->([1, 12, 1, 126])
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s
        #print(f"attn_t3.shape: {attn_t.shape}")  # torch.Size([1, 256])->torch.Size([1, 180])->torch.Size([1, 126])

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)
    # print(f"sorted_attn_t.shape: {sorted_attn.shape}")  # torch.Size([1, 256])->([1, 180])->([1, 126])

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]
    # print(f"topk_idx: {topk_idx.dtype}")  # torch.Size([1, 180]) torch.int64
    # print(f"non_topk_idx: {non_topk_idx.dtype}")   # torch.Size([1, 76]) torch.int64
    # print(f"global_index: {global_index.dtype}")  # torch.float32
    # ===================================================================================
    """先将topk_idx在按升序排列之后，再使用gather，注意力权重相对顺序就不会改变"""
    topk_idx, _ = torch.sort(topk_idx, dim=1)
    # ===================================================================================

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    # print(f"keep_index: {keep_index.long().dtype}")   # torch.Size([1, 180]) torch.float32
    # print(f"removed_index: {removed_index.dtype}")  # torch.Size([1, 76]) torch.float32
    # assert torch.equal(topk_idx, keep_index.long()), "indices not same"
    # assert torch.equal(non_topk_idx, removed_index.long()), "indices not same"

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # attentive_tokens1 = tokens_s.gather(dim=1, index=keep_index.long().unsqueeze(-1).expand(B, -1, C))
    # assert torch.equal(attentive_tokens, attentive_tokens1), "attn not same"
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


class CEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
        super().__init__()
        self.name = 'CEBlock'
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.keep_ratio_search = keep_ratio_search

    def forward(self, x, global_index_template, global_index_search, mask=None, ce_template_mask=None, keep_ratio_search=None):
        #print("ceb begin")
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]

        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            #print(f"keep_ratio_search: {keep_ratio_search}")
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

'''
这两个模块类和上面的候选消除函数都是自定义的,它们是基于多头自注意力机制和多层感知机的变种,用于计算图像或文本等数据的特征表示。它们的功能是：

- CEBlock类是使用了相对位置编码机制和候选消除机制的注意力模块,它可以增强位置信息的影响,减少计算量和消除噪声。
    它需要提供输入数据、模板和搜索区域令牌的全局索引、掩码、模板掩码和搜索区域令牌的保留比例,返回输入数据、模板和搜索区域令牌的全局索引、
    移除的搜索区域令牌的全局索引和注意力权重矩阵。
- Block类是使用了随机深度丢弃机制的注意力模块,它可以增强模型的泛化能力和鲁棒性。它需要提供输入数据和掩码,返回输入数据的特征表示。
- candidate_elimination函数是用于实现候选消除机制的函数,它可以根据注意力权重矩阵,选择关注的搜索区域令牌,并消除不关注的搜索区域令牌。
    它需要提供注意力权重矩阵、输入数据、模板长度、搜索区域令牌的保留比例、搜索区域令牌的全局索引和模板掩码,返回消除后的输入数据、
    保留的搜索区域令牌的全局索引和移除的搜索区域令牌的全局索引。
'''

class CMCBlock(nn.Module):
    ''' CrossModalCompensationBlock'''
    def __init__(self, dim, num_heads):
        super().__init__()
        self.name = 'CMCBlock'
        self.query_proj = nn.Linear(dim, dim)
        self.key_value_proj = nn.Linear(dim, dim)
        self.cmc_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        #self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)   # 这行暂时修改，名字切记还原
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, visible_features, infrared_features, return_attn=False):
        #print("cmc begin")
        # Transform infrared features to query
        query = self.query_proj(infrared_features)
        # Use visible features as key and value
        key_value = self.key_value_proj(visible_features)
        
        # Apply cross-modal attention
        attn_output, attn_weights = self.cmc_attn(query=query, key=key_value, value=key_value)
        #attn_output, attn_weights = self.attn(query=query, key=key_value, value=key_value) # 这行暂时修改，名字切记还原
        # Apply feed-forward network
        output = self.feed_forward(attn_output)
        
        # Normalize and add the residual
        output = self.norm(visible_features + output)
        # print(f"visible_features.shape: {visible_features.shape}")
        # print(f"infrared_features.shape: {infrared_features.shape}")
        # print(f"output_features.shape: {output.shape}")
        """  训练时
        visible_features.shape: torch.Size([32, 144, 768])
        infrared_features.shape: torch.Size([32, 144, 768])
        output_features.shape: torch.Size([32, 144, 768])
        """
        """  跟踪时
        visible_features.shape: torch.Size([1, 144, 768])
        infrared_features.shape: torch.Size([1, 144, 768])
        output_features.shape: torch.Size([1, 144, 768])
        """
        if return_attn:
            # print(f"attn_weights.shape: {attn_weights.shape}")
            """训练时
            attn_weights.shape: torch.Size([144, 32, 32])
            """
            """跟踪时
            attn_cmc.shape: torch.Size([144, 1, 1])
            """
            return output, attn_weights
        # attn_weights_sample = attn_weights[:, 0, :].detach().cpu().numpy()  # 调整索引以匹配权重形状
        return output
'''
在`CrossModalCompensationBlock`中，输出的`enhanced_features`的形状设计上是与输入的`visible_features`形状一致的。
这是因为在该模块的最后，通过层归一化（LayerNorm）和残差连接将加强特征（即，通过注意力机制和前馈网络处理后的特征）与原始的可见光特征相加。
这种设计模式确保了输出特征与输入特征在形状上的一致性，允许它们在模型的后续部分无缝使用。

具体来说，`output = self.norm(visible_features + output)`这一行将通过注意力机制加强的特征（`output`）与原始的可见光特征（`visible_features`）进行相加，然后应用层归一化。
因此，最终的`output`（即`enhanced_features`）将与`visible_features`有相同的维度和形状。

对于`infrared_features`，虽然它们在注意力机制中被用作查询（query）来引导信息融合，但在`CrossModalCompensationBlock`的输出中并不直接包含处理后的红外特征。
红外特征主要用于加强可见光特征，在最终输出的`enhanced_features`中，红外信息已经被融合进去，从而实现了跨模态补偿。

因此，如果你的模型需要同时处理和利用`enhanced_features`、`visible_features`和`infrared_features`，需要注意的是：

- `enhanced_features`将直接用于后续处理，因为它已经包含了来自红外模态的补偿信息。
- 原始的`visible_features`和`infrared_features`可以根据需要用于模型的其他部分，例如进行额外的融合、比较或分析。

在实现时，确保正确地管理这些特征的流动和使用，以最大化跨模态信息融合的效果。
'''

class EnhancedCrossModalCompensationBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # 自定义注意力层，用于跨模态信息融合
        self.custom_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        # 线性组合层，用于调整融合后的特征
        self.linear_combination = nn.Linear(input_dim=dim * 2, output_dim=dim)
        # 前馈网络，用于进一步处理融合特征
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim)
        # 用于计算加权平均的额外注意力层
        self.weighted_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, infrared_features, visible_features):
        # 使用自定义注意力层融合红外和可见光特征
        attn_output, attn_weights = self.custom_attn(infrared_features, visible_features, visible_features)
        # 应用加权平均，利用注意力权重
        weighted_avg, _ = self.weighted_attn(attn_output, attn_output, attn_output)
        # 将原始的可见光特征和加权平均后的特征进行线性组合
        combined_feature = torch.cat([visible_features, weighted_avg], dim=-1)
        combined_feature = self.linear_combination(combined_feature)
        # 通过前馈网络进一步处理融合后的特征
        output = self.feed_forward(combined_feature)
        # 应用归一化和残差连接
        output = self.norm(visible_features + output)
        
        return output
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class CMCBlock1(nn.Module):
    """CrossModalCompensationBlock with Custom Cross-Modal Attention Mechanism."""
    def __init__(self, dim, num_heads, max_len=500):
        super().__init__()
        # Replace the nn.MultiheadAttention with our CustomCrossModalAttention
        self.cross_modal_attn = CustomCrossModalAttention(dim, num_heads, max_len=max_len)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, visible_features, infrared_features, return_attn=False):
        # Process features through the custom cross-modal attention mechanism
        if return_attn:
            attn_output, attn_weights = self.cross_modal_attn(visible_features, infrared_features, return_attn=return_attn)
        else:
            attn_output = self.cross_modal_attn(visible_features, infrared_features, return_attn=return_attn)
        
        
        # Apply feed-forward network on top of attention output
        output = self.feed_forward(attn_output)
        
        # Normalize and add the residual
        output = self.norm(visible_features + output)
        
        if return_attn:
            return output, attn_weights
        return output
