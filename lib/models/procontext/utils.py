import math

import torch
import torch.nn.functional as F
import torch.nn as nn


def union_masks(mask_list):
    """取掩码列表的并集。
    
    Args:
        mask_list (list of torch.Tensor): 包含两个二维掩码张量的列表。
    
    Returns:
        torch.Tensor: 两个掩码的并集。
    """
    # 使用torch.logical_or来获取两个掩码的并集
    union_mask = torch.logical_or(mask_list[0], mask_list[1])
    return union_mask

def combine_tokens(template_tokens, search_tokens, mode='direct', return_res=False):
    if mode == 'direct':
        merged_feature = torch.cat((template_tokens, search_tokens), dim=1)
    else:
        raise NotImplementedError

    return merged_feature

def combine_tokens3(template_tokens, dynamic_tokens, search_tokens, mode='direct', return_res=False):
    # [B, HW, C]
    len_t = template_tokens.shape[1]
    len_s = search_tokens.shape[1]

    if mode == 'direct':
        merged_feature = torch.cat((template_tokens, dynamic_tokens, search_tokens), dim=1)
    elif mode == 'template_central':
        central_pivot = len_s // 2
        first_half = search_tokens[:, :central_pivot, :]
        second_half = search_tokens[:, central_pivot:, :]
        merged_feature = torch.cat((first_half, template_tokens, second_half), dim=1)
    elif mode == 'partition':
        feat_size_s = int(math.sqrt(len_s))
        feat_size_t = int(math.sqrt(len_t))
        window_size = math.ceil(feat_size_t / 2.)
        # pad feature maps to multiples of window size
        B, _, C = template_tokens.shape
        H = W = feat_size_t
        template_tokens = template_tokens.view(B, H, W, C)
        pad_l = pad_b = pad_r = 0
        # pad_r = (window_size - W % window_size) % window_size
        pad_t = (window_size - H % window_size) % window_size
        template_tokens = F.pad(template_tokens, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, Hp // window_size, window_size, W, C)
        template_tokens = torch.cat([template_tokens[:, 0, ...], template_tokens[:, 1, ...]], dim=2)
        _, Hc, Wc, _ = template_tokens.shape
        template_tokens = template_tokens.view(B, -1, C)
        merged_feature = torch.cat([template_tokens, search_tokens], dim=1)

        # calculate new h and w, which may be useful for SwinT or others
        merged_h, merged_w = feat_size_s + Hc, feat_size_s
        if return_res:
            return merged_feature, merged_h, merged_w

    else:
        raise NotImplementedError

    return merged_feature


def combine_multi_tokens(template_tokens, search_tokens, mode='direct'):
    if mode == 'direct':
        if not isinstance(template_tokens, list):
            merged_feature = torch.cat((template_tokens, search_tokens), dim=1)
        elif len(template_tokens) >= 2:
            merged_feature = torch.cat((template_tokens[0], template_tokens[1]), dim=1)
            for i in range(2, len(template_tokens)):
                merged_feature = torch.cat((merged_feature, template_tokens[i]), dim=1)
            merged_feature = torch.cat((merged_feature, search_tokens), dim=1)
        else:
            merged_feature = torch.cat((template_tokens[0], template_tokens[1]), dim=1)
    else:
        raise NotImplementedError    
    return merged_feature
'''
这个函数 `combine_multi_tokens` 的作用是将模板特征 (`template_tokens`) 和搜索特征 (`search_tokens`) 
结合起来。这在目标跟踪或相关的深度学习任务中很常见，通常用于将目标的不同视角或状态的特征与当前搜索区域的特征结合起来。
以下是函数的详细分析：

### 参数
- `template_tokens`: 表示模板特征的张量或张量列表。
- `search_tokens`: 表示搜索区域特征的张量。
- `mode`: 合并模式，默认为 'direct'。

### 功能逻辑
- **`mode == 'direct'`**：
  - 如果 `template_tokens` 不是列表（即单个张量），
        则直接将其与 `search_tokens` 沿着第二维（`dim=1`）拼接起来。
  - 如果 `template_tokens` 是一个列表并且长度大于或等于2，
        函数会先将列表中的前两个模板特征沿着第二维拼接起来，然后遍历列表中剩余的特征并继续拼接，
        最后将这个合并后的特征与 `search_tokens` 拼接。
  - 如果 `template_tokens` 是一个列表但长度小于2（即长度为1），
        则直接将这个唯一的模板特征与 `search_tokens` 拼接。

- **其他 `mode`**：
  - 目前函数中只实现了 `direct` 模式，如果传入其他 `mode` 值，将会触发 `NotImplementedError` 异常。

### 返回值
- 返回拼接后的特征张量 `merged_feature`。

### 应用场景
在目标跟踪中，通常需要将目标的历史视角（模板特征）与当前搜索区域的特征结合起来，
以便网络可以更好地理解目标的当前状态和位置。这个函数就是实现这种特征合并的一种方法。

### 注意事项
- 需要确保 `template_tokens` 和 `search_tokens` 的其他维度（除了第二维）是相同的，以便它们可以被正确地拼接。
- 当处理模板特征列表时，函数假设所有特征张量的尺寸都是兼容的。如果列表中的特征张量尺寸不同，直接拼接可能会引发错误。
'''


def combine_multi_tokens_att(template_tokens, search_tokens, mode='direct', use_attention=True):
    if mode == 'direct':
        if use_attention:
            # 如果使用注意力机制
            if not isinstance(template_tokens, list):
                template_features = template_tokens.unsqueeze(0)
            else:
                # 将模板特征堆叠成一个新的维度
                template_features = torch.stack(template_tokens, dim=0)

            # 计算注意力权重
            attention_weights = F.softmax(template_features.mean(dim=2), dim=0)
            
            # 使用加权和融合模板特征
            merged_template_feature = torch.sum(attention_weights * template_features, dim=0)

            # 将搜索特征与融合后的模板特征拼接
            merged_feature = torch.cat((merged_template_feature, search_tokens), dim=1)
        else:
            # 不使用注意力机制，直接拼接
            if not isinstance(template_tokens, list):
                merged_feature = torch.cat((template_tokens, search_tokens), dim=1)
            else:
                merged_feature = torch.cat(template_tokens + [search_tokens], dim=1)
    else:
        raise NotImplementedError    
    return merged_feature

# ==========================================================================================

# class CustomAttentionLayer(nn.Module):
#     def __init__(self, d_model, nhead):
#         super().__init__()
#         self.multihead_attn = nn.MultiheadAttention(d_model, nhead)

#     def forward(self, query, key, value):
#         return self.multihead_attn(query, key, value)[0]

# def combine_multi_tokens_vit_att(template_tokens, search_tokens, attention_layer, mode='direct'):

#     if mode == 'direct':
#         if template_tokens.dim() >= 2:

#             # 计算自注意力
#             template_features = attention_layer(template_tokens, template_tokens, template_tokens)
            
#             # 平均融合模板特征
#             merged_template_feature = template_features.mean(dim=0)
#         else:
#             merged_template_feature = template_tokens

#         # 将搜索特征与融合后的模板特征拼接
#         merged_feature = torch.cat((merged_template_feature, search_tokens), dim=1)
#     else:
#         raise NotImplementedError    

#     return merged_feature


def weighted_average(features, weights):
    print("Features shape:", features.shape)
    print("Weights shape:", weights.shape)
    # 聚合注意力权重
    weights_aggregated = weights.mean(dim=1)   # 实验对比 sum 的效果
    print("weights_aggregated shape: ", weights_aggregated.shape)
    # weights_aggregated = weights.sum(dim=1, keepdim=True)
    # print("weights_aggregated_sum shape: ", weights_aggregated.shape)
    weights_aggregated = weights_aggregated.unsqueeze(-1) # 添加维度以进行广播
    print("weights_aggregated unsqueeze shape: ", weights_aggregated.shape)
    # 将特征转置以匹配权重的形状
    features_transposed = features.transpose(0, 1)  # [32, 288, 768]
    print("Features transposed shape: ", features_transposed.shape)
    # 应用加权平均
    weighted_features = features_transposed * weights_aggregated
    print("weughted_features shape: ", weighted_features.shape)
    # 计算加权平均
    weight_average = weighted_features.sum(dim=1) / weights_aggregated.sum(dim=1, keepdim = True)
    """
    keepdim=True 用于在求和操作后保持原有的维度数，这在进行后续的广播乘法时很有用。
    """
    print("weight_average shape: ", weight_average.shape)
    return weight_average

# def weighted_average(features, weights):
#     print("Features shape:", features.shape)
#     print("Weights shape:", weights.shape)

#     # 聚合注意力权重，可以选择平均或求和
#     weights_aggregated = weights.sum(dim=1, keepdim=True)
#     print("weights_aggregated shape: ", weights_aggregated.shape)

#     # 应用加权平均
#     weighted_features = features * weights_aggregated
#     print("weughted_features shape: ", weighted_features.shape)

#     # 计算加权平均
#     weighted_average = weighted_features.sum(dim=1) / weights_aggregated.sum(dim=1)
#     print("weighted_average shape: ", weighted_average.shape)

#     return weighted_average


# 假设我们要将两个特征合并后通过一个线性层
def combine_with_linear_transformation(feature1, feature2, linear_layer):
    combined_feature = torch.cat([feature1, feature2], dim=1)
    return linear_layer(combined_feature)
#=========================================================================================================

# recover_tokens函数可以根据不同的拼接模式,将特征序列恢复为模板区域和搜索区域的小块嵌入,用于输出结果或进行窗口划分(window partitioning)。
def recover_tokens(merged_tokens, len_template_token, len_search_token, mode='direct'):
    if mode == 'direct':
        recovered_tokens = merged_tokens
    elif mode == 'template_central':
        central_pivot = len_search_token // 2
        len_remain = len_search_token - central_pivot
        len_half_and_t = central_pivot + len_template_token

        first_half = merged_tokens[:, :central_pivot, :]
        second_half = merged_tokens[:, -len_remain:, :]
        template_tokens = merged_tokens[:, central_pivot:len_half_and_t, :]

        recovered_tokens = torch.cat((template_tokens, first_half, second_half), dim=1)
    elif mode == 'partition':
        recovered_tokens = merged_tokens
    else:
        raise NotImplementedError

    return recovered_tokens


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
