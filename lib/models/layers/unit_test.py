# import torch

# # 假设我们有一个随机初始化的注意力权重张量，形状为 [batch_size, num_elements]
# # 这里我们用一个小的批量大小和元素数量来简化示例
# batch_size = 1
# num_elements = 5
# attn_t = torch.rand(batch_size, num_elements)

# # 打印原始注意力权重张量
# print("原始注意力权重张量:")
# print(attn_t)

# # 沿着第一个维度（dim=0）对注意力权重进行排序
# # descending=True 表示按降序排序
# sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

# # 打印排序后的张量和索引
# print("\n排序后的注意力权重张量:")
# print(sorted_attn)
# print("\n原始张量中元素的索引:")
# print(indices)

# topk_attn, topk_idx = sorted_attn[:, :3], indices[:, :3]
# print("\n排序筛选后的注意力权重张量:")
# print(topk_attn)
# print("\n排序筛选后原始张量中元素的索引:")
# print(topk_idx)
# global_index = torch.tensor([[0, 1, 2, 3, 4]])
# keep_index = global_index.gather(dim=1, index=topk_idx)
# print("\ngather后原始张量中元素的索引:")
# print(keep_index)

# attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))

import torch

# 假设我们有一个形状为 [2, 4] 的张量 x
x = torch.tensor([[1, 2, 3, 4],
                  [5, 6, 7, 8]])

# 假设我们有一些索引，我们想要将 x 中的元素根据这些索引进行重排
index_all = torch.tensor([[0, 1],
                          [2, 3]])

# 我们需要将 index_all 扩展到与 x 的最后一个维度大小相同
index_all_expanded = index_all.unsqueeze(-1).expand(2, 2, 4)

# 创建一个与 x 形状相同的全零张量
x_zero = torch.zeros_like(x)

# 使用 scatter_ 函数根据 index_all_expanded 重排 x 中的元素到 x_zero 中
x_zero.scatter_(dim=1, index=index_all_expanded, src=x)

print(x_zero)