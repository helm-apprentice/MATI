import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, query, key, value):
        # 在MultiheadAttention中，输出是 (output, attn_output_weights)
        # print(f"query: {query}")
        # print(f"key: {key}")
        # print(f"value: {value}")
        output, attn_weights = self.multihead_attn(query, key, value)
        return output, attn_weights

class LinearCombinationLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)