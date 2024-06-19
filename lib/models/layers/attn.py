import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe =rpe
        if self.rpe:  # false
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        else:
            return x


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Attention_st(nn.Module):
    def __init__(self, dim, mode, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # NOTE: Small scale for attention map normalization

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        # x: B, N, C
        # mask: [B, N, ] torch.bool
        B, N, C = x.shape
        
        lens_z = 64  # Number of template tokens
        lens_x = 256  # Number of search region tokens
        if self.mode == 's2t':  # Search to template
            q = x[:, :lens_z]  # B, lens_z, C
            k = x[:, lens_z:]  # B, lens_x, C
            v = x[:, lens_z:]  # B, lens_x, C
        elif self.mode == 't2s':  # Template to search
            q = x[:, lens_z:]  # B, lens_x, C
            k = x[:, :lens_z]  # B, lens_z, C
            v = x[:, :lens_z]  # B, lens_z, C
        elif self.mode=='t2t':  # Template to template
            q = x[:, :lens_z]  # B, lens_z, C
            k = x[:, lens_z:]  # B, lens_z, C
            v = x[:, lens_z:]  # B, lens_z, C
        elif self.mode=='s2s':  # Search to search
            q = x[:, :lens_x]  # B, lens_x, C
            k = x[:, lens_x:]  # B, lens_x, C
            v = x[:, lens_x:]  # B, lens_x, C
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, lens_z, lens_x; B, lens_x, lens_z

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # B, lens_z/x, C
        x = x.transpose(1, 2)  # B, C, lens_z/x
        x = x.reshape(B, -1, C)  # B, lens_z/x, C; NOTE: Rearrange channels, marginal improvement
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.mode == 's2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 't2s':
            x = torch.cat([k, x], dim=1)
        elif self.mode == 't2t':
            x = torch.cat([x, k], dim=1)
        elif self.mode == 's2s':
            x = torch.cat([x, k], dim=1)

        if return_attention:
            return x, attn
        else:
            return x
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class GateLayer(nn.Module):
    """门控机制，动态调节主副模态信息的融合比例。"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        gate_weight = self.gate(torch.cat([x, y], dim=-1))
        return x * gate_weight + y * (1 - gate_weight)

class ModalSpecificLayer(nn.Module):
    """模态特定的预处理层。"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.linear(x))

class RelativePositionEncoding(nn.Module):
    def __init__(self, dim, max_len=500):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(max_len, dim))
        # print(f"self.pos_embedding.shape: {self.pos_embedding.shape}") # torch.Size([500, 768])

    def forward(self, length):
        return self.pos_embedding[:length, :]

class CustomCrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads, max_len=500):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == self.dim, "dim must be divisible by num_heads"

        self.query_proj = ModalSpecificLayer(dim)
        self.key_proj = ModalSpecificLayer(dim)
        self.value_proj = ModalSpecificLayer(dim)

        self.relative_pos_encoding = RelativePositionEncoding(self.dim, max_len=max_len)
        self.scale = self.head_dim ** -0.5

        self.attn_dropout = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.gate_layer = GateLayer(dim)

    def forward(self, visible_features, infrared_features, return_attn=False):
        B, N, _ = visible_features.size()
        # print(f"visible_features.shape: {visible_features.shape}")  # torch.Size([32, 64, 768])

        # Apply modal-specific processing
        query = self.query_proj(infrared_features).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key_proj(visible_features).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = self.value_proj(visible_features).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # print(f"self.relative_pos_encoding(N).shape: {self.relative_pos_encoding(N).shape}")  # torch.Size([64, 64])
        pos_encoding = self.relative_pos_encoding(N).reshape(1, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention with Relative Position Encoding
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores += torch.matmul(query, pos_encoding.transpose(-2, -1))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Attention(Q, K, V)
        attn_output = torch.matmul(attn, value).transpose(1, 2).reshape(B, N, self.dim)

        # Apply projection and combine with the input
        attn_output = self.proj(attn_output)

        # Gated fusion
        output = self.gate_layer(visible_features, attn_output)

        # Normalize
        output = self.norm(output)

        if return_attn:
            return output, attn
        return output
