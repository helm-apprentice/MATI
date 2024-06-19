双模vs其他  公开(RGBT234+GTOT)

|      | SiamCSR | MDNet+RGBT | C-COT | ECO  | SOWP+RGBT | SRDCF | CSR-DCF | KCF+RGBT | Ours*  |
| ---- | ------- | ---------- | ----- | ---- | --------- | ----- | ------- | -------- | ------ |
| MPR  | 0.76    | 0.72       | 0.71  | 0.70 | 0.70      | 0.64  | 0.62    | 0.46     | 0.78   |
| MSR  | 0.67    | 0.62       | 0.62  | 0.61 | 0.60      | 0.56  | 0.55    | 0.40     | 0.71   |
| FPS  |         |            |       |      |           |       |         |          | 128.54 |

双模 训练对比

|      | CMC1+CEB+CMC | CMC1+CMC | CMC1   | CMC+CEB+CMC | CMC+CMC | CMC*   |
| ---- | ------------ | -------- | ------ | ----------- | ------- | ------ |
| MPR  | 0.69         | 0.72     | 0.76   | 0.73        | 0.75    | 0.78   |
| MSR  | 0.57         | 0.64     | 0.68   | 0.63        | 0.68    | 0.71   |
| FPS  | 109.62       | 117.41   | 127.30 | 109.04      | 117.87  | 128.54 |

单模，token filter 对比

|         | SR0.5<br />before/after | SR0.75<br />before/after | SR0.9<br />before/after | AOR<br />before/after | APE<br />before/after |
| :-----: | :---------------------: | :----------------------: | :---------------------: | :-------------------: | :-------------------: |
|  self   |     0.98/0.99(+1%)      |     0.74/0.85(+15%)      |     0.38/0.45(+18%)     |   0.83/0.86(+3.6%)    |    9.57/6.71(-30%)    |
| GOT-10k |        0.95/0.95        |      0.86/0.85(-1%)      |    0.58/0.53(-8.6%)     |    0.85/0.84(-1%)     |   26.09/26.90(+3%)    |
| OTB-100 |     0.93/0.95(+2%)      |      0.63/0.68(+8%)      |     0.13/0.15(+15%)     |   0.76/0.78(+2.6%)    |    8.57/5.88(-31%)    |

|               | 单模<br />e | CMC1+CMC<br />extreme | CMC+CMC<br />e | CMC1<br />e | CMC<br />e |
| :-----------: | :---------: | :-------------------: | :------------: | :---------: | :--------: |
| track_success |   False/    |         True/         |     True/      |    True/    |   True/    |
|   max_ratio   |   0.0312/   |        0.017/         |    0.0112/     |   0.0151/   |  0.0141/   |
|  miss_count   |     1/      |          0/           |       0/       |     0/      |     1/     |
|      MPR      |    0.69/    |         0.77/         |     0.92/      |    0.75/    |   0.91/    |
|      MSR      |    0.44/    |         0.62/         |     0.85/      |    0.57/    |   0.87/    |

|               | CMC1+CMC<br />normal | CMC1<br />n | 单模<br />n | CMC+CMC<br />n | CMC<br />n |
| :-----------: | :------------------: | :---------: | :---------: | :------------: | :--------: |
| track_success |         True         |    True     |    True     |      True      |    True    |
|   max_ratio   |        0.0039        |   0.0042    |   0.0046    |     0.0059     |   0.0046   |
|  miss_count   |          1           |      0      |      0      |       0        |     0      |
|      MPR      |         0.99         |    0.98     |    0.98     |      0.99      |    0.99    |
|      MSR      |         0.89         |    0.89     |    0.89     |      0.99      |    0.99    |





```c++
class CMCBlock(nn.Module):
    ''' CrossModalCompensationBlock'''
    def __init__(self, dim, num_heads):
        super().__init__()
        self.name = 'CMCBlock'
        self.query_proj = nn.Linear(dim, dim)
        self.key_value_proj = nn.Linear(dim, dim)
        self.cmc_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, visible_features, infrared_features, return_attn=False):
        # Transform infrared features to query
        query = self.query_proj(infrared_features)
        # Use visible features as key and value
        key_value = self.key_value_proj(visible_features)

        # Apply cross-modal attention
        attn_output, attn_weights = self.cmc_attn(query=query, key=key_value, value=key_value)
        # Apply feed-forward network
        output = self.feed_forward(attn_output)

        # Normalize and add the residual
        output = self.norm(visible_features + output)
        if return_attn:
            return output, attn_weights
        return output
```

​      



```c++
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
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape
    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None
    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)
    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]
    # ===================================================================================
    """先将topk_idx在按升序排列之后，再使用gather，注意力权重相对顺序就不会改变"""
    topk_idx, _ = torch.sort(topk_idx, dim=1)
    # ===================================================================================
    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]
    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # concatenate these tokens
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)
    return tokens_new, keep_index, removed_index


class CEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0,):
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
        x_attn, attn = self.attn(self.norm1(x), mask, True)
        x = x + self.drop_path(x_attn)
        lens_t = global_index_template.shape[1]
        removed_index_search = None
        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x, global_index_search, removed_index_search = candidate_elimination(attn, x, lens_t, keep_ratio_search, global_index_search, ce_template_mask)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, global_index_template, global_index_search, removed_index_search, attn
```

```
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
```





Building on the principles established by prior advancements, the Token Spatial Filtering Module (TSF) introduces a refined focus on the spatial orientation of attention weights within the attention mechanism. This focus not only differentiates our work from preceding efforts but also infuses the tracking model with an intrinsic spatial awareness that reflects the target’s locational nuances more precisely.

Central to the TSF’s operation is the methodical evaluation of the attention map \( A \), which discerns and prioritizes tokens based on their spatial relevance to the target area. This is achieved through an attentive curation process:

\[
A_{t,s} = A[:, :, :L_t, L_t:], \quad T = [T_{t}; T_{s}]
\]

Upon obtaining the mean attention across the template and search regions, a sorting operation yields indices that highlight the spatially pivotal areas:

\[
A'_{\text{mean}} = \text{mean}(A_{t,s}, \text{dim}=2), \quad \text{indices} = \text{argsort}(A'_{\text{mean}}, \text{descending})
\]

From this sorted array, a subset of indices corresponding to the highest attention scores is extracted, encapsulated by the keep ratio \( \rho \):

\[
\rho = \text{keep\_ratio}, \quad \text{topk\_indices} = \text{indices}[:, : \lceil \rho \times L_s \rceil]
\]

In an innovative step that preserves the spatial coherence post-filtering, the top-k indices undergo a reordering process. This ensures that the spatial layout of the search feature tokens remains intact, thereby maintaining the integrity of the target's spatial information:

\[
\text{topk\_indices} = \text{sort}(\text{topk\_indices})
\]

The spatially sorted tokens, represented as \( T' \), are recombined with the template tokens \( T_t \) to form a spatially coherent and feature-enriched representation \( T'' \):

\[
T' = T \odot \text{keep\_index}, \quad T'' = T_{t} \oplus T'
\]

By incorporating this step of re-sorting the top-k indices, the CEBlock not only ensures that the most informative tokens are selected but also that their spatial arrangement echoes the structural layout of the target within the scene. This level of spatial fidelity is what sets our model apart, offering enhanced tracking performance by dynamically concentrating on the critical areas of interest within the image space. This precise spatial attention within the attention mechanism fortifies the CEBlock's contribution to the evolution of robust and accurate multi-modal tracking systems.
