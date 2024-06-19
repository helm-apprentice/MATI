import torch.nn as nn

from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

'''
在给定的`PatchEmbed`类中，输入图像通过一个卷积层被转换为一系列嵌入向量，这些向量代表图像中的不同区域（或称为“补丁”）。下面是`z`在这个过程中的构成和维度解析：

1. **输入维度** (`x`): 假设输入`x`是一个形状为`(B, C, H, W)`的张量，其中：
   - `B` 是批量大小（batch size），即一次处理的图像数量。
   - `C` 是通道数（channel），对于RGB图像来说通常是3。
   - `H` 和 `W` 分别是图像的高度和宽度。

2. **通过卷积层** (`self.proj`): 输入`x`通过一个具有`patch_size`大小核（kernel）和步长（stride）的卷积层，输出的形状变为`(B, embed_dim, grid_H, grid_W)`，其中：
   - `embed_dim` 是设定的嵌入维度。
   - `grid_H` 和 `grid_W` 分别是补丁在垂直和水平方向上的数量，计算方式为图像的高度`H`除以补丁的高度，图像的宽度`W`除以补丁的宽度。这里假设`H`和`W`都能被补丁的尺寸整除。

3. **扁平化和转置** (`flatten` 和 `transpose`): 如果`flatten`为`True`，则将卷积输出的每个补丁展平成一维向量，并将通道维度和新的补丁维度进行交换，结果的形状为`(B, num_patches, embed_dim)`，其中：
   - `num_patches` 是补丁的总数，即`grid_H * grid_W`。

4. **归一化** (`self.norm`): 应用归一化层（如果有）到扁平化和转置后的张量上，形状保持为`(B, num_patches, embed_dim)`。

因此，对于`z = self.patch_embed(z)`这一步，最终`z`的形状为`(B, num_patches, embed_dim)`。这里的每个维度代表的含义如下：

- 第一个维度`B`代表批量大小，即一次处理的图像数量。
- 第二个维度`num_patches`代表将输入图像分割成多少个补丁（或token）。
- 第三个维度`embed_dim`代表每个补丁的嵌入向量维度，即每个补丁被编码成的向量空间的维数。

最后，`z += self.pos_embed_z`这一步可能是将位置嵌入加到每个补丁的嵌入向量上，以提供模型关于每个补丁位置的信息。这个操作不改变`z`的形状，但会增加位置信息到嵌入表示中。
'''