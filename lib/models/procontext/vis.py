import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, vit_b_32, ViT_B_16_Weights, ViT_B_32_Weights
from torchvision.datasets import CIFAR100, ImageFolder
from torchvision.transforms import InterpolationMode
import torchvision.transforms as v2

import matplotlib.pyplot as plt

# model properties (can be extracted by trial and error or inspecting the model code)
num_channels = 3
patch_size = 16
image_size = 224
num_classes = 1000
num_patches = image_size // patch_size  # patches per row or col
shuffle = True
dataset_source = "drive"

# build the NN
# source: https://pytorch.org/vision/main/models/vision_transformer.html
vit = None
if patch_size == 16:
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
if patch_size == 32:
    vit = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
assert vit is not None

# sanity check
batch_size = 3
img = torch.randn(batch_size, num_channels, image_size, image_size)
out = vit(img)
print("sanity check")
print(" - input :", img.size())
print(" - output:", out.size())
assert out.size() == torch.Size([batch_size, num_classes])

# load the data
dataset = None
transform = v2.Compose(
    [
        #v2.ToPILImage(),
        #v2.ConvertImageDtype(torch.float32),
        v2.Resize([image_size, image_size]),
        v2.ToTensor(),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

if dataset_source == "cifar100":
    dataset = CIFAR100(
        root="~/datasets/cifar100",
        train=False,
        download=True,
        transform=transform,
    )
if dataset_source == "drive":
    # load image from drive
    dataset = ImageFolder(
        root="/home/helm/tracker/ProContEXT-main/data/plane1",
        transform=transform,
    )
assert dataset is not None

loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
img, _ = next(iter(loader))
print("dataset item dimensions and range")
print(" - image:", img.size())
print(" - - min:", torch.min(img))
print(" - - max:", torch.max(img))


# monkey patch the forward function
class WrappedEncoderBlock(nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.attn = None

    def forward(self, input):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.encoder.ln_1(input)

        # forward with attention maps
        x, attn = self.encoder.self_attention(x, x, x)

        # save attention weights
        self.attn = attn.detach().cpu()

        x = self.encoder.dropout(x)
        x = x + input

        y = self.encoder.ln_2(x)
        y = self.encoder.mlp(y)
        return x + y


vit.encoder.layers[-1] = WrappedEncoderBlock(vit.encoder.layers[-1])
attn_layer = vit.encoder.layers[-1]


# visualize
# def unnormalize(img):
#     imin = img.min()
#     imax = img.max()
#     return (img - imin) / (imax - imin)


# def plot_img(ax, img, scale=None):  # img is a tensor [channels, size, size]
#     img = unnormalize(img) * 255
#     if scale is not None:
#         img = img * unnormalize(scale)
#     img = img.to(dtype=torch.int)
#     img = img.permute((1, 2, 0))
#     ax.imshow(img)


# def plot_attn(ax, attn):  # attn is a tensor [num_patches^2]
#     attn = attn.reshape([1, num_patches, num_patches])
#     plot_img(ax, attn)


# def plot_img_with_attn(ax, img, attn):
#     attn = attn.reshape([1, num_patches, num_patches])
#     attn = v2.functional.resize(
#         attn,
#         [image_size, image_size],
#         interpolation=InterpolationMode.BICUBIC,
#     )
#     plot_img(ax, img, scale=attn)


# # pass the image through the vit
# print("plotting...")
# _ = vit(img)
# attn = attn_layer.attn
# # dimensions:
# # - img: [batch, channels, image_size, image_size]
# # - attn: [batch, num_patches ^ 2 + 1, num_patches ^ 2 + 1]
# # print(f"attn.shape: {attn.shape}")
# # attn.shape: torch.Size([1, 197, 197])
# img_idx = 0

# class_token_idx = 0

# fig, ax = plt.subplots(nrows=2, ncols=3)

# maps = [
#     attn[img_idx, class_token_idx, 1:],
#     attn[img_idx, 1:, class_token_idx],
# ]
# for i, attn_map in enumerate(maps):
#     plot_img(ax[i][0], img[img_idx])
#     plot_attn(ax[i][1], attn_map)
#     plot_img_with_attn(ax[i][2], img[img_idx], attn_map)

# plt.show()


import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

# def visualize_attention(img, attn, image_size, num_patches, fig_scale=1.5):
#     """
#     可视化Vision Transformer的注意力权重。

#     参数:
#     - img: 输入图像，张量格式为[batch, channels, image_size, image_size]。
#     - attn: 注意力权重，张量格式为[batch, num_patches ^ 2 + 1, num_patches ^ 2 + 1]。
#     - image_size: 图像的尺寸。
#     - num_patches: 图像每行（或每列）的补丁数目。
#     - fig_scale: 图像放大的倍数。
#     """
#     def unnormalize(img_tensor):
#         """将图像张量反归一化到[0, 255]范围内。"""
#         img_tensor = img_tensor.squeeze(0)  # 假设batch_size=1，移除批量维度
#         img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())  # 归一化到[0, 1]
#         return img_tensor

#     def plot_img_with_attention(ax, img, attn_map):
#         """将图像和相应的注意力图层结合起来展示。"""
#         # 将注意力权重调整至图像尺寸
#         attn_map_resized = F.resize(attn_map, [image_size, image_size], InterpolationMode.BICUBIC)
#         img_with_attention = img * attn_map_resized
#         ax.imshow(img_with_attention.permute(1, 2, 0).numpy())
    
#     # 预处理图像和注意力权重
#     img_processed = unnormalize(img)
#     attn_processed = attn.squeeze(0)  # 假设batch_size=1

#     class_token_idx = 0
#     attn_map = attn_processed[class_token_idx, 1:].reshape(num_patches, num_patches)  # 忽略类别令牌的注意力
    
#     # 创建图像展示
#     fig, axs = plt.subplots(1, 2, figsize=(fig_scale * 6, fig_scale * 3))
#     axs[0].imshow(img_processed.permute(1, 2, 0).numpy())  # 原图
#     axs[0].set_title("Original Image")
#     plot_img_with_attention(axs[1], img_processed, attn_map)  # 带注意力权重的图
#     axs[1].set_title("Attention Overlay")
    
#     for ax in axs:
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()




import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

def visualize_attention(img, attn, num_patches, image_size):
    def unnormalize(img):
        imin = img.min()
        imax = img.max()
        return (img - imin) / (imax - imin)

    def plot_img(ax, img, scale=None):  # img is a tensor [channels, size, size]
        img = unnormalize(img) * 255
        if scale is not None:
            img = img * unnormalize(scale)
        img = img.to(dtype=torch.int)
        img = img.permute((1, 2, 0))
        ax.imshow(img)
        

    def plot_attn(ax, attn):  # attn is a tensor [num_patches^2]
        attn = attn.reshape([1, num_patches, num_patches])
        plot_img(ax, attn)

    def plot_img_with_attn(ax, img, attn):
        attn = attn.reshape([1, num_patches, num_patches])
        attn = v2.functional.resize(
            attn,
            [image_size, image_size],
            interpolation=InterpolationMode.BICUBIC,
        )
        plot_img(ax, img, scale=attn)

    img_idx = 0
    class_token_idx = 0

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
    
    maps = [
        attn[img_idx, class_token_idx, 1:],
        attn[img_idx, 1:, class_token_idx],
    ]
    for i, attn_map in enumerate(maps):
        print(f"attn_map: {attn_map.shape}") # attn_map: torch.Size([196])
        plot_img(ax[i][0], img[img_idx])
        print(f"img.size: {img[img_idx].shape}")
        plot_attn(ax[i][1], attn_map)
        plot_img_with_attn(ax[i][2], img[img_idx], attn_map)
    plt.tight_layout()
    plt.show()


_ = vit(img)
attn = attn_layer.attn
print(f"attn.shape: {attn.shape}") # attn.shape: torch.Size([1, 197, 197])
visualize_attention(img, attn, 14, 224)