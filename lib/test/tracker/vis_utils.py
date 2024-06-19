import numpy as np
import torch
from matplotlib import pyplot as plt

############## used for visulize eliminated tokens #################
def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices


def gen_masked_tokens(tokens, indices, alpha=0.2):
    # indices = [i for i in range(196) if i not in indices]
    indices = indices[0].astype(int)
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens


def recover_image(tokens, H, W, Hp, Wp, patch_size):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(Hp, Wp, patch_size, patch_size, 3).swapaxes(1, 2).reshape(H, W, 3)
    return image


def pad_img(img):
    height, width, channels = img.shape
    im_bg = np.ones((height, width + 8, channels)) * 255
    im_bg[0:height, 0:width, :] = img
    return im_bg

'''
这段代码实现了Vision Transformer网络层层MASK的可视化。
主要步骤:
1. 输入图像image和每层需要mask的indice mask_indices
2. 将图像分成patches,得到image_tokens
3. 遍历每层mask索引:
   - 根据之前层的索引构建当前层索引
   - 基于当前层索引mask图像patches,得到masked_tokens
   - 恢复masked_tokens为图像
4. 最终可视化每层mask的图像拼接结果
这样通过逐层累积mask,并recover图像,实现了Vision Transformer对图像理解的层次过程可视化。
mask_indices记录了每层需要屏蔽的patches,recover图像则反向补充被屏蔽的内容。拼接可视化每个过程。
这种可视化有助于理解ViT网络的内部表示和作用机制
'''
def gen_visualization(image, mask_indices, patch_size=16):
    # image [224, 224, 3]
    # mask_indices, list of masked token indices

    # mask mask_indices need to cat
    # mask_indices = mask_indices[::-1]
    num_stages = len(mask_indices)
    for i in range(1, num_stages):
        mask_indices[i] = np.concatenate([mask_indices[i-1], mask_indices[i]], axis=1)

    # keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)

    stages = [
        recover_image(gen_masked_tokens(image_tokens, mask_indices[i]), H, W, Hp, Wp, patch_size)
        for i in range(num_stages)
    ]
    imgs = [image] + stages
    imgs = [pad_img(img) for img in imgs]
    viz = np.concatenate(imgs, axis=1)
    return viz






# def attn_to_img(attn_weights, raw_img):
#     # 计算所有头的平均注意力
#     attn = attn_weights.mean(dim=0).cpu()
#     # 插值调整注意力图的大小
#     resized_attn = torch.nn.functional.interpolate(attn.unsqueeze(0).unsqueeze(0), 
#                                                    size=(192, 192), 
#                                                    mode='bilinear', 
#                                                    align_corners=False).squeeze()
#     plt.figure()
#     # 显示原始图像
#     # plt.imshow(raw_img.permute(1, 2, 0).cpu().detach().numpy())
#     # 叠加热图

#     plt.imshow(resized_attn.cpu().detach().numpy(), cmap='hot', alpha=0.5)
#     # plt.colorbar()
#     save_path = "/home/helm/tracker/ProContEXT-main/data/attn.png"
#     plt.savefig(save_path)
#     plt.close()

def attn_to_image(attn_matrix):
    """获取的是多头平均注意力图"""
    attn_matrix = attn_matrix.detach()
        # 根据维度的不同，进行不同的平均操作
    if attn_matrix.dim() == 4:  # 假设形状为 [32, 12, 400, 400]
        # 首先在头维度上平均，然后在样本维度上平均
        attn_matrix = attn_matrix.mean(dim=1).mean(dim=0).cpu().numpy()
    elif attn_matrix.dim() == 3:  # 假设形状为 [144, 32, 32]
        # 直接在头维度上平均
        attn_matrix = attn_matrix.mean(dim=0).cpu().numpy()
    else:
        # 其他情况，直接转换为numpy数组（仅作为示例，实际应根据需要调整）
        attn_matrix = attn_matrix.cpu().numpy()
    fig, ax = plt.subplots()
    cax = ax.matshow(attn_matrix, cmap='viridis')
    fig.colorbar(cax)
    save_path = "/home/helm/tracker/ProContEXT-main/data/attn.png"
    plt.show()    
    plt.close(fig)
    