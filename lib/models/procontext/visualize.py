import os
import torch
import numpy as np
import math
from functools import partial
import torch
import torch.nn as nn

import ipywidgets as widgets
from IPython.display import display
import io
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

import warnings
warnings.filterwarnings("ignore")

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    ## type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            # print(f"attn.shape: {attn.shape}")
            # attn.shape: torch.Size([1, 6, 1281, 1281])
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ 
    Vision Transformer 
    """

    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(
                math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(
            w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        # print(x.shape)
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class VitGenerator(object):
    def __init__(self, name_model, patch_size, device, evaluate=True, random=False, verbose=False):
        self.name_model = name_model
        self.patch_size = patch_size
        self.evaluate = evaluate
        self.device = device
        self.verbose = verbose
        self.model = self._getModel()
        self._initializeModel()
        if not random:
            self._loadPretrainedWeights()

    def _getModel(self):
        if self.verbose:
            print(
                f"[INFO] Initializing {self.name_model} with patch size of {self.patch_size}")
        if self.name_model == 'vit_tiny':
            model = VisionTransformer(patch_size=self.patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        elif self.name_model == 'vit_small':
            model = VisionTransformer(patch_size=self.patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        elif self.name_model == 'vit_base':
            model = VisionTransformer(patch_size=self.patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                      qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        else:
            raise f"No model found with {self.name_model}"

        return model

    def _initializeModel(self):
        if self.evaluate:
            for p in self.model.parameters():
                p.requires_grad = False

            self.model.eval()

        self.model.to(self.device)

    def _loadPretrainedWeights(self):
        if self.verbose:
            print("[INFO] Loading weights")
        url = None
        if self.name_model == 'vit_small' and self.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"

        elif self.name_model == 'vit_small' and self.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"

        elif self.name_model == 'vit_base' and self.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"

        elif self.name_model == 'vit_base' and self.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

        if url is None:
            print(
                f"Since no pretrained weights have been found with name {self.name_model} and patch size {self.patch_size}, random weights will be used")

        else:
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

    def get_last_selfattention(self, img):
        return self.model.get_last_selfattention(img.to(self.device))

    def __call__(self, x):
        return self.model(x)
    

def transform(img, img_size):
    img = transforms.Resize(img_size)(img)
    img = transforms.ToTensor()(img)
    return img


def visualize_predict(model, img, img_size, patch_size, device):
    img_pre = transform(img, img_size)
    # print(f"img_pre.shape: {img_pre.shape}")
    attention = visualize_attention(model, img_pre, patch_size, device)
    # print(f"attn.shape: {attention.shape}")
    # attn.shape: (6, 256, 320)
    plot_attention(img, attention)


def visualize_attention(model, img, patch_size, device):
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
        img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    #print(img.shape)
    attentions = model.get_last_selfattention(img.to(device))
    # print(f"attn.shape: {attentions.shape}")
    # attn.shape: torch.Size([1, 6, 1281, 1281])
    nh = attentions.shape[1]  # number of head

    # keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(
        0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    return attentions


def plot_attention(img, attention):
    # print(f"attn.shape: {attention.shape}")
    # attn.shape: (6, 256, 320)
    n_heads = attention.shape[0]

    plt.figure(figsize=(20, 20), dpi=200)
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.mean(attention, 0)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(fig, cmap='inferno')
        plt.title(text[i])
    plt.show()

    plt.figure(figsize=(20, 20), dpi=200)
    for i in range(n_heads):
        plt.subplot(n_heads//3, 3, i+1)
        plt.imshow(attention[i], cmap='inferno')
        plt.title(f"Head n: {i+1}")
    plt.tight_layout()
    plt.show()



class Loader(object):
    def __init__(self):
        self.uploader = widgets.FileUpload(accept='image/*', multiple=False)
        self._start()

    def _start(self):
        display(self.uploader)

    def getLastImage(self):
        try:
            for uploaded_filename in self.uploader.value:
                uploaded_filename = uploaded_filename
            img = Image.open(io.BytesIO(
                bytes(self.uploader.value[uploaded_filename]['content'])))

            return img
        except:
            return None

    def saveImage(self, path):
        with open(path, 'wb') as output_file:
            for uploaded_filename in self.uploader.value:
                content = self.uploader.value[uploaded_filename]['content']
                output_file.write(content)



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if device.type == "cuda":
    torch.cuda.set_device(0)

name_model = 'vit_small'
patch_size = 16

# model = VitGenerator(name_model, patch_size, 
#                      device, evaluate=True, random=False, verbose=True)



# Visualizing Dog Image
path = '/home/helm/tracker/ProContEXT-main/data/plane1/visible/00000030.bmp'
img = Image.open(path)
factor_reduce = 1
img_size = tuple(np.array(img.size[::-1]) // factor_reduce) 
# visualize_predict(model, img, img_size, patch_size, device)


def visual(img, attention, patch_size, factor_reduce=2):
    
    def visualize_predict(attention, img, img_size, patch_size):
        img_pre = transform(img, img_size)
        attention = visualize_attention(attention, img_pre, patch_size)
        # print(f"attn.shape: {attention.shape}")
        # attn.shape: (6, 256, 320)
        plot_attention(img, attention)

    def transform(img, img_size):
        img = transforms.Resize(img_size)(img)
        img = transforms.ToTensor()(img)
        return img
    
    def visualize_attention(attentions, img, patch_size):
        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - \
            img.shape[2] % patch_size
        img = img[:, :w, :h].unsqueeze(0)
        print(f"img.shape: {img.shape}")
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        # attentions = model.get_last_selfattention(img.to(device))
        print(f"attn0.shape: {attentions.shape}")
        # attn.shape: torch.Size([1, 6, 1281, 1281])
        nh = attentions.shape[1]  # number of head

        # keep only the output patch attention
        attentions = attentions[0, :, 0, :].reshape(nh, -1)
        """这段代码将原始的四维注意力矩阵转换为一个形状为 [12, 233] 的二维张量，其中每一行对应一个注意力头的输出，每一列对应序列中一个补丁的注意力权重。
        这个二维张量只包含了特定补丁（可能是类嵌入）对整个序列的关注程度，而不是完整的序列对序列的注意力关系。
        这样的表示方式简化了数据结构，使得更容易分析或可视化每个头对整个序列的注意力分布。"""
        print(f"attn1.shape: {attentions.shape}")
        # -------------------------------------------------------
        attentions = attentions.unsqueeze(0).unsqueeze(-1)
        attentions = nn.functional.interpolate(attentions, size=(w_featmap*h_featmap, 1),
                                               mode='bicubic', align_corners=True)
        attentions = attentions.squeeze()
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        print(f"attn2.shape: {attentions.shape}")
        attentions = nn.functional.interpolate(attentions.unsqueeze(
            0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
        print(f"attn3.shape: {attentions.shape}")
        return attentions

    def plot_attention(img, attention):
        # print(f"attn.shape: {attention.shape}")
        # attn.shape: (6, 256, 320)
        n_heads = attention.shape[0]

        plt.figure(figsize=(20, 20), dpi=200)
        text = ["Original Image", "Head Mean"]
        for i, fig in enumerate([img, np.mean(attention, 0)]):
            plt.subplot(1, 2, i+1)
            plt.imshow(fig, cmap='inferno')
            plt.title(text[i])
        plt.show()

        plt.figure(figsize=(20, 20), dpi=200)
        for i in range(n_heads):
            plt.subplot(n_heads//3, 3, i+1)
            plt.imshow(attention[i], cmap='inferno')
            plt.title(f"Head n: {i+1}")
        plt.tight_layout()
        plt.show()

    img_size = tuple(np.array(img.size[::-1]) // factor_reduce)
    visualize_predict(attention, img, img_size, patch_size)
class Visualize_attention:
    """   左下矩阵
    一共12个block，只在第4，7，10个block使用消除，那么在0，1，2层，就是正常去可视化256*144的矩阵，沿着模板补丁的维度（列）对这些权重求和以及取平均得到256*1；
    然后在第3，4，5层，左下角矩阵从180*144，求和或者求平均之后得到180*1，然后按照第一次消除的序号（180+76）填充到256*1；
    在第6，7，8层，左下角矩阵从126*144，求和或者求平均之后得到126*1，然后按照第一次消除的序号和第二次消除的序号（126+76+54）填充到256*1；
    在第9，10，11层，左下角矩阵从89*144，求和或者求平均之后得到89*1，然后按照第一二三次消除的序号（89+76+54+37）填充到256*1
    """
    def __init__(self, remove_indexes, global_indexes, template_size=128, search_size=256, patch_size=16, ce_loc=[3, 6, 9]):
        #self.attn = attention  # torch.Size([1, 12, 144+s, 144+s])
        #self.img = image
        self.patch_size = patch_size
        self.template_size = template_size
        self.patches = (search_size // patch_size)
        self.remove_indexes_list = remove_indexes
        self.global_indexes_list = global_indexes
        self.ce_loc = ce_loc

    def hook_attn(self, attn, template_size=192, patch_size=16):
        """
        torch.Size([1, 12, 233, 233])->torch.Size([12, 89, 144])->torch.Size([89, 144])
        """
        index = template_size // patch_size
        #print(f"index: {index}")
        #print(f"attn0.size: {attn.shape}")
        search_to_template_attention = attn[0, :, index*index:, :index*index]
        #print(f"attn0.size: {search_to_template_attention.shape}")
        # search_to_template_attention_norm = (search_to_template_attention - search_to_template_attention.min()) / (search_to_template_attention.max() - search_to_template_attention.min())
        #print(f"search_to_template_attn_norm.shape: {search_to_template_attention_norm.shape}")  # torch.Size([12, 89, 144])
        mean_attn = search_to_template_attention.mean(dim=0)  # torch.Size([89, 144])
        return mean_attn
    
    def squeeze_attn(self, attn, type='sum'):
        """torch.Size([89, 144])->torch.Size([89, 1])"""
        if  type == 'sum':
            single_attn = torch.sum(attn, dim=1, keepdim=True)
        elif type == 'mean':
            single_attn = torch.mean(attn, dim=1, keepdim=True)
        # print(f"attn: {single_attn}")
        return single_attn
    
    def pad_attn(self, single_attn, remove_indexes, global_indexes, patches):
        """torch.Size([89, 1])->torch.Size([256, 1])"""
        single_attn = single_attn.view(-1)
        remove_indexes = torch.sort(remove_indexes.long(), descending=False).values.squeeze()
        global_indexes = torch.sort(global_indexes.long(), descending=False).values.squeeze()
        #print(f"global_indexes: {global_indexes}")
        
        full_attn = torch.zeros(patches*patches, 1, device=torch.device("cuda"))
        #print(f"full_attn: {full_attn}")
        #print(f"single_attn: {single_attn}")
        full_attn[remove_indexes] = 0
        full_attn = full_attn.view(-1)
        full_attn[global_indexes] = single_attn
        #print(f"full_attn: {full_attn}")
        full_attn = full_attn.view(patches*patches, 1)
        # print(f"full_attn: {full_attn}")
        full_attn_norm = (full_attn - full_attn.min()) / (full_attn.max() - full_attn.min())
        # print(f"full_attn_norm: {full_attn_norm}")
        return full_attn_norm
    
    def reshape_attn(self, attn, patches, patch_size):
        """torch.Size([256, 1])->torch.Size([16, 16])->(256, 256)"""
        #print(f"attn0.shape: {attn.shape}")  # torch.Size([256, 1])
        attn = attn.reshape(patches, patches)
        #print(f"attn1.shape: {attn.shape}")  # torch.Size([16, 16])
        img_size = (patch_size*patches, patch_size*patches)
        attention = nn.functional.interpolate(attn.unsqueeze(
            0).unsqueeze(0), size=img_size, mode="nearest").squeeze()
        #print(f"attn2.shape: {attention.shape}")
        # attention = attention.unsqueeze(2)
        attn_np = attention.detach().cpu().numpy()
        return attn_np  # 已经转换为可以直接imshow的（256， 256）

    
    def process_attn(self, attn, attn_layer):
        attn = self.hook_attn(attn, self.template_size, self.patch_size)
        attn_sum = self.squeeze_attn(attn, 'sum')
        #print(f"attn_sum: {attn_sum}")
        attn_mean = self.squeeze_attn(attn, 'mean')
        #print(f"attn_mean: {attn_mean}")
        if (self.ce_loc[0] <= attn_layer):
            if (attn_layer <= self.ce_loc[1]):
                remove_indexes = self.remove_indexes_list[0]
                global_indexes = self.global_indexes_list[0]
                # print(f"remove_indexes.shape: {remove_indexes.shape}")
                # print(f"global_indexes.shape: {global_indexes.shape}")
                
            elif (self.ce_loc[1] < attn_layer <= self.ce_loc[2]):
                remove_indexes = torch.cat((self.remove_indexes_list[0], self.remove_indexes_list[1]), dim=1)
                global_indexes = self.global_indexes_list[1]
                # print(f"remove_indexes.shape: {remove_indexes.shape}")
                # print(f"global_indexes.shape: {global_indexes.shape}")

            elif (self.ce_loc[2] < attn_layer):
                remove_indexes = torch.cat(self.remove_indexes_list, dim=1)
                global_indexes = self.global_indexes_list[2]
                # print(f"remove_indexes.shape: {remove_indexes.shape}")
                # print(f"global_indexes.shape: {global_indexes.shape}")

            attn_sum = self.pad_attn(attn_sum, remove_indexes, global_indexes, self.patches)
            #print(f"attn_sum_pad: {attn_sum}")
            attn_mean = self.pad_attn(attn_mean, remove_indexes, global_indexes, self.patches)
            #print(f"attn_mean_pad: {attn_mean}")

        attn_sum = self.reshape_attn(attn_sum, self.patches, self.patch_size)
        attn_mean = self.reshape_attn(attn_mean, self.patches, self.patch_size)

        return attn_sum, attn_mean  # numpy数组（256， 256）
    
    
    def img_attn(self, img, attn, attn_layer ,save_dir, index, mark):
        attn_sum, attn_mean = self.process_attn(attn, attn_layer)  # （256， 256）
        #print(f"attn_sum.shape: {attn_sum.shape}")
        #print(f"attn_mean.shape: {attn_mean.shape}")
        fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=200)
        # 第一行：原始图像和求和注意力热图
        axs[0, 0].imshow(img)
        axs[0, 0].set_title('Search Image')
            
        cax1 = axs[0, 1].imshow(attn_sum, cmap='viridis')
        fig.colorbar(cax1, ax=axs[0, 1], orientation='vertical')
        axs[0, 1].set_title(f'Sum Attention Heatmap({mark})')

        axs[0, 2].imshow(img)
        axs[0, 2].imshow(attn_sum, cmap='viridis', alpha=0.5)
        axs[0, 2].set_title(f'Search Image with Sum Attention({mark})')

        # 第二行：原始图像和平均注意力热图
        axs[1, 0].imshow(img)
        axs[1, 0].set_title('Search Image')
        
        cax2 = axs[1, 1].imshow(attn_mean, cmap='inferno')
        fig.colorbar(cax2, ax=axs[1, 1], orientation='vertical')
        axs[1, 1].set_title(f'Mean Attention Heatmap({mark})')
        
        axs[1, 2].imshow(img)
        axs[1, 2].imshow(attn_mean, cmap='inferno', alpha=0.5)
        axs[1, 2].set_title(f'Search Image with Mean Attention({mark})')

        # 保存图形到指定目录，并按照索引命名
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'attn_layer{attn_layer}_{index}_{mark}.png')
        plt.tight_layout()
        plt.axis('off')
        fig.savefig(save_path)
        plt.close(fig)  # 关闭图形以释放资源
    

    # def img_attn(self, img, attn, attn_layer ,save_dir, index, mark):
    #     if index == 217:
    #         attn_sum, attn_mean = self.process_attn(attn, attn_layer)
    #         for i, (im, title) in enumerate(zip([img, attn_sum], ['Original Image', 'Sum Attention Heatmap'])):
    #             # 创建新的 Figure 对象
    #             fig = plt.figure(figsize=(5, 5), dpi=200)

    #             # 添加一个 Axes 对象，并在其中绘制图像
    #             ax = fig.add_subplot(1, 1, 1)
    #             ax.imshow(im, cmap='viridis' if i == 1 else None)  # 第二个图使用 'viridis' 颜色映射，其他图根据需要调整

    #             # 隐藏坐标轴标签和刻度线
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             ax.set_xlabel('')
    #             ax.set_ylabel('')

    #             # 设置图像标题（如果需要）
    #             #ax.set_title(title)

    #             # 保存图像
    #             if not os.path.exists(save_dir):
    #                 os.makedirs(save_dir)
    #             save_path = os.path.join(save_dir, f'attn_layer{attn_layer}_{index}_{mark}.png')
    #             plt.savefig(save_path, bbox_inches='tight', dpi=200)

    #             # 关闭当前 Figure，准备绘制下一个图像
    #             plt.close(fig)


class Visualize_attention1:
    """   右上矩阵
    一共12个block，只在第4，7，10个block使用消除，那么在0，1，2层，就是正常去可视化256*144的矩阵，沿着模板补丁的维度（列）对这些权重求和以及取平均得到256*1；
    然后在第3，4，5层，左下角矩阵从180*144，求和或者求平均之后得到180*1，然后按照第一次消除的序号（180+76）填充到256*1；
    在第6，7，8层，左下角矩阵从126*144，求和或者求平均之后得到126*1，然后按照第一次消除的序号和第二次消除的序号（126+76+54）填充到256*1；
    在第9，10，11层，左下角矩阵从89*144，求和或者求平均之后得到89*1，然后按照第一二三次消除的序号（89+76+54+37）填充到256*1
    """
    def __init__(self, remove_indexes, global_indexes, template_size=192, search_size=256, patch_size=16, ce_loc=[3, 6, 9]):
        #self.attn = attention  # torch.Size([1, 12, 144+s, 144+s])
        #self.img = image
        self.patch_size = patch_size
        self.template_size = template_size
        self.patches = (search_size // patch_size)
        self.remove_indexes_list = remove_indexes
        self.global_indexes_list = global_indexes
        self.ce_loc = ce_loc

    def hook_attn(self, attn, template_size=192, patch_size=16):
        """
        torch.Size([1, 12, 233, 233])->torch.Size([12, 89, 144])->torch.Size([89, 144])
        """
        index = template_size // patch_size
        #print(f"index: {index}")
        #print(f"attn0.size: {attn.shape}")
        # search_to_template_attention = attn[0, :, index*index:, :index*index]
        search_to_template_attention = attn[0, :, :index*index, index*index:] # torch.Size([12, 144, 89])
        #print(f"attn0.size: {search_to_template_attention.shape}")
        # search_to_template_attention_norm = (search_to_template_attention - search_to_template_attention.min()) / (search_to_template_attention.max() - search_to_template_attention.min())
        #print(f"search_to_template_attn_norm.shape: {search_to_template_attention_norm.shape}")  # torch.Size([12, 89, 144])
        mean_attn = search_to_template_attention.mean(dim=0)  # torch.Size([89, 144])
        return mean_attn
    
    def squeeze_attn(self, attn, type='sum'):
        """torch.Size([144, 89])->torch.Size([1, 89])"""
        if  type == 'sum':
            single_attn = torch.sum(attn, dim=0, keepdim=True)
        elif type == 'mean':
            single_attn = torch.mean(attn, dim=0, keepdim=True)
        # print(f"attn: {single_attn}")
        return single_attn
    
    def pad_attn(self, single_attn, remove_indexes, global_indexes, patches):
        """torch.Size([1, 89])->torch.Size([1, 256])"""
        single_attn = single_attn.view(-1)
        remove_indexes = torch.sort(remove_indexes.long(), descending=False).values.squeeze()
        global_indexes = torch.sort(global_indexes.long(), descending=False).values.squeeze()
        #print(f"global_indexes: {global_indexes}")
        
        full_attn = torch.zeros(1, patches*patches, device=torch.device("cuda"))
        #print(f"full_attn: {full_attn}")
        #print(f"single_attn: {single_attn}")
        full_attn[remove_indexes] = 0
        full_attn = full_attn.view(-1)
        full_attn[global_indexes] = single_attn
        #print(f"full_attn: {full_attn}")
        full_attn = full_attn.view(1, patches*patches)
        # print(f"full_attn: {full_attn}")
        full_attn_norm = (full_attn - full_attn.min()) / (full_attn.max() - full_attn.min())
        # print(f"full_attn_norm: {full_attn_norm}")
        return full_attn_norm
    
    def reshape_attn(self, attn, patches, patch_size):
        """torch.Size([1, 256])->torch.Size([16, 16])->(256, 256)"""
        #print(f"attn0.shape: {attn.shape}")  # torch.Size([256, 1])
        attn = attn.reshape(patches, patches)
        #print(f"attn1.shape: {attn.shape}")  # torch.Size([16, 16])
        img_size = (patch_size*patches, patch_size*patches)
        attention = nn.functional.interpolate(attn.unsqueeze(
            0).unsqueeze(0), size=img_size, mode="nearest").squeeze()
        #print(f"attn2.shape: {attention.shape}")
        # attention = attention.unsqueeze(2)
        attn_np = attention.detach().cpu().numpy()
        return attn_np  # 已经转换为可以直接imshow的（256， 256）

    
    def process_attn(self, attn, attn_layer):
        attn = self.hook_attn(attn, self.template_size, self.patch_size)
        attn_sum = self.squeeze_attn(attn, 'sum')
        #print(f"attn_sum: {attn_sum}")
        attn_mean = self.squeeze_attn(attn, 'mean')
        #print(f"attn_mean: {attn_mean}")
        if (self.ce_loc[0] <= attn_layer):
            if (attn_layer < self.ce_loc[1]):
                remove_indexes = self.remove_indexes_list[0]
                global_indexes = self.global_indexes_list[0]
                # print(f"remove_indexes.shape: {remove_indexes.shape}")
                # print(f"global_indexes.shape: {global_indexes.shape}")
                
            elif (self.ce_loc[1] <= attn_layer < self.ce_loc[2]):
                remove_indexes = torch.cat((self.remove_indexes_list[0], self.remove_indexes_list[1]), dim=1)
                global_indexes = self.global_indexes_list[1]
                # print(f"remove_indexes.shape: {remove_indexes.shape}")
                # print(f"global_indexes.shape: {global_indexes.shape}")

            elif (self.ce_loc[2] <= attn_layer):
                remove_indexes = torch.cat(self.remove_indexes_list, dim=1)
                global_indexes = self.global_indexes_list[2]
                # print(f"remove_indexes.shape: {remove_indexes.shape}")
                # print(f"global_indexes.shape: {global_indexes.shape}")

            attn_sum = self.pad_attn(attn_sum, remove_indexes, global_indexes, self.patches)
            #print(f"attn_sum_pad: {attn_sum}")
            attn_mean = self.pad_attn(attn_mean, remove_indexes, global_indexes, self.patches)
            #print(f"attn_mean_pad: {attn_mean}")

        attn_sum = self.reshape_attn(attn_sum, self.patches, self.patch_size)
        attn_mean = self.reshape_attn(attn_mean, self.patches, self.patch_size)

        return attn_sum, attn_mean  # numpy数组（256， 256）
    
    def img_attn(self, img, attn, attn_layer ,save_dir, index, mark):
        attn_sum, attn_mean = self.process_attn(attn, attn_layer)  # （256， 256）
        #print(f"attn_sum.shape: {attn_sum.shape}")
        #print(f"attn_mean.shape: {attn_mean.shape}")
        fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=200)
        # 第一行：原始图像和求和注意力热图
        axs[0, 0].imshow(img)
        axs[0, 0].set_title('Search Image')
            
        cax1 = axs[0, 1].imshow(attn_sum, cmap='viridis')
        fig.colorbar(cax1, ax=axs[0, 1], orientation='vertical')
        axs[0, 1].set_title(f'Sum Attention Heatmap({mark})')

        axs[0, 2].imshow(img)
        axs[0, 2].imshow(attn_sum, cmap='viridis', alpha=0.5)
        axs[0, 2].set_title(f'Search Image with Sum Attention({mark})')

        # 第二行：原始图像和平均注意力热图
        axs[1, 0].imshow(img)
        axs[1, 0].set_title('Search Image')
        
        cax2 = axs[1, 1].imshow(attn_mean, cmap='inferno')
        fig.colorbar(cax2, ax=axs[1, 1], orientation='vertical')
        axs[1, 1].set_title(f'Mean Attention Heatmap({mark})')
        
        axs[1, 2].imshow(img)
        axs[1, 2].imshow(attn_mean, cmap='inferno', alpha=0.5)
        axs[1, 2].set_title(f'Search Image with Mean Attention({mark})')

        # 保存图形到指定目录，并按照索引命名
        save_path = os.path.join(save_dir, f'attn_layer{attn_layer}_{index}_{mark}.png')
        plt.tight_layout()
        plt.axis('off')
        fig.savefig(save_path)
        plt.close(fig)  # 关闭图形以释放资源
        


def main():
    model = VitGenerator(name_model, patch_size, 
                        device, evaluate=True, random=False, verbose=True)
    factor_reduce = 1
    img_size = tuple(np.array(img.size[::-1]) // factor_reduce) 
    # print(img_size) # (512, 640)
    img_pre = transform(img, img_size)
    w, h = img_pre.shape[1] - img_pre.shape[1] % patch_size, img_pre.shape[2] - \
        img_pre.shape[2] % patch_size
    img_pre = img_pre[:, :w, :h].unsqueeze(0)

    attentions = model.get_last_selfattention(img_pre.to(device))
    # print(f"attentions: {attentions.shape}")
    visual(img, attentions, 16, factor_reduce=1)

if __name__ == '__main__':
    main()