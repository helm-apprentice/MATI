import torch
import numpy as np
from lib.utils.misc import NestedTensor


class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return NestedTensor(img_tensor_norm, amask_tensor)

"""
初始化 (__init__ 方法):

    设置了两个张量 self.mean 和 self.std，分别代表图像通道的均值和标准差。这些值通常用于图像的标准化处理，有助于模型更好地学习和泛化。
        这里的值看起来是针对ImageNet数据集预训练模型的标准化参数。
    view((1, 3, 1, 1)) 调整形状以适配输入图像的维度，使其能在批次、通道、高度、宽度的格式上进行广播操作。
    .cuda() 将这些张量移动到CUDA设备上，以便在GPU上进行快速计算。
处理 (process 方法):

    输入参数包括 img_arr（一个图像数组）和 amask_arr（一个对应的注意力掩码数组）。
    首先，img_arr 被转换为一个PyTorch张量，并通过.cuda().float().permute((2,0,1)).unsqueeze(dim=0)进行处理，使其适配深度学习模型的输入格式。具体来说：
        .cuda() 将图像数据移动到CUDA设备。
        .float() 将图像数据类型转换为浮点数，这对于后续的数学运算很重要。
        .permute((2,0,1)) 重排数组维度，将图像从高度x宽度x通道格式转换为通道x高度x宽度格式，以符合PyTorch期望的输入格式。
        .unsqueeze(dim=0) 增加一个批次维度，使其成为一个四维张量，以便可以作为模型的输入。
    然后，对图像张量进行标准化处理：先除以255.0将像素值归一化到[0,1]，然后减去均值并除以标准差进行归一化。
    对于注意力掩码 amask_arr，它被转换为一个布尔型的PyTorch张量，并添加一个批次维度，以匹配图像数据的格式。
    最后，返回一个NestedTensor对象，其中包含了处理后的图像张量和注意力掩码张量。NestedTensor可能是一个专门用于封装这两种数据的类，以方便后续操作，但其具体实现在这段代码中并未给出。
"""

class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return img_tensor_norm, amask_tensor


class PreprocessorX_onnx(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std  # (1, 3, H, W)
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)
