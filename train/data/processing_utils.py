import torch
import math
import cv2 as cv
import torch.nn.functional as F
import numpy as np

'''modified from the original test implementation
Replace cv.BORDER_REPLICATE with cv.BORDER_CONSTANT
Add a variable called att_mask for computing attention and positional encoding later'''


def sample_target(im, target_bb, search_area_factor, output_sz=None, mask=None):
    """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

    args:
        im - cv image
        target_bb - target box [x, y, w, h]
        search_area_factor - Ratio of crop size to target size
        output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

    returns:
        cv image - extracted crop
        float - the factor by which the crop has been resized to make the crop size equal output_size
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb
    # Crop image
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    x2 = x1 + crop_sz

    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    y2 = y1 + crop_sz

    x1_pad = max(0, -x1)
    x2_pad = max(x2 - im.shape[1] + 1, 0)

    y1_pad = max(0, -y1)
    y2_pad = max(y2 - im.shape[0] + 1, 0)

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
    if mask is not None:
        mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

    # Pad
    im_crop_padded = cv.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv.BORDER_CONSTANT)
    # deal with attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H,W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0
    if mask is not None:
        mask_crop_padded = F.pad(mask_crop, pad=(x1_pad, x2_pad, y1_pad, y2_pad), mode='constant', value=0)

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        if mask is None:
            return im_crop_padded, resize_factor, att_mask
        mask_crop_padded = \
        F.interpolate(mask_crop_padded[None, None], (output_sz, output_sz), mode='bilinear', align_corners=False)[0, 0]
        return im_crop_padded, resize_factor, att_mask, mask_crop_padded

    else:
        if mask is None:
            return im_crop_padded, att_mask.astype(np.bool_), 1.0
        return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded
'''
sample_target函数的作用：
目标提取：从输入图像im中提取以目标边界框target_bb为中心的区域。这个区域的大小是原始目标大小的search_area_factor的平方倍。这样做的目的是获取足够的背景信息，以便更好地定位目标。
尺寸调整：如果提供了output_sz，提取的区域将被调整大小到这个指定的尺寸（正方形）。这通常是为了满足网络输入的尺寸要求或保持尺度一致性。
输出：   函数返回调整大小后的图像、调整大小的因子resize_factor（如果进行了尺寸调整），以及一个注意力掩码att_mask。如果提供了mask，还会处理和返回相应的掩码。
'''

def transform_image_to_crop(box_in: torch.Tensor, box_extract: torch.Tensor, resize_factor: float,
                            crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box_in - the box for which the co-ordinates are to be transformed
        box_extract - the box about which the image crop has been extracted.
        resize_factor - the ratio between the original image scale and the scale of the image crop
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """
    box_extract_center = box_extract[0:2] + 0.5 * box_extract[2:4]

    box_in_center = box_in[0:2] + 0.5 * box_in[2:4]

    box_out_center = (crop_sz - 1) / 2 + (box_in_center - box_extract_center) * resize_factor
    box_out_wh = box_in[2:4] * resize_factor

    box_out = torch.cat((box_out_center - 0.5 * box_out_wh, box_out_wh))
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out


def jittered_center_crop(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz, m)
                                for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)
    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors

    return frames_crop, box_crop, att_mask, masks_crop


def jittered_center_crop_multi_scale(frames, box_extract, box_gt, search_area_factor, output_sz, masks=None):
    """ For each frame in frames, extracts a square crop centered at box_extract, of area search_area_factor^2
    times box_extract area. The extracted crops are then resized to output_sz. Further, the co-ordinates of the box
    box_gt are transformed to the image crop co-ordinates

    args:
        frames - list of frames
        box_extract - list of boxes of same length as frames. The crops are extracted using anno_extract
        box_gt - list of boxes of same length as frames. The co-ordinates of these boxes are transformed from
                    image co-ordinates to the crop co-ordinates
        search_area_factor - The area of the extracted crop is search_area_factor^2 times box_extract area
        output_sz - The size to which the extracted crops are resized

    returns:
        list - list of image crops
        list - box_gt location in the crop co-ordinates
        """

    if masks is None:
        crops_resize_factors = [sample_target(f, a, search_area_factor, output_sz)
                                for f, a in zip(frames, box_extract)]
        frames_crop, resize_factors, att_mask = zip(*crops_resize_factors)
        masks_crop = None
    else:
        crops_resize_factors = [sample_target(f, a, factor, output_sz, m)
                                    for factor in search_area_factor
                                    for f, a, m in zip(frames, box_extract, masks)]
        frames_crop, resize_factors, att_mask, masks_crop = zip(*crops_resize_factors)

    # frames_crop: tuple of ndarray (128,128,3), att_mask: tuple of ndarray (128,128)
    crop_sz = torch.Tensor([output_sz, output_sz])

    box_gt = box_gt * len(search_area_factor)
    box_extract = box_extract * len(search_area_factor)
    # find the bb location in the crop
    '''Note that here we use normalized coord'''
    box_crop = [transform_image_to_crop(a_gt, a_ex, rf, crop_sz, normalize=True)
                for a_gt, a_ex, rf in zip(box_gt, box_extract, resize_factors)]  # (x1,y1,w,h) list of tensors

    return frames_crop, box_crop, att_mask, masks_crop
'''
jittered_center_crop_multi_scale 函数是一个处理多尺度图像裁剪和目标框转换的函数。
它主要用于从一系列帧中提取以特定目标框为中心的裁剪区域，并调整这些裁剪区域的大小。
同时，它还负责转换关联的目标框坐标。下面详细解释这个函数的作用和参数：

功能
多尺度裁剪：从输入的帧（frames）中以 box_extract 指定的框为中心，提取一系列裁剪区域。
        这些裁剪区域的面积是 box_extract 面积的 search_area_factor 的平方倍。
裁剪区域调整：将提取的裁剪区域调整（resize）到指定的输出大小 output_sz。
目标框坐标转换：转换 box_gt 中指定的目标框坐标，使之与裁剪区域的坐标系统一致。
参数
    frames: 原始帧的列表。
    box_extract: 用于提取裁剪区域的目标框列表，与 frames 长度相同。
    box_gt: 要转换坐标的目标框列表，与 frames 长度相同。
    search_area_factor: 裁剪区域面积与 box_extract 面积之比的因子。
    output_sz: 裁剪区域的输出尺寸。
    masks: （可选）掩码列表，与 frames 长度相同，用于裁剪操作。
返回值
    裁剪后的帧列表。
    转换后的 box_gt 坐标列表。
    裁剪区域内的注意力掩码（att_mask）。
    （如果提供）裁剪后的掩码列表。
使用场景
这个函数在多尺度跟踪场景中特别有用，其中目标可能在不同尺度下出现。
通过提取多尺度的裁剪区域并调整它们的大小，可以更灵活地处理目标的尺度变化。
同时，通过调整目标框的坐标，可以确保目标框与裁剪后的图像保持一致，从而适用于后续的跟踪算法。
'''

'''
`jittered_center_crop` 函数和 `jittered_center_crop_multi_scale` 
函数都用于从一系列帧中提取以特定目标框为中心的裁剪区域，但主要区别在于它们处理尺度变化的方式。
以下是两个函数的主要区别：

### `jittered_center_crop` 函数

- **单一尺度处理**：这个函数针对每个帧只进行一次裁剪，
    裁剪区域的面积由 `search_area_factor` 决定。
- **裁剪逻辑**：对于每个帧和对应的 `box_extract`，提取一个裁剪区域，
    其面积是 `box_extract` 面积的 `search_area_factor` 的平方倍，
    然后将该裁剪区域调整到 `output_sz`。
- **目标框坐标转换**：转换 `box_gt` 中指定的目标框坐标，使之与裁剪区域的坐标系统一致。

### `jittered_center_crop_multi_scale` 函数

- **多尺度处理**：这个函数能够对每个帧进行多次裁剪，
    每次裁剪根据不同的 `search_area_factor` 值产生不同尺度的裁剪区域。
- **裁剪逻辑**：对于每个帧和 `box_extract`，
    根据多个 `search_area_factor` 值提取多个裁剪区域，
    每个区域的面积分别是 `box_extract` 面积的每个 `search_area_factor` 值的平方倍，
    然后将这些裁剪区域调整到 `output_sz`。
- **目标框坐标转换**：类似于 `jittered_center_crop`，转换 `box_gt` 中指定的目标框坐标。

### 区别总结

- **尺度变化处理**：`jittered_center_crop_multi_scale` 用于处理多尺度的情况，
    可以为同一帧提取多个不同尺度的裁剪区域，而 `jittered_center_crop` 只
    提取单一尺度的裁剪区域。
- **应用场景**：`jittered_center_crop_multi_scale` 更适合于需要考虑目标尺度变化的场景，
    例如，当目标大小在视频序列中有显著变化时。
    相比之下，`jittered_center_crop` 更适用于目标尺度相对稳定的情况。

在实际应用中，选择哪个函数取决于您的具体需求，特别是是否需要处理目标在视频序列中的尺度变化。
'''

def transform_box_to_crop(box: torch.Tensor, crop_box: torch.Tensor, crop_sz: torch.Tensor, normalize=False) -> torch.Tensor:
    """ Transform the box co-ordinates from the original image co-ordinates to the co-ordinates of the cropped image
    args:
        box - the box for which the co-ordinates are to be transformed
        crop_box - bounding box defining the crop in the original image
        crop_sz - size of the cropped image

    returns:
        torch.Tensor - transformed co-ordinates of box_in
    """

    box_out = box.clone()
    box_out[:2] -= crop_box[:2]

    scale_factor = crop_sz / crop_box[2:]

    box_out[:2] *= scale_factor
    box_out[2:] *= scale_factor
    if normalize:
        return box_out / crop_sz[0]
    else:
        return box_out

