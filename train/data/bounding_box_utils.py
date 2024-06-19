import torch


def rect_to_rel(bb, sz_norm=None):
    """Convert standard rectangular parametrization of the bounding box [x, y, w, h]
    to relative parametrization [cx/sw, cy/sh, log(w), log(h)], where [cx, cy] is the center coordinate.
    args:
        bb  -  N x 4 tensor of boxes.
        sz_norm  -  [N] x 2 tensor of value of [sw, sh] (optional). sw=w and sh=h if not given.
    """

    c = bb[...,:2] + 0.5 * bb[...,2:]
    if sz_norm is None:
        c_rel = c / bb[...,2:]
    else:
        c_rel = c / sz_norm
    sz_rel = torch.log(bb[...,2:])
    return torch.cat((c_rel, sz_rel), dim=-1)
'''
1. rect_to_rel 函数
此函数将标准的矩形边界框表示 [x, y, w, h] 转换为相对参数化形式 [cx/sw, cy/sh, log(w), log(h)]，
其中 [cx, cy] 是边界框的中心坐标。

bb: 边界框的张量，形状为 N x 4。
sz_norm: 可选的 [N] x 2 张量，表示 [sw, sh] 的值。如果未给出，则假定 sw = w 和 sh = h。
函数计算边界框的中心，并将其相对于 sz_norm 进行标准化（如果提供了 sz_norm）。
然后，它计算边界框大小的对数表示。
'''

def rel_to_rect(bb, sz_norm=None):
    """Inverts the effect of rect_to_rel. See above."""

    sz = torch.exp(bb[...,2:])
    if sz_norm is None:
        c = bb[...,:2] * sz
    else:
        c = bb[...,:2] * sz_norm
    tl = c - 0.5 * sz
    return torch.cat((tl, sz), dim=-1)
'''
2. rel_to_rect 函数
该函数是 rect_to_rel 的逆操作，将相对参数化形式的边界框转换回标准的矩形表示。

bb: 相对参数化的边界框张量。
sz_norm: 和 rect_to_rel 中相同，用于标准化尺寸。
'''

def masks_to_bboxes(mask, fmt='c'):

    """ Convert a mask tensor to one or more bounding boxes.
    Note: This function is a bit new, make sure it does what it says.  /Andreas
    :param mask: Tensor of masks, shape = (..., H, W)
    :param fmt: bbox layout. 'c' => "center + size" or (x_center, y_center, width, height)
                             't' => "top left + size" or (x_left, y_top, width, height)
                             'v' => "vertices" or (x_left, y_top, x_right, y_bottom)
    :return: tensor containing a batch of bounding boxes, shape = (..., 4)
    """
    batch_shape = mask.shape[:-2]
    mask = mask.reshape((-1, *mask.shape[-2:]))
    bboxes = []

    for m in mask:
        mx = m.sum(dim=-2).nonzero()
        my = m.sum(dim=-1).nonzero()
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]
        bboxes.append(bb)

    bboxes = torch.tensor(bboxes, dtype=torch.float32, device=mask.device)
    bboxes = bboxes.reshape(batch_shape + (4,))

    if fmt == 'v':
        return bboxes

    x1 = bboxes[..., :2]
    s = bboxes[..., 2:] - x1 + 1

    if fmt == 'c':
        return torch.cat((x1 + 0.5 * s, s), dim=-1)
    elif fmt == 't':
        return torch.cat((x1, s), dim=-1)

    raise ValueError("Undefined bounding box layout '%s'" % fmt)
'''
3. masks_to_bboxes 函数
此函数将一个或多个图像掩码转换为边界框。

mask: 掩码张量，形状为 (..., H, W)。
fmt: 边界框的布局格式，可以是 c（中心 + 大小）、t（左上角 + 大小）或 v（顶点）。
函数遍历每个掩码，找出掩码中的非零区域，并计算包围这些区域的边界框。
'''

def masks_to_bboxes_multi(mask, ids, fmt='c'):
    assert mask.dim() == 2
    bboxes = []

    for id in ids:
        mx = (mask == id).sum(dim=-2).nonzero()
        my = (mask == id).float().sum(dim=-1).nonzero()
        bb = [mx.min(), my.min(), mx.max(), my.max()] if (len(mx) > 0 and len(my) > 0) else [0, 0, 0, 0]

        bb = torch.tensor(bb, dtype=torch.float32, device=mask.device)

        x1 = bb[:2]
        s = bb[2:] - x1 + 1

        if fmt == 'v':
            pass
        elif fmt == 'c':
            bb = torch.cat((x1 + 0.5 * s, s), dim=-1)
        elif fmt == 't':
            bb = torch.cat((x1, s), dim=-1)
        else:
            raise ValueError("Undefined bounding box layout '%s'" % fmt)
        bboxes.append(bb)

    return bboxes
'''
4. masks_to_bboxes_multi 函数
这个函数是 masks_to_bboxes 的扩展，用于从单个掩码中提取多个边界框，每个边界框对应于不同的标识符 id。

mask: 二维掩码张量。
ids: 一个包含不同标识符的列表，每个标识符对应于掩码中的一个目标。
函数遍历每个 id，找出掩码中对应于该 id 的区域，并计算边界框。
这些函数在目标跟踪和图像分析中非常有用，特别是在需要从图像掩码提取目标信息或在不同边界框表示之间转换时。
'''