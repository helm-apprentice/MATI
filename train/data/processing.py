import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F

import traceback


def stack_tensors(x):
    if isinstance(x, dict):
        return {key: stack_tensors(val) for key, val in x.items()}
    elif isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        # 调用的时候只提供了transform 和 joint_transform
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform} #     transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),tfm.RandomHorizontalFlip(probability=0.5))


    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor   
        '''
        settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR, # 是个列表
                                        'search': cfg.DATA.SEARCH.FACTOR}
        '''
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode  # mode = 'squence'
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        if len(data['template_images']['infrared']) > 0:
            data['use_infrared'] = True   # 是否使用红外
        else:
            data['use_infrared'] = False

        image_types = ['visible', 'infrared'] if data['use_infrared'] else ['visible']
        # Apply joint transforms
        if self.transform['joint'] is not None:
            for t in image_types:
                data['template_images'][t], data['template_anno'][t], data['template_masks'][t] = self.transform['joint'](
                    image=data['template_images'][t], bbox=data['template_anno'][t], mask=data['template_masks'][t])
                
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)
        #print(f'middle_process_data: {data}')
        # for s in ['template', 'search']:
        s = 'template'
        assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
            "In pair mode, num train/test frames must be 1"
        #print(f'going {s}')
        
        try:
            data[s + '_att'] = {}
            for i, t in enumerate(image_types):
            # Add a uniform noise to the center pos
            # jittered_anno = {'visible': {}, 'infrared': {}}
                jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno'][t]]
                # print(f'jittered_anno: {jittered_anno}')
                w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
                crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s][i])
                if (crop_sz < 1).any():
                    data['valid'] = False
                    # print("Too small box is found. Replace it with new data.")
                    print('if (crop_sz < 1).any():')
                    return data
                
                # Crop image region centered at jittered_anno box and get the attention mask
                crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'][t], jittered_anno,
                                                                                data[s + '_anno'][t], self.search_area_factor[s][i],
                                                                                self.output_sz[s], masks=data[s + '_masks'][t])
                # Apply transforms
                
                data[s + '_images'][t], data[s + '_anno'][t], data[s + '_att'][t], data[s + '_masks'][t] = self.transform[s](
                    image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
                # print(f"data1: {data['template_att']}")    
                # print('data[s_att]: {}'.format(data[s + '_att']))
                for ele in data[s + '_att'][t]:
                    if (ele == 1).all():
                        data['valid'] = False
                        print('if (ele == 1).all():')
                        return data
                # print(f"data2: {data['template_att']}")        
                for ele in data[s + '_att'][t]:
                    feat_size = self.output_sz[s] // 16
                    mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                    if (mask_down == 1).all():
                        data['valid'] = False
                        print('if (mask_down == 1).all():')
                        return data
            # print(f"data3: {data['template_att']}") 
        except Exception as e:
            stack_trace = traceback.format_exc()
            print("Exception occurred:", e)
            print(stack_trace)
        #print('if s == template  end!')
        s = 'search'
        assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
            "In pair mode, num train/test frames must be 1"
        #print(f's == {s}')
        try:
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]  # 计算抖动后的注释框

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3] # 计算宽度和高度

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s]) # 计算裁剪尺寸
            if (crop_sz < 1).any(): # 如果裁剪尺寸小于1
                data['valid'] = False # 设置数据为无效
                # print("Too small box is found. Replace it with new data.")
                return data # 返回数据
            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                            data[s + '_anno'], self.search_area_factor[s],
                                                                            self.output_sz[s], masks=data[s + '_masks'])
            #print(f'search_before_transform: {data}')
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)
            #print(f'search_after_transform: {data}')
            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']: # 对每个注意力掩码元素进行检查
                if (ele == 1).all(): # 如果所有值都为1
                    data['valid'] = False # 设置数据为无效
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data # 返回数据
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']: # 对每个注意力掩码元素
                # 计算特征尺寸
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride # 16是神经网络的步长
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                # 计算降采样后的掩码
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all(): # 如果所有值都为1
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data
        except Exception as e:
            stack_trace = traceback.format_exc()
            print("Exception occurred:", e)
            print(stack_trace)
            # else :
            #     raise(NotImplementedError)

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


        #     jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

        #     # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
        #     w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]
        #     if isinstance(self.search_area_factor[s], list):
        #         crop_sz = [torch.ceil(torch.sqrt(w * h) * scale) for scale in self.search_area_factor[s]]
        #         if (crop_sz[0] < 1).any():
        #             data['valid'] = False
        #             # print("Too small box is found. Replace it with new data.")
        #             return data
        #     else:
        #         crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
        #         if (crop_sz < 1).any():
        #             data['valid'] = False
        #             # print("Too small box is found. Replace it with new data.")
        #             return data

        #     # Crop image region centered at jittered_anno box and get the attention mask
        #     if s == "search":
        #         crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
        #                                                                         data[s + '_anno'], self.search_area_factor[s],
        #                                                                         self.output_sz[s], masks=data[s + '_masks'])
        #     elif s == "template":
        #         crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop_multi_scale(data[s + '_images'], jittered_anno,
        #                                                                         data[s + '_anno'], self.search_area_factor[s],
        #                                                                         self.output_sz[s], masks=data[s + '_masks'])
        #     else:
        #         raise(NotImplementedError)
        #     # Apply transforms
        #     data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
        #         image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

        #     # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
        #     # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
        #     for ele in data[s + '_att']:
        #         if (ele == 1).all():
        #             data['valid'] = False
        #             # print("Values of original attention mask are all one. Replace it with new data.")
        #             return data
        #     # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
        #     for ele in data[s + '_att']:
        #         feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
        #         # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
        #         mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
        #         if (mask_down == 1).all():
        #             data['valid'] = False
        #             # print("Values of down-sampled attention mask are all one. "
        #             #       "Replace it with new data.")
        #             return data

        # data['valid'] = True
        # # if we use copy-and-paste augmentation
        # if data["template_masks"] is None or data["search_masks"] is None:
        #     data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
        #     data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # # Prepare output
        # if self.mode == 'sequence':
        #     data = data.apply(stack_tensors)
        # else:
        #     data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        # return data
'''
尺度变化处理：对于搜索图像（search），使用的是 jittered_center_crop 函数，该函数进行单一尺度的裁剪。
            对于模板图像（template），使用的是 jittered_center_crop_multi_scale 函数，
            该函数可以进行多尺度的裁剪。

处理流程：在搜索图像和模板图像上执行不同的裁剪和变换操作。模板图像的处理考虑到了目标可能在不同尺度下出现的情况。

数据有效性检查：对于裁剪的大小和注意力掩码进行检查，确保数据有效。
            如果裁剪的区域太小或者注意力掩码全部为1，则认为数据无效。

额外处理：在处理完图像裁剪和变换后，对于序列模式（sequence），对数据进行堆叠；对于成对模式（pair），
        选择列表中的第一个元素。
'''

'''
1. **`__init__` 方法中的参数**: 在您最初给出的代码中，
    `__init__` 方法的参数只包括 `search_area_factor`, `output_sz`, `center_jitter_factor`,
      `scale_jitter_factor`, 和 `mode`。在您后来提供的代码中，还增加了一个名为 `settings` 的参数。

2. **处理多尺度搜索区域**: 在您后来提供的代码中，对于 `'template'` 和 `'search'` 模式，处理方式有所不同。
    对于 `'search'` 模式，使用 `prutils.jittered_center_crop` 方法进行图像裁剪。
    而对于 `'template'` 模式，使用 `prutils.jittered_center_crop_multi_scale` 方法，
    这表明在处理模板图像时可能考虑了多个尺度。

3. **更复杂的搜索区域尺寸处理**: 在您后来提供的代码中，`search_area_factor` 可以是一个列表，
    这意味着可能会处理多个不同尺寸的搜索区域。这在最初的代码中没有体现。

4. **额外的有效性检查**: 在您后来提供的代码中，对于处理后的注意力掩码，除了检查是否所有元素都为1，
    还对降采样后的掩码进行了额外的检查。

这些区别表明，后来提供的代码版本在处理模板和搜索图像时更加灵活和复杂，
尤其是在处理多尺度的情况以及在数据有效性检查方面。这些改进可能是为了提高处理流程的精确度和适应性，
以便更好地适应不同的图像处理需求。
'''