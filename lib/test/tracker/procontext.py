import math
import numpy as np
from PIL import Image
from lib.models.procontext import build_procontext
from lib.models.procontext.visualize import Visualize_attention, Visualize_attention1
from lib.test.tracker.basetracker import BaseTracker
import torch
import matplotlib.pyplot as plt
from lib.test.tracker.vis_utils import gen_visualization, attn_to_image
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from copy import deepcopy

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

# from lib.test.evaluation.environment import


class ProContEXT(BaseTracker):
    def __init__(self, params, dataset_name):
        super(ProContEXT, self).__init__(params)
        network = build_procontext(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True) # CPU上加载
        self.cfg = params.cfg
        # self.network = network
        self.network = network.cuda() # 转移到GPU上推理
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        #print(f"debug: {self.debug}")
        self.use_visdom = params.use_visdom
        #print(f"use_visdom: {self.use_visdom}")
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                pass
                # self.add_hook()
                # self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.z_dict_list = []
        self.update_intervals = [200]
        self.has_infrared = False
        self.vis_index = 0

    def initialize(self, image, info: dict):    # 需要修改
        self.z_dict_list = []
        # print(f"type of image: {type(image)}")
        factors = self.params.template_factor
        # print(f"factors: {factors}")
        if isinstance(image, list):
            self.has_infrared = True
            assert len(image) == 2, "input is not a pair of images"
            crop_resize_patches = [sample_target(image[0], info['init_bbox'], factors[0], output_sz=self.params.template_size),
                                   sample_target(image[1], info['inf_init_bbox'], factors[1], output_sz=self.params.template_size)]
        else:
            self.has_infrared = False
            crop_resize_patches = [sample_target(image, info['init_bbox'], factors[0], output_sz=self.params.template_size)]
        # sample_target函数返回裁剪并调整大小后的模板图像、缩放因子以及对应的注意力掩码（如果提供了输出大小）。
        z_patch_arr, resize_factor, z_amask_arr = zip(*crop_resize_patches)
        self.z_patch_arr = z_patch_arr
        '''
        crop_resize_patches是一个列表，其中包含一个或两个元组，每个元组由sample_target函数返回的三个值组成：裁剪并调整大小后的模板图像、缩放因子和对应的注意力掩码。
        使用zip(*crop_resize_patches)将这些元组“解压”，按照它们的位置，将相同位置的元素组合到新的元组中。
        这样，所有第一个位置的元素（裁剪并调整大小后的模板图像）组合成一个元组，所有第二个位置的元素（缩放因子）组合成另一个元组，以此类推。
        '''
        for idx in range(len(z_patch_arr)):
            template = self.preprocessor.process(z_patch_arr[idx], z_amask_arr[idx])
            with torch.no_grad():
                self.z_dict1 = template
            self.z_dict_list.append(self.z_dict1)
        # print(f"len(z_patch_arr): {len(z_patch_arr)}")
        # print(f"self.z_dict_list: {self.z_dict_list}")
        # print("Type of self.z_dict_list:", type(self.z_dict_list[0]))
        self.box_mask_z = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            # for i in range(len(self.params.template_factor) * 2):
            if self.has_infrared:
                v_template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor[0],
                                                            template.tensors.device).squeeze(1)
                self.box_mask_z.append(generate_mask_cond(self.cfg, 1, template.tensors.device, v_template_bbox))
                i_template_bbox = self.transform_bbox_to_crop(info['inf_init_bbox'], resize_factor[1],
                                                            template.tensors.device).squeeze(1)
                self.box_mask_z.append(generate_mask_cond(self.cfg, 1, template.tensors.device, i_template_bbox))
            else:
                template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor[0],
                                                            template.tensors.device).squeeze(1)
                self.box_mask_z.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))

        if len(self.box_mask_z) == 1:
            self.box_mask_z = self.box_mask_z[0]
        # 在后续调用的时候会根据是否列表来采取不同的措施

        # #init dynamic templates with static templates用静态模板初始化动态模板列表
        # for idx in range(len(self.params.template_factor)):
        #     self.z_dict_list.append(deepcopy(self.z_dict_list[idx]))

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}
        # ++++++++++++++++++++++++++++++分割++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # # forward the template once
        # # 这个函数会从输入图像中采样出一个目标区域，并返回一个包含目标区域张量数组、缩放因子和注意力掩码数组的元组。
        # z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
        #                                             output_sz=self.params.template_size)
        # self.z_patch_arr = z_patch_arr
        # # 对模板图像做预处理,然后进行前向传播提取特征。这一步进行了网络计算,可以视为前向计算过程的一部分。
        # template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        # with torch.no_grad(): # 使用with语句和torch.no_grad()上下文管理器，禁止梯度计算
        #     self.z_dict1 = template

        # self.box_mask_z = None
        # if self.cfg.MODEL.BACKBONE.CE_LOC:
        #     template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
        #                                                 template.tensors.device).squeeze(1)
        #     self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)

        # # save states
        # self.state = info['init_bbox']
        # self.frame_id = 0
        # if self.save_all_boxes:
        #     '''save all predicted boxes'''
        #     all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
        #     return {"all_boxes": all_boxes_save}

    def track(self, image, seq_name = None, info: dict = None):
        assert isinstance(image, list)
        if len(image) == 2:
            self.has_infrared = True
        elif len(image) == 1:
            self.has_infrared = False
        H, W, _ = image[0].shape
        # print(f"h,w: {H, W}")  # h,w: (512, 640)
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image[0], self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            # print(f"self.z_dict_list: {type(self.z_dict_list)}")
            if isinstance(self.z_dict_list, (list, tuple)):
                self.z_dict = []
                for i in range(len(self.z_dict_list)):
                    self.z_dict.append(self.z_dict_list[i].tensors)
            if len(self.z_dict_list) == 1:
                self.z_dict = self.z_dict_list[0].tensors
            # print(f"self.z_dict: {type(self.z_dict)}")
            '''------------------------------------------------------------------------------------------------------------'''
            out_dict = self.network.forward(template=self.z_dict, search=x_dict.tensors, ce_template_mask=self.box_mask_z, return_last_attn=True)
            '''   network = class ProContEXT   '''
            # print(f"out_dict: {out_dict.keys()}")
            # out_dict: dict_keys(['pred_boxes', 'score_map', 'size_map', 'offset_map', 'score', 'attn_cmc', 'attn', 'removed_indexes_s', 'backbone_feat'])
        # add hann windows
        pred_score_map = out_dict['score_map']
        # conf_score = out_dict['score']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for idx, update_i in enumerate([100]):
        #     if self.frame_id % update_i == 0 and conf_score > 0.7:
        #         crop_resize_patches2 = [sample_target(image, self.state, factor, output_sz=self.params.template_size)
        #                                             for factor in self.params.template_factor]
        #         z_patch_arr2, _, z_amask_arr2 = zip(*crop_resize_patches2)
        #         for idx_s in range(len(z_patch_arr2)):
        #             template_t = self.preprocessor.process(z_patch_arr2[idx_s], z_amask_arr2[idx_s])
        #             self.z_dict_list[idx_s+len(self.params.template_factor)] = template_t
        # print(f"debug: {self.debug}")
        # print(f"visdom: {self.visdom}")
        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image[0], cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, seq_name)
                if not os.path.exists(save_path): os.makedirs(save_path)
                save_path = os.path.join(save_path, "%04d.jpg" % self.frame_id)
                # print(save_path)
                cv2.imwrite(save_path, image_BGR)
            else:
                # print("Run visdom")
                #print(f"info: {info}")
                #self.visdom.register((image, info['init_bbox'], self.state), 'Tracking', 1, 'Tracking')

                # self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                # self.visdom.register(torch.from_numpy(self.z_patch_arr[0]).permute(2, 0, 1), 'image', 1, 'template')
                # # self.visdom.register(torch.from_numpy(self.z_patch_arr[1]).permute(2, 0, 1), 'image', 1, 'inf_template')
                # self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                # self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                # if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                #     removed_indexes_s = out_dict['removed_indexes_s']
                #     removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                #     masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                #     self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
                if 'attn' in out_dict and out_dict['attn']:
                    atten_layers = [5, 8]  # CEB是从369层进行的筛选，但是在层上attn是完整的，残缺的token进入下一层才会出现残缺的attn
                    # print(len(out_dict['attn']))  # 添加的CMC不参与这里的计数，因为它不返回注意力
                    for atten_layer in atten_layers:
                        attn = out_dict['attn'][atten_layer]
                        img = torch.from_numpy(x_patch_arr)
                        # save_dir = "/home/helm/tracker/ProContEXT-main/output/heatmap(rightop)/"
                        # save_dir = f"/home/helm/tracker/ProContEXT-main/output/heatmap(otb_no_sort)/{seq_name}" if seq_name else f"/home/helm/tracker/ProContEXT-main/output/heatmap(otb_no_sort)"
                        save_dir = f"/home/helm/tracker/heatmap(plane_test)/{seq_name}"
                        remove_indexes = out_dict['removed_indexes_s']
                        global_indexes = out_dict['global_indexes_s']
                        template_size = self.cfg.TEST.TEMPLATE_SIZE
                        search_size = self.cfg.TEST.SEARCH_SIZE
                        ce_loc = self.cfg.MODEL.BACKBONE.CE_LOC
                        self.vis_index += 1
                        track_idx = self.vis_index // len(atten_layers) + 1
                        vis = Visualize_attention(remove_indexes, global_indexes, template_size, search_size, ce_loc=ce_loc)
                        mark = 'double' if self.has_infrared else 'single'
                        vis.img_attn(img, attn, atten_layer, save_dir, track_idx, mark)
                    # # 假设 attentions 是完整的注意力矩阵，形状为 [1, 12, 233, 233]
                    # # 提取左下角的 89x144 矩阵
                    # search_to_template_attention = attn[0, :, 144:, :144]  # 假设 :144 表示搜索区域的起始索引，: 表示模板区域的结束索引
                    # print(f"search_to_template_attn.shape: {search_to_template_attention.shape}")  # torch.Size([12, 89, 144])
                    # search_to_template_attention_norm = (search_to_template_attention - search_to_template_attention.min()) / (search_to_template_attention.max() - search_to_template_attention.min())
                    # print(f"search_to_template_attn_norm.shape: {search_to_template_attention_norm.shape}")  # torch.Size([12, 89, 144])
                    # mean_attn = search_to_template_attention_norm.mean(dim=0)
                    # # 将注意力权重转换为numpy数组以便可视化
                    # attention_array = mean_attn.detach().cpu().numpy()

                    # 创建热图
                    # plt.imshow(attention_array, cmap='viridis')
                    # plt.colorbar()
                    # plt.title('Search Region Attention to Template Center')
                    # plt.xlabel('Search Patches')
                    # plt.ylabel('Template Patches')
                    # plt.show()
                #     # print(f"attn.shape: {attn.shape}")  # attn.shape: torch.Size([1, 12, 233, 233])
                #     #print(f"image.type: {type(image[0])}") # image.type: <class 'numpy.ndarray'>
                #     img = Image.fromarray(image[0])
                #     #print(f"img.type: {type(img)}") # img.type: <class 'PIL.Image.Image'>
                #     visual(img, attn, 16, 1)

                #     for i, attn in enumerate(out_dict['attn']):
                #         self.visdom.register(attn.detach().mean(dim=1).mean(dim=0).cpu().numpy(), 'heatmap', 1, f"CEBlock Attention Heatmap {i+1}")
                # if 'attn_cmc' in out_dict:
                #     attn_cmc = out_dict['attn_cmc']
                #     raw_template = torch.from_numpy(self.z_patch_arr[0]).permute(2, 0, 1)
                #     img = attn_to_image(attn_cmc)
                    # self.visdom.visdom.image(img, opts=dict(title='CMC Attention Map'))
                #     attn_img = self.visdom.attn_to_image(attn_cmc)
                #     self.visdom.visdom.image(attn_img, opts=dict(title='CMC Attention Map'))
                #     attn_heatmap = self.visdom.plot_attention_heatmap(attn_cmc)
                    # self.visdom.visdom.image(attn_heatmap, opts=dict(title='CMC Attention Heatmap'))
                    # self.visdom.register(attn_heatmap, 'heatmap', 1, "CMC Attention Heatmap")

                # while self.pause_mode:
                #     if self.step:
                #         self.step = False
                #         break
        if self.save_all_boxes:
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return ProContEXT


'''
当然，让我们更详细地探讨 `OSTrack` 和 `ProContEXT` 的模板处理和跟踪过程的差异：

### 模板处理

1. **OSTrack**:
   - 在初始化 (`initialize`) 阶段，`OSTrack` 处理单个模板图像。
        它对这个图像执行采样和预处理，然后将其传递给网络以提取特征。
   - `OSTrack` 使用单一的模板进行整个跟踪过程，不进行模板的动态更新。
        这意味着网络在跟踪过程中始终依赖于初始帧中的目标信息。

2. **ProContEXT**:
   - `ProContEXT` 在初始化阶段采用多尺度模板处理。它不仅处理一个模板，而是处理多个模板图像，
        并将它们存储在 `z_dict_list` 中。这允许 `ProContEXT` 在追踪过程中访问更多的上下文信息。
   - 此外，`ProContEXT` 根据特定条件（例如帧编号和置信度分数）动态更新模板。
        这意味着模型可以根据追踪过程中目标的变化来调整其模板，从而更好地适应场景变化。

### 跟踪过程

1. **OSTrack**:
   - 在每个跟踪步骤中，`OSTrack` 使用单个搜索区域图像，并结合其单一的模板特征进行目标位置的预测。
   - `OSTrack` 的主要焦点是利用初始模板与当前搜索区域之间的关系来确定目标的位置。

2. **ProContEXT**:
   - `ProContEXT` 不仅处理当前的搜索区域图像，还根据特定条件动态选择或更新模板。
        这允许 `ProContEXT` 在追踪过程中更好地适应目标或场景的变化。
   - `ProContEXT` 通过综合考虑多个模板提供的信息，可以更全面地理解目标的外观变化，
        从而有助于在长时间序列或复杂场景中维持稳定的跟踪。

总结来说，`OSTrack` 专注于使用单个初始模板进行高效追踪，而 `ProContEXT` 采用更动态的方式，
    通过多模板和模板更新机制来适应目标和环境的变化。这种方法在处理目标外观显著变化的场景中可能更有优势。
'''

"""
Visdom 是一个由 Facebook Research 创建的用于创建、组织和共享实时丰富数据可视化的工具。它支持多种类型的可视化，包括图表、图像和文本等，非常适合用于监控实时的实验进程和结果。

在你的代码片段中，通过使用 Visdom 进行可视化，可以直观地展示跟踪任务中的关键信息和中间结果。这里是具体的可视化类型和用途：

### 图像可视化
- **搜索区域和模板图像**：通过注册图像到 Visdom，可以实时查看跟踪算法的输入，即当前帧的搜索区域(`search_region`)和参考模板(`template`)。这有助于理解模型是如何在每一帧中寻找目标的。
  ```python
  self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
  self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
  ```

### 热图可视化
- **得分图**：将模型输出的得分图(`score_map`)以热图形式展示，可以看到模型预测目标位置的置信度分布。
            叠加汉宁窗(`score_map_hann`)的得分图可以观察到窗函数如何影响置信度分布，从而理解模型偏好的区域。
  ```python
  self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
  self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')
  ```

### 掩码图像可视化
- **被模型忽略的区域**：如果`removed_indexes_s`存在，意味着模型在某些区域的特征被忽略了。通过在搜索区域图像上应用一个掩码来表示这些区域，可以直观地看到模型忽视的部分。
                    这对于理解模型的注意力集中点以及可能的优化方向非常有价值。
  ```python
  if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
      masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
      self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
  ```

### 交互和控制
代码中的`while self.pause_mode:`循环以及与之相关的`if self.step:`条件判断提供了一种交互式控制机制。这使得用户可以在实验过程中暂停、检查可视化结果，并在需要时逐步前进。
                                这是一个非常有用的功能，特别是在调试模型或深入分析特定帧时。

总的来说，Visdom 的这种用法提供了一种灵活且强大的方式来实时监控和分析模型的性能和行为，特别是在复杂的任务如目标跟踪中。
通过丰富的可视化支持，研究人员和开发人员可以更深入地理解他们的模型，并及时调整策略以改进结果。
"""