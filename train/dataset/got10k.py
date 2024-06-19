import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class Got10k(BaseVideoDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().got10k_dir if root is None else root
        super().__init__('GOT10k', root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        # if split is not None:
        #     if seq_ids is not None:
        #         raise ValueError('Cannot set both split_name and seq_ids.')
        #     ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        #     if split == 'train':
        #         file_path = os.path.join(ltr_path, 'data_specs', 'got10k_train_split.txt')
        #     elif split == 'val':
        #         file_path = os.path.join(ltr_path, 'data_specs', 'got10k_val_split.txt')
        #     elif split == 'train_full':
        #         file_path = os.path.join(ltr_path, 'data_specs', 'got10k_train_full_split.txt')
        #     elif split == 'vottrain':
        #         file_path = os.path.join(ltr_path, 'data_specs', 'got10k_vot_train_split.txt')
        #     elif split == 'votval':
        #         file_path = os.path.join(ltr_path, 'data_specs', 'got10k_vot_val_split.txt')
        #     else:
        #         raise ValueError('Unknown split name.')
        #     # seq_ids = pandas.read_csv(file_path, header=None, squeeze=True, dtype=np.int64).values.tolist()
        #     seq_ids = pandas.read_csv(file_path, header=None, dtype=np.int64).squeeze("columns").values.tolist()
        # elif seq_ids is None:
        #     seq_ids = list(range(0, len(self.sequence_list)))

        seq_ids = list(range(len(self.sequence_list)))
        print(f"Length of sequence_list: {len(self.sequence_list)}")
        print(f"self.sequence_list: {self.sequence_list}")
        print(f"Sequence IDs: {seq_ids}")
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'got10k'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            # print(f"seq_path: {seq_path}")
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                # print("check1")
                meta_info = f.readlines()
                # print("check2")
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]
                                       })
            # print("check3")
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        # print(f"object_meta: {object_meta}")
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    # def get_sequence_info(self, seq_id):
    #     seq_path = self._get_sequence_path(seq_id)
    #     bbox = self._read_bb_anno(seq_path)

    #     valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
    #     visible, visible_ratio = self._read_target_visible(seq_path)
    #     visible = visible & valid.byte()

    #     return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)

        # 检查是否同时存在可见与红外
        visible_path = os.path.join(seq_path, 'visible')
        infrared_path = os.path.join(seq_path, 'infrared')

        has_visible = os.path.exists(visible_path)
        has_infrared = os.path.exists(infrared_path)

        bbox = {'visible': [], 'infrared': []}
        visible = {'visible': [], 'infrared': []}
        visible_ratio = {'visible': [], 'infrared': []}

        if has_infrared and has_visible:
            bbox['visible'] = self._read_bb_anno(visible_path)
            bbox['infrared'] = self._read_bb_anno(infrared_path)
            visible['visible'], visible_ratio['visible'] = self._read_target_visible(visible_path)
            visible['infrared'], visible_ratio['infrared'] = self._read_target_visible(infrared_path)
            # 如果边界框的宽度和高度都大于0，则相应的元素为 True，表示这个边界框是有效的（即宽度和高度都非零，目标被视为在帧中存在）。
            valid = (bbox['visible'][:, 2] > 0) & (bbox['visible'][:, 3] > 0) & (bbox['infrared'][:, 2] > 0) & (bbox['infrared'][:, 3] > 0)
            # 这个操作确保一个目标只有在同时满足两个条件时才被认为是可见的：它在原始数据中被标记为可见（visible），且其对应的边界框是有效的（即宽度和高度都大于0，由 valid 表示）。
            visible = visible['visible'] & visible['infrared'] & valid.byte()
        else:
            bbox['visible'] = self._read_bb_anno(seq_path)
            visible['visible'], visible_ratio['visible'] = self._read_target_visible(seq_path)
            valid = (bbox['visible'][:, 2] > 0) & (bbox['visible'][:, 3] > 0)
            visible = visible['visible'] & valid.byte()
            
        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id):
        #print(os.path.join(seq_path, '{:08}.jpg'.format(frame_id+1)))
        return os.path.join(seq_path, '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        #print(seq_path)
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    # def get_frames(self, seq_id, frame_ids, anno=None):
    #     seq_path = self._get_sequence_path(seq_id)
    #     obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

    #     frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

    #     if anno is None:
    #         anno = self.get_sequence_info(seq_id)

    #     anno_frames = {}
    #     for key, value in anno.items():
    #         anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

    #     return frame_list, anno_frames, obj_meta
    # -----------------------------------------------------------------------------------------------------------
    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]
        #print(f'frame_ids: {frame_ids}')
        # 检查是否同时存在可见与红外
        visible_path = os.path.join(seq_path, 'visible')
        infrared_path = os.path.join(seq_path, 'infrared')

        has_visible = os.path.exists(visible_path)
        has_infrared = os.path.exists(infrared_path)

        frame_list = {'visible': [], 'infrared': []}  # 字典

        for f_id in frame_ids:
            if has_infrared and has_visible:
                frame_list['visible'].append(self._get_frame(visible_path, f_id))
                frame_list['infrared'].append(self._get_frame(infrared_path, f_id))
            else:
                frame_list['visible'].append(self._get_frame(seq_path, f_id))
        #print('anno: ', anno)
        if anno is None:
            anno = self.get_sequence_info(seq_id)
            #print('anno: ', anno)

        anno_frames = { 'visible_bbox': {},
                        'visible_visible_ratio': {}, 
                        'infrared_bbox': {},
                        'infrared_visible_ratio': {},
                        'valid': {},
                        'visible': {}}
        # 遍历注释中的每个键（如 'bbox', 'valid', 'visible', 'visible_ratio'）
        image_types = ['visible', 'infrared'] if has_infrared else {'visible'}
        # print(image_types)
        #print('anno.keys(): ', anno.keys()) # dict_keys(['bbox', 'valid', 'visible', 'visible_ratio'])
        for key in anno.keys():
            # 检查每种图像类型
            #print('key: ', key)
            for img_type in image_types:
                #print(f'anno{[key]}: ', anno[key])
                #print(f'anno{[key][img_type]}: ', anno[key][img_type])
                # 如果这种类型的图像存在，则提取对应的帧注释
                # if img_type in anno[key] and len(anno[key][img_type] > 0):
                if isinstance(anno[key], dict):
                    # print(anno[key][img_type])
                    # print(len(anno[key][img_type]))
                    #print('0.1')
                    try:

                        # print(f'type: {type(anno[key][img_type])}')
                        # print(f'len: {len(anno[key][img_type])}')
                        # print(f'anno_frames[{img_type}]: {anno_frames[img_type]}')
                        anno_frames[img_type + '_' + key] = [anno[key][img_type][f_id, ...].clone() for f_id in frame_ids]  # 报错行$$$$$$$$$$$$$$$$$
                        #anno_frames[img_type][key] = [0, 0]
                        # print('0.2')
                        #print(f'anno_frames{[img_type][key]}: {anno_frames[img_type][key]}')
                    except Exception as e:
                        print(f"Assignment failed with error: {e}")
                # elif key == 'visible_ration':
                #     try:
                #         print('0.3')
                #         print(f'anno_frames[img_type]: {anno_frames[img_type]}')
                #         print('0.4')
                #         print(f'anno_frames[img_type][key]: {anno_frames[img_type][key]}')
                #     except Exception as e:
                #         print(f"Assignment failed with error: {e}")
                else:
                    #print('0.3')
                    try:
                        anno_frames[key] = [anno[key][f_id, ...].clone() for f_id in frame_ids]
                        #print(f'anno_frames{[key]}: {anno_frames[key]}')
                    except Exception as e:
                        print(f"Assignment failed with error: {e}")
                #print('1')
            #print('2')
        # print(f'anno_frames: {anno_frames}')
        return frame_list, anno_frames, obj_meta
        # anno_frames: {'visible_bbox': [tensor([794., 399., 334., 364.])], 'visible_visible_ratio': [tensor(0.7500)], 
        #               'infrared_bbox': {}, 'infrared_visible_ratio': {}, 'valid': [tensor(True)], 'visible': [tensor(1, dtype=torch.uint8)]}
        
                    # print(anno[key][img_type])
                    
                    # print(len(anno[key][img_type]))