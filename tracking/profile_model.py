import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import torch
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='procontext', choices=['procontext'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='cmc3', help='yaml configure file name')
    args = parser.parse_args()

    return args


def evaluate_vit(model, template, search):
    '''Speed Test
    评估给定模型的性能，包括计算复杂度（MACs）和参数量，以及测试模型的执行速度。
    
    参数:
    - model: 要评估的模型。
    - template: 模型输入的模板图像或数据。
    - search: 模型输入的搜索图像或数据。
    
    返回值:
    无。函数主要通过打印输出来展示评估结果。
    '''
    macs1, params1 = profile(model, inputs=(template, search),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 500
    T_t = 1000
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for _ in range(T_w):
            _ = model(template, search)
        start = time.time()
        for _ in range(T_t):
            _ = model(template, search)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))
        # for i in range(T_w):
        #     _ = model(template, search)
        # start = time.time()
        # for i in range(T_t):
        #     _ = model(template, search)
        # end = time.time()
        # avg_lat = (end - start) / T_t
        # print("The average backbone latency is %.2f ms" % (avg_lat * 1000))


def evaluate_vit_separate(model, template, search):
    '''Speed Test
    
    评估模型的处理速度。
    
    参数:
    - model: 使用的模型，应具有前向传播骨干网络。
    - template: 模板图像，用于模型处理。
    - search: 查询图像，用于与模板图像进行比较。
    
    无返回值，但会打印出平均总体延迟时间。
    '''
    T_w = 50
    T_t = 1000
    print("testing speed ...")
    z = model.forward_backbone(template, image_type='template')
    x = model.forward_backbone(search, image_type='search')
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        start = time.time()
        for i in range(T_t):
            _ = model.forward_backbone(search, image_type='search')
            _ = model.forward_cat(z, x)
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))


def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    att_mask = torch.rand(bs, sz, sz) > 0.5
    return NestedTensor(img_patch, att_mask)


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    has_infrared = cfg.TRAIN.HAS_INFRARED

    if args.script == "procontext":
        model_module = importlib.import_module('lib.models')
        model_constructor = model_module.build_procontext
        model = model_constructor(cfg, training=False)
        # get the template and search
        template = torch.randn(bs, 3, z_sz, z_sz)
        
        
        search = torch.randn(bs, 3, x_sz, x_sz)
        # transfer to device
        model = model.to(device)
        template = template.to(device)
        if has_infrared:
            print("Double modal")
            template_i = torch.randn(bs, 3, z_sz, z_sz)
            template_i = template_i.to(device)
            template = [template, template_i]
        search = search.to(device)

        merge_layer = cfg.MODEL.BACKBONE.MERGE_LAYER
        if merge_layer <= 0:
            evaluate_vit(model, template, search)
        else:
            evaluate_vit_separate(model, template, search)

    else:
        raise NotImplementedError
