from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    raise ImportError(
        'Please run "pip install wandb" to install wandb')


class WandbWriter:
    def __init__(self, exp_name, cfg, output_dir, cur_step=0, step_interval=0):
        self.wandb = wandb
        self.step = cur_step
        self.interval = step_interval
        wandb.init(project="procontext", name=exp_name, config=cfg, dir=output_dir)

    def write_log(self, stats: OrderedDict, epoch=-1):
        self.step += 1
        for loader_name, loader_stats in stats.items():
            if loader_stats is None:
                continue

            log_dict = {}
            for var_name, val in loader_stats.items():
                if hasattr(val, 'avg'):
                    log_dict.update({loader_name + '/' + var_name: val.avg})
                else:
                    log_dict.update({loader_name + '/' + var_name: val.val})

                if epoch >= 0:
                    log_dict.update({loader_name + '/epoch': epoch})

            self.wandb.log(log_dict, step=self.step*self.interval)

    def write_image_log(self, image_data, epoch=-1):
        """
        专门用于记录图像数据到Weights & Biases的函数。
        
        参数:
        - image_data: 包含图像数据的字典，键是图像的名称，值是wandb.Image对象。
        - epoch: 当前的epoch数，用于记录。
        """
        self.step += 1  # 假设这个step是记录当前步骤的变量，根据你的需要调整

        # 直接记录图像数据
        self.wandb.log(image_data, step=self.step*self.interval)


    def attn_to_image(self, attn_matrix):
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
        plt.close(fig)
        return wandb.Image(fig)