import torch


class AvgMeter:
    """用于计算平均值的工具类，用于记录训练过程中的平均损失"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置计数器"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        更新平均值
        
        Args:
            val: 当前值（通常是损失值）
            n: 样本数量（通常是batch_size）
        """
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
    
    def show(self):
        """返回当前平均值"""
        return self.avg


def adjust_lr(optimizer, base_lr, epoch, decay_rate, decay_epoch):
    """
    根据epoch调整学习率
    
    学习率衰减策略：每经过decay_epoch个epoch，学习率乘以decay_rate
    
    Args:
        optimizer: 优化器
        base_lr: 基础学习率
        epoch: 当前epoch
        decay_rate: 学习率衰减率
        decay_epoch: 学习率衰减的epoch间隔
    """
    # 计算衰减次数
    decay_times = epoch // decay_epoch
    
    # 计算新的学习率
    new_lr = base_lr * (decay_rate ** decay_times)
    
    # 更新优化器的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

