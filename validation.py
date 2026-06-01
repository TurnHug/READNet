import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from configs import config


def validation(model, test_size, sw, validation_step):
    """
    验证函数，计算模型在验证集上的MAE（平均绝对误差）
    
    Args:
        model: 训练好的模型
        test_size: 测试图像大小
        sw: tensorboardX的SummaryWriter，用于记录验证结果
        validation_step: 当前验证步数
        
    Returns:
        mae: 平均绝对误差
        validation_step: 更新后的验证步数
    """
    # 设置模型为评估模式
    model.eval()
    
    # 创建验证数据集（使用训练数据，但不进行数据增强）
    # 使用训练数据集的路径，但创建一个不进行数据增强的版本
    val_config = config.copy()
    val_config['train_size'] = test_size
    
    # 创建验证数据集（不进行数据增强）
    # Use 'val_root' for validation data path
    image_root = config['val_root']
    val_dataset = ImageDataVal(image_root, test_size, config['means'], config['stds'])
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,  # 验证时使用batch_size=1
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    mae_sum = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, pack in enumerate(val_loader):
            images = pack['sal_image'].cuda()
            gts = pack['sal_label'].cuda()
            
            # 模型推理
            side_out1, side_out2, side_out3, side_out4 = model(images)
            
            # 使用side_out1作为最终预测结果
            pred = torch.sigmoid(side_out1)
            
            # 将预测结果和真实标签调整到相同大小
            pred = F.interpolate(pred, size=gts.shape[2:], mode='bilinear', align_corners=False)
            
            # 计算MAE
            mae = torch.abs(pred - gts).mean()
            mae_sum += mae.item()
            total_samples += 1
    
    # 计算平均MAE
    avg_mae = mae_sum / total_samples if total_samples > 0 else 0.0
    
    # 记录到tensorboard
    sw.add_scalar('validation/mae', avg_mae, global_step=validation_step)
    validation_step += 1
    
    # 恢复训练模式（虽然这里不会立即使用，但保持状态一致）
    model.train()
    
    return avg_mae, validation_step


class ImageDataVal:
    """验证数据集类，不进行数据增强"""
    def __init__(self, data_root, testsize, means, stds):
        self.testsize = testsize
        # 确保路径末尾有斜杠
        data_root_normalized = data_root.rstrip('/') + '/'
        image_root = data_root_normalized + 'input/'
        gt_root = data_root_normalized + 'gt/'
        
        # 获取图像和标签文件列表
        self.images = [image_root + f for f in os.listdir(image_root) 
                      if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) 
                   if f.endswith('.jpg') or f.endswith('.png')]
        
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.sal_num = len(self.images)
        
        # 图像变换（不进行数据增强）
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])
        
        # 标签变换
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __getitem__(self, index):
        # 加载图像和标签
        sal_image = self.load_image(self.images[index])
        sal_label = self.load_sal_label(self.gts[index])
        
        # 只进行resize和tensor转换，不进行数据增强
        sal_image = self.img_transform(sal_image)
        sal_label = self.gt_transform(sal_label)
        
        sample = {'sal_image': sal_image, 'sal_label': sal_label}
        return sample
    
    def __len__(self):
        return self.sal_num
    
    def load_sal_label(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def load_image(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

