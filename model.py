
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilevit_v3.mobilevit import get_mobilevit_v3_s

class DiffConvUtils:
    @staticmethod
    def get_cdc_kernel(weight, theta=0.7):
        weight_sum = weight.sum(dim=[2, 3], keepdim=True)
        kernel_diff = weight.clone()
        kernel_diff[:, :, 1, 1] -= weight_sum.squeeze(-1).squeeze(-1) * theta
        return kernel_diff

    @staticmethod
    def get_adc_kernel(weight, theta=0.7):
        shape = weight.shape
        w_flat = weight.view(shape[0], shape[1], -1)
        indices = [3, 0, 1, 6, 4, 2, 7, 8, 5]
        w_rotated = w_flat[:, :, indices]
        w_new = w_flat - theta * w_rotated
        return w_new.view(shape)

class RepEPDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 dilation=1, groups=1, bias=False, theta=0.7, deploy=False):
        super(RepEPDC, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.theta = theta
        self.deploy = deploy
        
        # PyTorch 卷积权重形状逻辑: [Out, In/Groups, K, K]
        self.kernel_dim_in = in_channels // groups


        if deploy:
            # 部署模式：单层卷积
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                         stride=stride, padding=padding, 
                                         dilation=dilation, groups=groups, bias=bias)
        else:
            # 训练模式：多分支结构
            
            # 1. Vanilla 分支
            self.conv_vanilla = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                                          stride=stride, padding=padding, 
                                          dilation=dilation, groups=groups, 
                                          bias=False)
            
            # 2. CDC 分支
            self.weight_cdc = nn.Parameter(torch.Tensor(out_channels, self.kernel_dim_in, kernel_size, kernel_size))
            
            # 3. ADC 分支
            self.weight_adc = nn.Parameter(torch.Tensor(out_channels, self.kernel_dim_in, kernel_size, kernel_size))

            # 偏置
            self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
            
            # 学习率参数 alpha
            self.alpha_logits = nn.Parameter(torch.randn(3, out_channels, 1, 1))
            self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv_vanilla.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_cdc, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.weight_adc, mode='fan_out', nonlinearity='relu')
        
        with torch.no_grad():
            self.alpha_logits[0].fill_(math.log(0.5))
            self.alpha_logits[1].fill_(math.log(0.3))
            self.alpha_logits[2].fill_(math.log(0.2))

    def get_equivalent_params(self):
        alpha_probs = F.softmax(self.alpha_logits, dim=0).detach()
        
        # 获取权重 [Out, In/Groups, K, K]
        k_v = self.conv_vanilla.weight.detach()
        k_c = DiffConvUtils.get_cdc_kernel(self.weight_cdc.detach(), self.theta)
        k_a = DiffConvUtils.get_adc_kernel(self.weight_adc.detach(), self.theta)
        
        a_v = alpha_probs[0].view(-1, 1, 1, 1)
        a_c = alpha_probs[1].view(-1, 1, 1, 1)
        a_a = alpha_probs[2].view(-1, 1, 1, 1)
        
        # 加权融合基础三分支
        final_weight = (k_v * a_v) + (k_c * a_c) + (k_a * a_a)
        final_bias = self.bias.detach() if self.bias is not None else torch.zeros(self.out_channels, device=final_weight.device)


        return final_weight, final_bias

    def switch_to_deploy(self):
        if self.deploy: return
        device = next(self.parameters()).device
        final_weight, final_bias = self.get_equivalent_params()
        
        # 即使原始没有 bias，融合 BN 后通常会有 bias，所以 bias=True
        self.reparam_conv = nn.Conv2d(self.in_channels, self.out_channels, 
                                     kernel_size=self.kernel_size, stride=self.stride, 
                                     padding=self.padding, dilation=self.dilation, 
                                     groups=self.groups, bias=True).to(device)
        
        self.reparam_conv.weight.data = final_weight
        self.reparam_conv.bias.data = final_bias
        
        # 删除训练参数
        for param_name in ['conv_vanilla', 'weight_cdc', 'weight_adc', 'bias', 'alpha_logits', 'identity_bn']:
            if hasattr(self, param_name): delattr(self, param_name)
        self.deploy = True

    def forward(self, x):
        if self.deploy: return self.reparam_conv(x)
        
        alpha_probs = F.softmax(self.alpha_logits, dim=0)
        
        # 1. Vanilla
        y_v = self.conv_vanilla(x)
        
        # 2. CDC
        k_c = DiffConvUtils.get_cdc_kernel(self.weight_cdc, self.theta)
        y_c = F.conv2d(x, k_c, None, self.stride, self.padding, self.dilation, self.groups)
        
        # 3. ADC
        k_a = DiffConvUtils.get_adc_kernel(self.weight_adc, self.theta)
        y_a = F.conv2d(x, k_a, None, self.stride, self.padding, self.dilation, self.groups)
        
        out = (y_v * alpha_probs[0]) + (y_c * alpha_probs[1]) + (y_a * alpha_probs[2])
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
            
        return out


class RepDSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RepDSConv, self).__init__()
        
        self.depthwise = RepEPDC(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
            padding=1, 
            bias=False, 
            deploy=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.SiLU()

        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()

    def forward(self, x):
        # 空间提取 (含 Vanilla/CDC/ADC + Identity)
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # 通道融合
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def switch_to_deploy(self):
        self.depthwise.switch_to_deploy()


class MultiScaleRepEPDC(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=False):
        super().__init__()
        self.stride = stride
        
 
        assert in_channels % 4 == 0
        group_width = in_channels // 4
        
        # 分支 1: 1x1 变换 (保留局部细节)
        self.branch1 = nn.Sequential(
            nn.Conv2d(group_width, group_width, 1, bias=False),
            nn.BatchNorm2d(group_width),
            nn.SiLU()
        )
        
        # 分支 2: RepEPDC (Dilation=1)
        self.branch2 = nn.Sequential(RepEPDC(group_width, group_width, kernel_size=3, stride=stride, 
                               padding=1, dilation=1, bias=False, deploy=deploy),
                               nn.BatchNorm2d(group_width),
                               nn.SiLU())
        
        # 分支 3: RepEPDC (Dilation=2) -> 感受野变大
        self.branch3 = nn.Sequential(RepEPDC(group_width, group_width, kernel_size=3, stride=stride, 
                               padding=2, dilation=2, bias=False, deploy=deploy),
                               nn.BatchNorm2d(group_width),
                               nn.SiLU())
        
        # 分支 4: RepEPDC (Dilation=3) -> 感受野更大
        self.branch4 = nn.Sequential(RepEPDC(group_width, group_width, kernel_size=3, stride=stride, 
                               padding=3, dilation=3, bias=False, deploy=deploy),
                               nn.BatchNorm2d(group_width),
                               nn.SiLU())
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        # 1. 通道切分 (Channel Splitting)
        splits = torch.chunk(x, 4, dim=1)
        
        # 2. 多尺度处理
        x1 = self.branch1(splits[0])
        x2 = self.branch2(splits[1])
        x3 = self.branch3(splits[2])
        x4 = self.branch4(splits[3])
        
        # 3. 拼接 (Concatenation)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        
        # 4. 融合
        out = self.fusion(out)
        return out
    
    def switch_to_deploy(self):
        for branch in [self.branch2, self.branch3, self.branch4]:
            for m in branch.modules():
                if m is branch:
                    continue
                if hasattr(m, 'switch_to_deploy'):
                    m.switch_to_deploy()

class VarianceChannelGate(nn.Module):
    """
    子模块A: 基于方差统计的通道门控 (全卷积实现)
    改进点：
    1. 使用 1x1 Conv 替代 Linear，避免维度重排 (Reshape) 开销。
    2. 使用 BatchNorm 替代 LayerNorm，推理时可融合，速度更快。
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        # 降维比例
        mid_channels = max(channels // reduction, 8)
        
        # 使用 1x1 卷积构建 MLP
        # 结构：Conv1x1 -> BN -> SiLU -> Conv1x1 -> Sigmoid
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels), # BN 替代 LN，推理时融合加速
            nn.SiLU(),                    # 保持 Qwen 的激活风格
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        
        # Mean: [B, C, 1, 1]
        avg_pool = x.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        
        # Std: [B, C, 1, 1] (创新点：标准差池化)
        std_pool = x.view(b, c, -1).std(dim=2).view(b, c, 1, 1)
        
        #  拼接 (Channel 维度拼接) -> [B, 2C, 1, 1]
        stat_cat = torch.cat([avg_pool, std_pool], dim=1)
        
        # 生成门控权重 (全程卷积，无需 view)
        # Input: [B, 2C, 1, 1] -> Output: [B, C, 1, 1]
        gate = self.mlp(stat_cat)
        
        # 4. 广播乘法
        return x * gate

class ContextSpatialGate(nn.Module):
    """
    子模块B: 上下文差分空间门控
    区别于CBAM：不使用简单的Conv，而是利用 Group Conv 提取轻量级上下文差异
    """
    def __init__(self, channels):
        super().__init__()
        # 轻量化：使用 Group Convolution 减少参数
        self.group_conv = nn.Sequential(
            RepEPDC(in_channels=channels, out_channels=channels, 
                   kernel_size=3, stride=1, padding=2, 
                   groups=channels, bias=False, dilation=2, deploy=False),
            nn.BatchNorm2d(channels),
            nn.SiLU() 
        )
        
        # 这是一个 1x1 卷积，用于整合信息生成 Mask
        self.mask_gen = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        # 1. 提取局部上下文特征
        context = self.group_conv(x)
        
        # 2. 计算“特征-上下文”差异 (Difference) - 创新点
        # 如果一个像素是被遮挡的(如雾)，它和周围环境的差异模式会与显著性物体不同
        diff = x - context 
        
        # 3. 生成空间 Mask
        # 使用 HardSigmoid 产生更稀疏的激活 (接近0或1)，模拟“Drop”掉恶劣区域
        mask = F.hardsigmoid(self.mask_gen(diff))
        
        return x * mask

class CADG(nn.Module):
    """
    整体模块: Condition-Adaptive Dual-Gating
    """
    def __init__(self, channels):
        super().__init__()
        self.channel_gate = VarianceChannelGate(channels)
        self.spatial_gate = ContextSpatialGate(channels)
        
        # 可学习的缩放参数，类似 ResNet 的 Gamma
        self.gamma_logit = nn.Parameter(torch.tensor([-2.5]))

    def forward(self, x):
        residual = x
        
        # 串行处理：先滤除坏通道，再定位好像素
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        gamma = torch.sigmoid(self.gamma_logit)
        out = (1 - gamma) * residual + gamma * x
        return out
class RepInvertedBlock(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0., deploy=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 1. Point-wise Expansion
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.act = act_layer()

        # 2. Depth-wise Feature Extraction (Core Innovation: RepEPDC)
        # Replaced with MultiScaleRepEPDC as requested
        self.conv = MultiScaleRepEPDC(
            in_channels=hidden_features,
            out_channels=hidden_features,
            stride=1,
            deploy=deploy
        )
        self.bn2 = nn.BatchNorm2d(hidden_features)

        # 3. Point-wise Projection
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        res = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        
        x = self.conv(x)
        x = self.bn2(x)
        x = self.act(x)
        
        x = self.fc2(x)
        x = self.bn3(x)
        
        if self.drop.p > 0:
            x = self.drop(x)
            
        return x + res

    def switch_to_deploy(self):
        if hasattr(self.conv, 'switch_to_deploy'):
            self.conv.switch_to_deploy()


class CADGInteraction(nn.Module):

    def __init__(self, dim, low_dim, high_dim, bias=True):
        super().__init__()
        # 1. 对齐深层特征 (High) 到 当前层 (Low)
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_dim, dim, 1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        
        # 2. 对齐浅层特征 (Low)
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_dim, dim, 1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )

        # 3. 核心门控 (Core Innovation: CADG)
        self.gate = CADG(dim)

    def forward(self, x_low, x_high):
        # x_low: [B, C_l, H, W]
        # x_high: [B, C_h, H/2, W/2]
        
        x_high = self.high_conv(x_high)
        x_high_up = F.interpolate(x_high, size=x_low.shape[2:], mode='bilinear', align_corners=False)
        
        x_low = self.low_conv(x_low)
        x_low_filtered = self.gate(x_low)
        
        out = x_low_filtered + x_high_up
        
        return out


class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, bias=True, group=1, dilation=1,
                 act=nn.SiLU()):
        super(CBR, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias, groups=group,
                      dilation=dilation),
            nn.BatchNorm2d(out_channel),
            act)

    def forward(self, x):
        return self.conv(x)


class READNet(nn.Module):
    def __init__(self, deploy=False):
        super().__init__()
        self.backbone = get_mobilevit_v3_s(load=True)
        print("backbone loaded.32,64,128,256,320")
        # --- High Level Interaction ---
        self.cadg_inter5_4 = CADGInteraction(dim=256, low_dim=256, high_dim=320)
        self.cadg_inter4_3 = CADGInteraction(dim=128, low_dim=128, high_dim=256)
        self.cadg_inter3_2 = CADGInteraction(dim=64, low_dim=64, high_dim=128)
        self.cadg_inter2_1 = CADGInteraction(dim=32, low_dim=32, high_dim=64)

        # Decoder Stages
        # x4_2 -> match x2(64)
        self.dim_conv4_3 = nn.Conv2d(256, 128, 1)
        self.dim_conv3_2 = nn.Conv2d(128, 64, 1)
        self.dim_conv2_1 = nn.Conv2d(64, 32, 1)

        self.rep_irb4_3 = RepInvertedBlock(128, 128 * 2, 128, deploy=deploy)

        self.rep_irb3_2 = RepInvertedBlock(64, 64 * 2, 64, deploy=deploy)
        
        self.rep_irb2_1 = RepInvertedBlock(32, 32 * 2, 32, deploy=deploy)
        
        # Restoring dims for fusion
        self.dim_conv2 = nn.Conv2d(128, 64, 1)
        self.dim_conv3 = nn.Conv2d(64, 32, 1)

        self.sideout_4 = nn.Conv2d(256, 1, 1)
        self.sideout_3 = nn.Conv2d(128, 1, 1)
        self.sideout_2 = nn.Conv2d(64, 1, 1)
        self.sideout_1 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        w, h = x.size()[2:]
        feats  = self.backbone(x)
        _, x1, x2, x3, x4, x5, _ = feats
        # 1. Interaction
        x5_4 = self.cadg_inter5_4(x4, x5) # 256
        
        x4_3 = self.cadg_inter4_3(x3, x4) # 128

        x3_2 = self.cadg_inter3_2(x2, x3) # 64
        x2_1 = self.cadg_inter2_1(x1, x2) # 32


        # Stage 4: Fusion & Refinement
        # x5_4 (256) -> dim_conv4_3 -> upsample -> + x4_3 (128)
        x4_3_feat = x4_3 + F.interpolate(self.dim_conv4_3(x5_4), size=x4_3.shape[2:], mode='bilinear', align_corners=False)
        x4_3_feat = self.rep_irb4_3(x4_3_feat)
        
        # Stage 3: Fusion & Refinement
        # x4_3_feat (128) -> dim_conv3_2 -> upsample -> + x3_2 (64)
        x3_2_feat = x3_2 + F.interpolate(self.dim_conv3_2(x4_3_feat), size=x3_2.shape[2:], mode='bilinear', align_corners=False)
        x3_2_feat = self.rep_irb3_2(x3_2_feat)
        
        # Stage 2: Fusion & Refinement
        # x3_2_feat (64) -> dim_conv2_1 -> upsample -> + x2_1 (32)
        x2_1_feat = x2_1 + F.interpolate(self.dim_conv2_1(x3_2_feat), size=x2_1.shape[2:], mode='bilinear', align_corners=False)
        x2_1_feat = self.rep_irb2_1(x2_1_feat)

        # Sideouts
        side_out4 = self.sideout_4(x5_4)
        side_out3 = self.sideout_3(x4_3_feat)
        side_out2 = self.sideout_2(x3_2_feat)
        side_out1 = self.sideout_1(x2_1_feat)
        
        final_out = F.interpolate(side_out1, size=(w, h), mode='bilinear', align_corners=False)
        return final_out, side_out2, side_out3, side_out4

    def switch_to_deploy(self):
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model1 = READNet().to(device)
    def measure_fps(model, device='cuda', input_size=(1, 3, 352, 352), warmup=50, iters=200):
        model.eval()
        model.to(device)

        x = torch.randn(input_size).to(device)

        # 预热（非常重要）
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            for _ in range(iters):
                _ = model(x)

        torch.cuda.synchronize()
        end = time.time()

        fps = iters / (end - start)
        return fps
    # 切换部署并测试
    model1.eval()
    model1.switch_to_deploy()
    fps = measure_fps(model1)
    print(f"FPS: {fps:.2f}")
    dummy_input = torch.randn(2, 3, 352, 352).to(device)

    print("\n--- Model Forward Check ---")

    p_final= model1(dummy_input)
    print(f"Training Mode Outputs:")
    print(f"Final Prediction: {p_final[0].shape}")

            

    p_deploy = model1(dummy_input)
    if isinstance(p_deploy, tuple):
        print(f"Deploy Mode Output: {p_deploy[0].shape}")
    else:
        print(f"Deploy Mode Output: {p_deploy.shape}")
            
    # 计算最终部署模型的参数量
    params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f"\nHASNetS Total Deploy Parameters: {params / 1e6:.2f} M")

    from ptflops import get_model_complexity_info
    
    # Calculate FLOPs and Params using ptflops
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model1, (3, 352, 352), as_strings=False,
                                             print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Number of parameters: ', f'{params/1e6:.2f} M'))
      print('{:<30}  {:<8}'.format('Computational complexity (MACs): ', f'{macs/1e9:.2f} GMac'))
      print('{:<30}  {:<8}'.format('Computational complexity (FLOPs): ', f'{2 * macs/1e9:.2f} GFLOPs'))


    