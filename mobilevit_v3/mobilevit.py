#
# For licensing see accompanying LICENSE file.
#
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
from torch import nn
import argparse
from typing import Dict, Tuple, Optional

from utils import logger
# from models.classification import register_cls_models
from mobilevit_v3.config.mobilevit import get_configuration
from layers import ConvLayer, LinearLayer, GlobalPool, Dropout
from mobilevit_block import InvertedResidual, MobileViTv3Block
import torch
import yaml


class YAMLOptions:
    def __init__(self, yaml_file):
        with open(yaml_file, 'r') as file:
            self.data = yaml.safe_load(file)

    def __getattr__(self, name):
        try:
            properties = name.split('.')
            value = self.data
            for prop in properties:
                value = value[prop]
            return value
        except (KeyError, TypeError):
            raise AttributeError(f"'YAMLOptions' object has no attribute '{name}'")


# @register_cls_models("mobilevit_v3")
class MobileViTv3(nn.Module):
    """
        MobileViTv3:
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout", 0.2)

        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        image_channels = 3
        out_channels = 16

        mobilevit_config = get_configuration(opts=opts)

        # Segmentation architectures like Deeplab and PSPNet modifies the strides of the classification backbones
        # We allow that using `output_stride` arguments
        output_stride = kwargs.get("output_stride", None)
        dilate_l4 = dilate_l5 = False
        if output_stride == 8:
            dilate_l4 = True
            dilate_l5 = True
        elif output_stride == 16:
            dilate_l5 = True

        super(MobileViTv3, self).__init__()
        self.dilation = 1

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            opts=opts, in_channels=image_channels, out_channels=out_channels,
            kernel_size=3, stride=2, use_norm=True, use_act=True
        )

        self.model_conf_dict['conv1'] = {'in': image_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer1"]
        )
        self.model_conf_dict['layer1'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer2"]
        )
        self.model_conf_dict['layer2'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer3"]
        )
        self.model_conf_dict['layer3'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer4"], dilate=dilate_l4
        )
        self.model_conf_dict['layer4'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            opts=opts, input_channel=in_channels, cfg=mobilevit_config["layer5"], dilate=dilate_l5
        )
        self.model_conf_dict['layer5'] = {'in': in_channels, 'out': out_channels}

        in_channels = out_channels
        exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            opts=opts, in_channels=in_channels, out_channels=exp_channels,
            kernel_size=1, stride=1, use_act=True, use_norm=True
        )

        self.model_conf_dict['exp_before_cls'] = {'in': in_channels, 'out': exp_channels}

        self.classifier = nn.Sequential()
        self.classifier.add_module(name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False))
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(name="dropout", module=Dropout(p=classifier_dropout, inplace=True))
        self.classifier.add_module(
            name="fc",
            module=LinearLayer(in_features=exp_channels, out_features=num_classes, bias=True)
        )

        # check model
        # self.check_model()

        # weight initialization
        # self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title="".format(cls.__name__), description="".format(cls.__name__))
        group.add_argument('--model.classification.mit.mode', type=str, default=None,
                           choices=['xx_small', 'x_small', 'small'], help="MIT mode")
        group.add_argument('--model.classification.mit.attn-dropout', type=float, default=0.1,
                           help="Dropout in attention layer")
        group.add_argument('--model.classification.mit.ffn-dropout', type=float, default=0.0,
                           help="Dropout between FFN layers")
        group.add_argument('--model.classification.mit.dropout', type=float, default=0.1,
                           help="Dropout in Transformer layer")
        group.add_argument('--model.classification.mit.transformer-norm-layer', type=str, default="layer_norm",
                           help="Normalization layer in transformer")
        group.add_argument('--model.classification.mit.no-fuse-local-global-features', action="store_true",
                           help="Do not combine local and global features in MIT block")
        group.add_argument('--model.classification.mit.conv-kernel-size', type=int, default=3,
                           help="Kernel size of Conv layers in MIT block")

        group.add_argument('--model.classification.mit.head-dim', type=int, default=None,
                           help="Head dimension in transformer")
        group.add_argument('--model.classification.mit.number-heads', type=int, default=None,
                           help="No. of heads in transformer")
        return parser

    def _make_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg,
                dilate=dilate
            )
        else:
            return self._make_mobilenet_layer(
                opts=opts,
                input_channel=input_channel,
                cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(opts, input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def _make_mit_layer(self, opts, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[
        nn.Sequential, int]:
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                opts=opts,
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            logger.error("Transformer input dimension should be divisible by head dimension. "
                         "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(
            MobileViTv3Block(
                opts=opts,
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=getattr(opts, "model.classification.mit.dropout", 0.1),
                ffn_dropout=getattr(opts, "model.classification.mit.ffn_dropout", 0.0),
                attn_dropout=getattr(opts, "model.classification.mit.attn_dropout", 0.1),
                head_dim=head_dim,
                no_fusion=getattr(opts, "model.classification.mit.no_fuse_local_global_features", False),
                conv_ksize=getattr(opts, "model.classification.mit.conv_kernel_size", 3)
            )
        )

        return nn.Sequential(*block), input_channel

    def forward(self, x):
        # print(x.shape)
        # x = self.conv_1(x)
        # print('after conv1:\n', x.shape)
        # x = self.layer_1(x)
        # print('after layer1:\n', x.shape)
        # x = self.layer_2(x)
        # print('after layer2:\n', x.shape)
        # x = self.layer_3(x)
        # print('after layer3:\n', x.shape)
        # x = self.layer_4(x)
        # print('after layer4:\n', x.shape)
        # x = self.layer_5(x)
        # print('after layer5:\n', x.shape)
        # x = self.conv_1x1_exp(x)
        # print('after conv_1x1_exp:\n', x.shape)
        # x = self.classifier(x)
        # return x
        out = []
        x = self.conv_1(x)
        out.append(x)
        x = self.layer_1(x)
        out.append(x)
        x = self.layer_2(x)
        out.append(x)
        x = self.layer_3(x)
        out.append(x)
        x = self.layer_4(x)
        out.append(x)
        x = self.layer_5(x)
        out.append(x)
        # 重开工程测试
        x = self.conv_1x1_exp(x)
        out.append(x)
        # x = self.classifier(x)
        return out


def get_mobilevit_v3_xxs(load=True):
    config = YAMLOptions('./configs_weights/xxs/config.yaml')
    net = MobileViTv3(opts=config)
    if load:
        net.load_state_dict(torch.load('./configs_weights/xxs/checkpoint_ema_best.pt'))
    return net


def get_mobilevit_v3_xs(load=True):
    config = YAMLOptions('./configs_weights/xs/config.yaml')
    net = MobileViTv3(opts=config)
    if load:
        net.load_state_dict(torch.load('./configs_weights/xs/checkpoint_ema_best.pt', map_location=torch.device('cuda')))
    return net


def get_mobilevit_v3_s(load=True):
    config = YAMLOptions('./configs_weights/s/config.yaml')
    net = MobileViTv3(opts=config)
    if load:
        net.load_state_dict(torch.load('./configs_weights/s/checkpoint_ema_best.pt', map_location=torch.device('cuda')))
    return net


if __name__ == '__main__':
    # import yaml
    import torch
    from thop import profile

    config = YAMLOptions('../configs_weights/xxs/config.yaml')
    net = MobileViTv3(opts=config)
    from torchstat import stat

    #
    # stat(net, (3, 256, 256))
    from torchinfo import summary as torchinfo_summary

    torchinfo_summary(net, input_size=(1, 3, 256, 256))
