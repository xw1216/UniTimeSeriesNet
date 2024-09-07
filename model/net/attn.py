import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, down_sample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, *args, reduction=16, **kwargs):

        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv1d(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.se(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class MRCNN(nn.Module):
    def __init__(self, cfg, afr_reduced_cnn_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        drop_rate = cfg.model.cnn_dropout

        conv_kernel_size = cfg.model.cnn_conv_kernel_size
        conv_stride = cfg.model.cnn_conv_stride
        conv_padding = cfg.model.cnn_conv_padding

        self.GELU = nn.GELU()
        # 8Hz
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64,
                      kernel_size=conv_kernel_size[0], stride=conv_stride[0],
                      padding=conv_padding[0], bias=False),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drop_rate),

            nn.Conv1d(64, 128,
                      kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128,
                      kernel_size=8, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )
        # 1Hz
        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64,
                      kernel_size=conv_kernel_size[1], stride=conv_stride[1],
                      padding=conv_padding[1], bias=False),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drop_rate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drop_rate)
        self.in_planes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        down_sample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, down_sample)]
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


##########################################################################################

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, x):
        result = super(CausalConv1d, self).forward(x)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.conv = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(query, key, value, dropout=None):
        "Implementation of Scaled dot product attention"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        n_batch = query.size(0)

        query = query.view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        key = self.conv[1](key).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        value = self.conv[2](value).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn = self.attention(query, key, value, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(n_batch, -1, self.h * self.d_k)

        return self.linear(x)


class LayerNorm(nn.Module):
    """Construct a layer normalization module."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, n):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TCE(nn.Module):
    """
    Transformer Encoder

    It is a stack of N layers.
    """

    def __init__(self, layer, n):
        super(TCE, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sub layers have residual and layer norm, implemented by SublayerOutput.
    """

    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)

    def forward(self, x_in):
        """Transformer Encoder"""
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))  # Encoder self-attention
        return self.sublayer_output[1](x, self.feed_forward)


class PositionWiseFeedForward(nn.Module):
    """Position Wise feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Implements FFN equation."""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class AttnSleep(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_tce = cfg.model.n_tce  # number of TCE clones
        d_model = cfg.model.d_model  # set to be 100 for SHHS dataset
        d_ff = cfg.model.d_ff  # dimension of feed forward
        head = cfg.model.head  # number of attention heads
        dropout = cfg.model.position_wise_dropout
        num_classes = cfg.dataset.n_class
        afr_reduced_cnn_size = cfg.model.afr_reduced_cnn_size

        self.mrcnn = MRCNN(cfg, afr_reduced_cnn_size)

        attn = MultiHeadedAttention(head, d_model, afr_reduced_cnn_size)
        ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), n_tce)

        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)

    def forward(self, x):
        x_feat = self.mrcnn(x)
        encoded_features = self.tce(x_feat)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        final_output = self.fc(encoded_features)
        return final_output


if __name__ == '__main__':
    from torchinfo import summary
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('../../conf/attn_mice.yaml')

    batch_size = cfg.train.batch_size
    model = AttnSleep(cfg=cfg)

    summary(model,
            input_size=[
                (batch_size, 1, 2000),
            ])

######################################################################

# class MRCNN_SHHS(nn.Module):
#     def __init__(self, afr_reduced_cnn_size):
#         super(MRCNN_SHHS, self).__init__()
#         drate = 0.5
#         self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
#         self.features1 = nn.Sequential(
#             nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
#             nn.BatchNorm1d(64),
#             self.GELU,
#             nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
#             nn.Dropout(drate),
#
#             nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(128),
#             self.GELU,
#
#             nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
#             nn.BatchNorm1d(128),
#             self.GELU,
#
#             nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
#         )
#
#         self.features2 = nn.Sequential(
#             nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
#             nn.BatchNorm1d(64),
#             self.GELU,
#             nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
#             nn.Dropout(drate),
#
#             nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False, padding=3),
#             nn.BatchNorm1d(128),
#             self.GELU,
#
#             nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False, padding=3),
#             nn.BatchNorm1d(128),
#             self.GELU,
#
#             nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         )
#         self.dropout = nn.Dropout(drate)
#         self.inplanes = 128
#         self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)
#
#     def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x1 = self.features1(x)
#         x2 = self.features2(x)
#         x_concat = torch.cat((x1, x2), dim=2)
#         x_concat = self.dropout(x_concat)
#         x_concat = self.AFR(x_concat)
#         return x_concat
