import torch
from torch import nn

import numpy as np


class SubConv(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_cnt = cfg.model.skip_cnt

        self.n_layer = cfg.model.conv_layer
        self.relu = nn.LeakyReLU()

        self.conv = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.drop = nn.ModuleList()

        self.skip_conv = nn.ModuleList()
        self.skip_norm = nn.ModuleList()
        self.skip_drop = nn.ModuleList()

        skip_start_idx = 0
        for i in range(self.n_layer):
            # normal block
            n_in = cfg.model.conv_ch[i]
            n_out = cfg.model.conv_ch[i + 1]
            kernel_size = cfg.model.conv_kernel[i]
            stride = cfg.model.conv_stride[i]
            padding = int(np.floor((kernel_size - 1) / 2))
            drop_rate = cfg.model.dropout

            self.conv.append(nn.Conv1d(
                in_channels=n_in, out_channels=n_out,
                kernel_size=kernel_size, stride=stride,
                padding=padding, bias=False
            ))

            self.norm.append(nn.BatchNorm1d(num_features=n_in))
            self.drop.append(nn.Dropout(p=drop_rate))

            # Skip Block
            if i % self.skip_cnt == (self.skip_cnt - 1):
                skip_end_idx = i + 1
                skip_stride = 1

                for j in range(skip_start_idx, skip_end_idx):
                    skip_stride *= cfg.model.conv_stride[j]

                skip_kernel = cfg.model.conv_kernel[skip_start_idx]
                skip_padding = int(np.floor((skip_kernel - 1) / 2))
                self.skip_conv.append(nn.Conv1d(
                    in_channels=cfg.model.conv_ch[skip_start_idx],
                    out_channels=cfg.model.conv_ch[skip_end_idx - 1],
                    kernel_size=skip_kernel,
                    stride=skip_stride,
                    padding=skip_padding
                ))

                self.skip_norm.append(nn.BatchNorm1d(num_features=cfg.model.conv_ch[skip_start_idx]))
                self.skip_drop.append(nn.Dropout(p=drop_rate))

                skip_start_idx = skip_end_idx

    def forward(self, x: torch.Tensor):
        # Residual Pass
        x_skip = x
        skip_idx = 0

        for b_idx in range(self.n_layer):
            x = self.norm[b_idx](x)
            x = self.conv[b_idx](x)
            x = self.relu(x)
            x = self.drop[b_idx](x)

            if b_idx % self.skip_cnt == (self.skip_cnt - 1):
                x_skip = self.skip_norm[skip_idx](x_skip)
                x_skip = self.skip_conv[skip_idx](x_skip)
                x_skip = self.relu(x_skip)
                x_skip = self.skip_drop[skip_idx](x_skip)

                x = x + x_skip
                x_skip = x

                skip_idx += 1
        return x


class UTSN(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        out_ch = cfg.model.conv_ch[-1]

        self.use_spec = cfg.model.use_spec
        self.len_spec = cfg.model.n_spec_freq

        self.dropout_rate = cfg.model.dropout

        self.sub_conv = SubConv(cfg)
        self.norm = nn.BatchNorm1d(out_ch)

        self.drop = nn.Dropout(cfg.model.fc_drop)
        self.flat = nn.Flatten()

        self.freq_flat = nn.Flatten()
        self.freq_norm = nn.BatchNorm1d(self.len_spec)
        if self.use_spec:
            fc_in = self.len_spec + cfg.model.out_dim * out_ch
        else:
            fc_in = cfg.model.out_dim * out_ch

        self.fc = nn.Linear(
            in_features=fc_in,
            out_features=cfg.model.n_class
        )

    def forward(self, x: torch.Tensor, freq: torch.Tensor):
        x = self.sub_conv(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.flat(x)

        if self.use_spec:
            freq = self.freq_flat(freq)
            freq = self.freq_norm(freq)
            x = torch.concatenate((x, freq), dim=1)

        x = self.fc(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    from omegaconf import OmegaConf
    cfg = OmegaConf.load('E:\\Workspace\\Pycharm\\UniTimeSeriesNet\\conf\\utsn.yaml')

    batch_size = cfg.train.batch_size
    bins = cfg.model.n_spec_freq
    model = UTSN(cfg)

    pts = round(cfg.dataset.wnd_len * cfg.dataset.fs)
    summary(model,
            input_size=[
                (batch_size, 1, pts),
                (batch_size, bins)
            ])
