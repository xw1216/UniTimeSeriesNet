import torch
from torch import nn


class SubNet(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)


class UTSN(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

        self.use_spec = cfg.model.use_spec
        self.len_spec = cfg.model.n_spec_freq

        self.dropout_rate = cfg.model.dropout

