import torch
from torch import nn


class TinySleepNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.fs = config['sampling_rate']
        self.fs = config['fs_tgt']
        self.use_rnn = self.config['use_rnn']
        self.n_rnn_unit = self.config['n_rnn_units']
        self.rnn_input_size = self.config['rnn_input_size']

        self.padding = {
            'conv_1': (6, 7),
            'pool_1': (4, 5),
            'conv_2': (1, 2),
            'pool_2': (1, 2),
        }

        # fs should resampled to 250Hz
        # input shape (batch_size x seq_length, seq_length x seq_t x fs)
        first_filter_size = int(self.fs / 4.0)
        first_filter_stride = int(self.fs / 32.0)

        self.conv_1 = nn.Sequential(
            nn.ConstantPad1d(self.padding['conv_1'], 0),
            nn.Conv1d(in_channels=1,
                      out_channels=64,
                      kernel_size=first_filter_size,
                      stride=first_filter_stride,
                      bias=False
                      ),
            nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.pool_1 = nn.Sequential(
            nn.ConstantPad1d(self.padding['pool_1'], 0),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.25),
        )

        self.conv_2 = nn.Sequential(
            nn.ConstantPad1d(self.padding['conv_2'], 0),
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=4,
                      stride=1,
                      bias=False,
                      ),
            nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.conv_3 = nn.Sequential(
            nn.ConstantPad1d(self.padding['conv_2'], 0),  # conv3
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=4,
                      stride=1,
                      bias=False
                      ),
            nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.conv_4 = nn.Sequential(
            nn.ConstantPad1d(self.padding['conv_2'], 0),  # conv4
            nn.Conv1d(in_channels=64,
                      out_channels=64,
                      kernel_size=4,
                      stride=1,
                      bias=False
                      ),
            nn.BatchNorm1d(num_features=64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.pool_2 = nn.Sequential(
            nn.ConstantPad1d(self.padding['pool_2'], 0),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Flatten(),
            nn.Dropout(p=0.25),
        )

        self.cnn = nn.Sequential(
            self.conv_1,
            self.pool_1,
            self.conv_2,
            self.conv_3,
            self.conv_4,
            self.pool_2,
        )

        # self.rnn = nn.LSTM(input_size=32 * 16,
        #                    hidden_size=self.n_rnn_unit,
        #                    num_layers=1,
        #                    batch_first=True
        #                    )
        #
        # # Auto softmax with cross entropy loss
        # self.fc = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(in_features=self.n_rnn_unit,
        #               out_features=3),
        # )

        # Auto softmax with cross entropy loss
        self.fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_features=1024,
                      out_features=3),
        )

    def forward(self, x: torch.Tensor, h, c):
        x: torch.Tensor = self.cnn(x)
        # x = x.view(-1, self.config['seq_length'], 32 * 16)
        #
        # x, (h, c) = self.rnn(x, (h, c))
        # x = x.contiguous().view(-1, self.n_rnn_unit)

        x = self.fc(x)

        return x, h, c


if __name__ == '__main__':
    from torchinfo import summary
    from conf.tiny import config

    batch_size = config['batch_size']
    seq_length = config['seq_length']
    n_rnn_units = config['n_rnn_units']
    model = TinySleepNet(config=config)

    state = (
        torch.zeros(size=(1, batch_size, n_rnn_units)),
        torch.zeros(size=(1, batch_size, n_rnn_units))
    )
    summary(model,
            input_size=[
                (batch_size * seq_length, 1, 2500),
                (1, batch_size, n_rnn_units),
                (1, batch_size, n_rnn_units)
            ])
