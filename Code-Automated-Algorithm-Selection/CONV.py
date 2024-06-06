import torch
from torch import Tensor

from EncoderDecoder import Upsampling, Downsampling


class Conv(torch.nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        in_channels = 1
        hidden_channels = 2
        self.layers = [torch.nn.ConvTranspose2d(in_channels, hidden_channels * 2,
                                                kernel_size=(4, 4), stride=1,
                                                padding=0, output_padding=0),
                       Upsampling(hidden_channels * 2, hidden_channels * 4, dropout=True,
                                  kernel_size=(4, 4), stride=1,
                                  padding=0),
                       Upsampling(hidden_channels * 4, hidden_channels * 8, dropout=True,
                                  kernel_size=(4, 4), stride=2,
                                  padding=0),
                       Upsampling(hidden_channels * 8, hidden_channels * 16, dropout=True,
                                  kernel_size=(4, 4), stride=2,
                                  padding=1),

                       Downsampling(hidden_channels * 16, hidden_channels * 8,
                                    kernel_size=(4, 4), stride=2,
                                    padding=1),
                       Downsampling(hidden_channels * 8, hidden_channels * 4,
                                    kernel_size=(4, 4), stride=2,
                                    padding=0),
                       Downsampling(hidden_channels * 4, hidden_channels * 2,
                                    kernel_size=(4, 4), stride=1,
                                    padding=0),
                       Downsampling(hidden_channels * 2, hidden_channels * 1,
                                    kernel_size=(4, 4), stride=1,
                                    padding=0, norm=False),
                       torch.nn.Flatten(),
                       torch.nn.Linear(32, 100),
                       torch.nn.Linear(100, 6),
                       torch.nn.Softmax(dim=1)]
        torch.nn.init.xavier_normal_(self.layers[0].weight)
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, features: Tensor):
        features = self.layers(features)
        return features
