import torch


class Downsampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 stride=2, padding=1, norm=True, lrelu=True, relu=True):
        super().__init__()
        self.main = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, bias=not norm)
        torch.nn.init.xavier_normal_(self.main.weight)
        self.block = torch.nn.Sequential(self.main)
        if norm:
            self.block.append(torch.nn.InstanceNorm2d(out_channels, affine=True))
        if relu is True and lrelu is True:
            self.block.append(torch.nn.LeakyReLU(0.2, True))
        elif relu is True:
            self.block.append(torch.nn.ReLU(True))

    def forward(self, x):
        return self.block(x)


class Upsampling(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4,
                 stride=2, padding=1, output_padding=0, dropout=False, relu=True, norm=True):
        super().__init__()
        self.main = torch.nn.ConvTranspose2d(in_channels, out_channels, bias=False,
                                             kernel_size=kernel_size, stride=stride,
                                             padding=padding, output_padding=output_padding)
        torch.nn.init.xavier_normal_(self.main.weight)
        self.block = torch.nn.Sequential(self.main)
        if norm:
            self.block.append(torch.nn.InstanceNorm2d(out_channels, affine=True))
        if dropout:
            self.block.append(torch.nn.Dropout(0.5))
        if relu is True:
            self.block.append(torch.nn.ReLU(True))

    def forward(self, x):
        return self.block(x)
