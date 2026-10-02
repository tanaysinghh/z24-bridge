import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes, bottleneck_channels):
        super().__init__()
        self.use_bottleneck = in_channels > 1
        branch_in = bottleneck_channels if self.use_bottleneck else in_channels
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
        self.convs = nn.ModuleList(
            nn.Conv1d(branch_in, n_filters, k, padding="same", bias=False) for k in kernel_sizes
        )
        self.pool_conv = nn.Conv1d(in_channels, n_filters, 1, bias=False)
        self.bn = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))

    def forward(self, x):
        z = self.bottleneck(x) if self.use_bottleneck else x
        branches = [conv(z) for conv in self.convs]
        branches.append(self.pool_conv(F.max_pool1d(x, 3, stride=1, padding=1)))
        return F.relu(self.bn(torch.cat(branches, dim=1)))


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, residual):
        return F.relu(x + self.bn(self.conv(residual)))


class InceptionTime(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=17,
        n_filters=32,
        kernel_sizes=(10, 20, 40),
        bottleneck_channels=38,
        depth=6,
        residual_every=3,
    ):
        super().__init__()
        out_channels = n_filters * (len(kernel_sizes) + 1)
        self.residual_every = residual_every
        self.modules_list = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        channels = in_channels
        residual_channels = in_channels
        for d in range(depth):
            self.modules_list.append(InceptionModule(channels, n_filters, kernel_sizes, bottleneck_channels))
            channels = out_channels
            if d % residual_every == residual_every - 1:
                self.shortcuts.append(Shortcut(residual_channels, out_channels))
                residual_channels = out_channels
        self.head = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        residual = x
        shortcut_idx = 0
        for d, module in enumerate(self.modules_list):
            x = module(x)
            if d % self.residual_every == self.residual_every - 1:
                x = self.shortcuts[shortcut_idx](x, residual)
                residual = x
                shortcut_idx += 1
        return self.head(x.mean(dim=-1))
