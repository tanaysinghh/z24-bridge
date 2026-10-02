import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x):
        return super().forward(F.pad(x, (self.left_pad, 0)))


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.dilated = CausalConv1d(residual_channels, 2 * residual_channels, kernel_size, dilation)
        self.residual = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip = nn.Conv1d(residual_channels, skip_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        filt, gate = self.dilated(x).chunk(2, dim=1)
        z = self.dropout(torch.tanh(filt) * torch.sigmoid(gate))
        return x + self.residual(z), self.skip(z)


class WaveNetClassifier(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=17,
        residual_channels=64,
        skip_channels=64,
        kernel_size=2,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        dropout=0.1,
    ):
        super().__init__()
        self.input_conv = CausalConv1d(in_channels, residual_channels, 1)
        self.blocks = nn.ModuleList(
            ResidualBlock(residual_channels, skip_channels, kernel_size, d, dropout) for d in dilations
        )
        self.post = nn.Conv1d(skip_channels, skip_channels, 1)
        self.head = nn.Linear(skip_channels, num_classes)
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)

    def forward(self, x):
        x = self.input_conv(x)
        skip_sum = 0
        for block in self.blocks:
            x, s = block(x)
            skip_sum = skip_sum + s
        h = F.relu(self.post(F.relu(skip_sum)))
        return self.head(h.mean(dim=-1))
