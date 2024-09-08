import ipdb
import torch
from torch import nn
from torch.nn import functional as F


class CrossNonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(CrossNonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g_x = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.UpsamplingBilinear2d((16, 16)),
        )
        self.g_b = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.UpsamplingBilinear2d((16, 16)),
        )
        self.g_d = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.UpsamplingBilinear2d((16, 16)),
        )

        self.W_x = nn.Conv2d(
            in_channels=self.inter_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.W_b = nn.Conv2d(
            in_channels=self.inter_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.W_d = nn.Conv2d(
            in_channels=self.inter_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.W_xb = nn.Conv2d(
            in_channels=self.inter_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.W_xd = nn.Conv2d(
            in_channels=self.inter_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.theta_x = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.theta_b = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.theta_d = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.phi_x = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.UpsamplingBilinear2d((16, 16)),
        )
        self.phi_b = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.UpsamplingBilinear2d((16, 16)),
        )
        self.phi_d = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.UpsamplingBilinear2d((16, 16)),
        )

        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.bn2 = nn.BatchNorm2d(self.in_channels)

        self.out_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.t = nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.p = nn.Conv2d(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def _DotKernel(self, x):

        t = self.t(x)
        p = self.p(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)

        att = torch.matmul(torch.relu(t), torch.relu(p))
        att = (att + att.permute(0, 2, 1)) / 2
        d = torch.sum(att, dim=2)
        d[d != 0] = torch.sqrt(1.0 / d[d != 0])
        att = att * d.unsqueeze(1) * d.unsqueeze(2)

        return att

    def forward(self, x, ob, od):
        B, C, H, W = x.shape
        assert ob.shape == od.shape == (B, C, H, W)

        g_x = self.g_x(x).view(B, self.inter_channels, -1).permute(0, 2, 1)
        g_b = self.g_b(ob).view(B, self.inter_channels, -1).permute(0, 2, 1)
        g_d = self.g_d(od).view(B, self.inter_channels, -1).permute(0, 2, 1)

        # theta_x = (
        #     self.theta_x(x).view(B, self.inter_channels, -1).permute(0, 2, 1)
        # )
        # theta_b = (
        #     self.theta_b(ob).view(B, self.inter_channels, -1).permute(0, 2, 1)
        # )
        # theta_d = (
        #     self.theta_d(od).view(B, self.inter_channels, -1).permute(0, 2, 1)
        # )

        # phi_x = self.phi_x(x).view(B, self.inter_channels, -1)
        # phi_b = self.phi_b(ob).view(B, self.inter_channels, -1)
        # phi_d = self.phi_d(od).view(B, self.inter_channels, -1)

        # f_x = F.softmax(torch.matmul(theta_x, phi_x), dim=-1)
        # f_b = F.softmax(torch.matmul(theta_b, phi_b), dim=-1)
        # f_d = F.softmax(torch.matmul(theta_d, phi_d), dim=-1)

        f_x = self._DotKernel(x)
        f_b = self._DotKernel(ob)
        f_d = self._DotKernel(od)

        x_self = (
            torch.matmul(f_x, g_x)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, self.inter_channels, H, W)
        )
        ob_self = (
            torch.matmul(f_b, g_b)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, self.inter_channels, H, W)
        )
        od_self = (
            torch.matmul(f_d, g_d)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, self.inter_channels, H, W)
        )
        x_ob_cross = (
            torch.matmul(f_b, g_x)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, self.inter_channels, H, W)
        )
        x_od_cross = (
            torch.matmul(f_d, g_x)
            .permute(0, 2, 1)
            .contiguous()
            .view(B, self.inter_channels, H, W)
        )

        x_self = self.W_x(x_self)
        ob_self = self.W_b(ob_self)
        od_self = self.W_d(od_self)
        x_ob_cross = self.W_xb(x_ob_cross)
        x_od_cross = self.W_xd(x_od_cross)

        od = od_self + x_ob_cross
        od_self = self.bn1(od)
        ob = ob_self + x_od_cross
        ob_self = self.bn2(ob)

        output = od_self + ob_self + x_self
        output = self.out_conv(output)

        return output + x
