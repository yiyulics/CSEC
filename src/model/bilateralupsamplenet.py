import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from globalenv import *
from model.arch.unet_based.hist_unet import HistUNet

from .arch.drconv import DRConv2d
from .basic_loss import LTVloss
from .single_net_basemodel import SingleNetBaseModel


class LitModel(SingleNetBaseModel):
    def __init__(self, opt):
        super().__init__(
            opt, BilateralUpsampleNet(opt[RUNTIME]), [TRAIN, VALID]
        )
        low_res = opt[RUNTIME][LOW_RESOLUTION]

        self.down_sampler = lambda x: F.interpolate(
            x, size=(low_res, low_res), mode="bicubic", align_corners=False
        )
        self.use_illu = opt[RUNTIME][PREDICT_ILLUMINATION]

        self.mse = torch.nn.MSELoss()
        self.ltv = LTVloss()
        self.cos = torch.nn.CosineSimilarity(1, 1e-8)

        self.net.train()

    def training_step(self, batch, batch_idx):
        input_batch, gt_batch, output_batch = super().training_step_forward(
            batch, batch_idx
        )
        loss_lambda_map = {
            MSE: lambda: self.mse(output_batch, gt_batch),
            COS_LOSS: lambda: (1 - self.cos(output_batch, gt_batch).mean())
            * 0.5,
            LTV_LOSS: lambda: (
                self.ltv(input_batch, self.net.illu_map, 1)
                if self.use_illu
                else None
            ),
        }

        # logging:
        loss = self.calc_and_log_losses(loss_lambda_map)
        self.log_training_iogt_img(
            batch,
            extra_img_dict={
                PREDICT_ILLUMINATION: self.net.illu_map,
                GUIDEMAP: self.net.guidemap,
            },
        )
        return loss

    def validation_step(self, batch, batch_idx):
        super().validation_step(batch, batch_idx)

    def test_step(self, batch, batch_ix):
        super().test_step(batch, batch_ix)

    def forward(self, x):
        low_res_x = self.down_sampler(x)
        return self.net(low_res_x, x)


class ConvBlock(nn.Module):
    def __init__(
        self,
        inc,
        outc,
        kernel_size=3,
        padding=1,
        stride=1,
        use_bias=True,
        activation=nn.ReLU,
        batch_norm=False,
    ):
        super(ConvBlock, self).__init__()
        conv_type = OPT["conv_type"]
        if conv_type == "conv":
            self.conv = nn.Conv2d(
                int(inc),
                int(outc),
                kernel_size,
                padding=padding,
                stride=stride,
                bias=use_bias,
            )
        elif conv_type.startswith("drconv"):
            region_num = int(conv_type.replace("drconv", ""))
            self.conv = DRConv2d(
                int(inc),
                int(outc),
                kernel_size,
                region_num=region_num,
                padding=padding,
                stride=stride,
            )
        else:
            raise NotImplementedError()

        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class SliceNode(nn.Module):
    def __init__(self, opt):
        super(SliceNode, self).__init__()
        self.opt = opt

    def forward(self, bilateral_grid, guidemap):
        # bilateral_grid shape: Nx12x8x16x16
        device = bilateral_grid.get_device()
        N, _, H, W = guidemap.shape
        hg, wg = torch.meshgrid(
            [torch.arange(0, H), torch.arange(0, W)]
        )  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)

        hg = (
            hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1
        )  # norm to [-1,1] NxHxWx1
        wg = (
            wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1
        )  # norm to [-1,1] NxHxWx1
        guidemap = guidemap * 2 - 1
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()
        guidemap_guide = torch.cat([wg, hg, guidemap], dim=3).unsqueeze(1)

        guidemap_guide = guidemap_guide.type_as(bilateral_grid)
        coeff = F.grid_sample(
            bilateral_grid, guidemap_guide, "bilinear", align_corners=True
        )
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):
        """
        coeff shape: [bs, 12, h, w]
        input shape: [bs, 3, h, w]
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        """
        R = (
            torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True)
            + coeff[:, 3:4, :, :]
        )
        G = (
            torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True)
            + coeff[:, 7:8, :, :]
        )
        B = (
            torch.sum(
                full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True
            )
            + coeff[:, 11:12, :, :]
        )

        return torch.cat([R, G, B], dim=1)


class GuideNet(nn.Module):
    def __init__(self, params=None, out_channel=1):
        super(GuideNet, self).__init__()
        self.params = params
        self.conv1 = ConvBlock(3, 16, kernel_size=1, padding=0, batch_norm=True)
        self.conv2 = ConvBlock(
            16, out_channel, kernel_size=1, padding=0, activation=nn.Sigmoid
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))  # .squeeze(1)


class LowResHistUNet(HistUNet):
    def __init__(self, coeff_dim=12, opt=None):
        super(LowResHistUNet, self).__init__(
            in_channels=3,
            out_channels=coeff_dim * opt[LUMA_BINS],
            bilinear=True,
            **opt[HIST_UNET],
        )
        self.coeff_dim = coeff_dim

    def forward(self, x):
        y = super(LowResHistUNet, self).forward(x)
        y = torch.stack(torch.split(y, self.coeff_dim, 1), 2)
        return y


class BilateralUpsampleNet(nn.Module):
    def __init__(self, opt):
        super(BilateralUpsampleNet, self).__init__()
        self.opt = opt
        global OPT
        OPT = opt
        self.guide = GuideNet(params=opt)
        self.slice = SliceNode(opt)
        self.build_coeffs_network(opt)

    def build_coeffs_network(self, opt):
        Backbone = LowResHistUNet
        self.coeffs = Backbone(opt=opt)
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, lowres, fullres):
        bilateral_grid = self.coeffs(lowres)
        try:
            self.guide_features = self.coeffs.guide_features
        except Exception as e:
            print("[ WARN ] {}".format(e))
        guide = self.guide(fullres)
        self.guidemap = guide

        slice_coeffs = self.slice(bilateral_grid, guide)
        out = self.apply_coeffs(slice_coeffs, fullres)

        self.slice_coeffs = slice_coeffs
        self.illu_map = None

        return out
