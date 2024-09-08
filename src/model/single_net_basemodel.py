# -*- coding: utf-8 -*-

import os.path as osp

import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim

import utils.util as util
from globalenv import *

from .basemodel import BaseModel


class SingleNetBaseModel(BaseModel):
    # for models with only one self.net
    def __init__(self, opt, net, running_modes, print_arch=False):
        super().__init__(opt, running_modes)
        self.net = net
        self.net.train()

        # config for SingleNetBaseModel
        if print_arch:
            print(str(net))
        self.tonemapper = cv2.createTonemapReinhard(2.2)

        # training step forward
        self.cnt_iters = 1

    def configure_optimizers(self):
        # self.parameters in LitModel is the same as nn.Module.
        # once you add nn.xxxx as a member in __init__, self.parameters will include it.
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )
        return [optimizer], [schedular]

    def forward(self, x):
        return self.net(x)

    def training_step_forward(self, batch, batch_idx):
        torch.cuda.empty_cache()

        if (
            not self.MODEL_WATCHED
            and not self.opt[DEBUG]
            and self.opt.logger == "wandb"
        ):
            self.logger.experiment.watch(
                self.net, log_freq=self.opt[LOG_EVERY] * 2, log_graph=True
            )
            self.MODEL_WATCHED = True

        input_batch, gt_batch = batch[INPUT], batch[GT]
        output_batch = self(input_batch)
        for k, v in self.net.res.items():
            if ILLU_MAP == k:
                illu_map = v
            elif INVERSE == k:
                inverse = v
            elif INVERSE_ILLU_MAP == k:
                inverse_illu_map = v
            elif BRIGHTEN_INPUT == k:
                brighten_input = v
            elif DARKEN_INPUT == k:
                darken_input = v
            elif NORMAL == k:
                normal_ex = v
            elif BRIGHTEN_OFFSET == k:
                brighten_offset = v
            elif DARKEN_OFFSET == k:
                darken_offset = v
            else:
                continue
        self.iogt = {
            INPUT: input_batch,
            ILLU_MAP: illu_map,
            BRIGHTEN_INPUT: brighten_input,
            BRIGHTEN_OFFSET: brighten_offset,
            NORMAL: normal_ex,
            INVERSE_ILLU_MAP: inverse_illu_map,
            DARKEN_INPUT: darken_input,
            DARKEN_OFFSET: darken_offset,
            OUTPUT: output_batch,
            GT: gt_batch,
        }
        return input_batch, gt_batch, output_batch

    def validation_step(self, batch, batch_idx): ...

    def on_validation_start(self):
        self.total_psnr = []
        self.total_ssim = []

    def on_validation_end(self): ...

    def log_training_iogt_img(self, batch, extra_img_dict=None):
        """
        Only used in training_step
        """
        if extra_img_dict:
            img_dict = {**self.iogt, **extra_img_dict}
        else:
            img_dict = self.iogt

        if self.global_step % self.opt[LOG_EVERY] == 0:
            self.log_images_dict(
                mode=TRAIN,
                input_fname=osp.basename(batch[INPUT_FPATH][0]),
                img_batch_dict=img_dict,
                gt_fname=osp.basename(batch[GT_FPATH][0]),
            )

    @staticmethod
    def logdomain2hdr(ldr_batch):
        return 10**ldr_batch - 1

    def on_test_start(self):
        self.total_psnr = []
        self.total_ssim = []
        self.global_test_step = 0

    def on_test_end(self):
        print(
            f"Test step: {len(self.total_psnr)}, Manual PSNR: {sum(self.total_psnr) / len(self.total_psnr)}, Manual SSIM: {sum(self.total_ssim) / len(self.total_ssim)}"
        )

    def test_step(self, batch, batch_ix):
        """
        save test result and calculate PSNR and SSIM for `self.net` (when have GT)
        """
        # test without GT image:
        self.global_test_step += 1
        input_batch = batch[INPUT]
        assert input_batch.shape[0] == 1
        output_batch = self(input_batch)
        save_num = 1

        # test with GT:
        if GT in batch:
            gt_batch = batch[GT]
            if output_batch.shape != batch[GT].shape:
                print(
                    f"[[ WARN ]] output.shape is {output_batch.shape} but GT.shape is {batch[GT].shape}. Resize GT to output to get PSNR."
                )
                gt_batch = F.interpolate(batch[GT], output_batch.shape[2:])

            output_ = util.cuda_tensor_to_ndarray(output_batch)
            y_ = util.cuda_tensor_to_ndarray(gt_batch)
            psnr = util.ImageProcessing.compute_psnr(output_, y_, 1.0)
            ssim = util.ImageProcessing.compute_ssim(output_, y_)
            self.total_psnr.append(psnr)
            self.total_ssim.append(ssim)

        # save images
        self.save_img_batch(
            output_batch,
            self.opt[IMG_DIRPATH],
            osp.basename(batch[INPUT_FPATH][0]),
            save_num=save_num,
        )
