import os
import os.path as osp
import pathlib
from collections.abc import Iterable

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import wandb

import utils.util as util
from globalenv import *


class BaseModel(pl.core.LightningModule):
    def __init__(self, opt, running_modes):
        """
        logger_img_group_names: images group names in wandb logger. recommand: ['train', 'valid']
        """

        super().__init__()
        self.save_hyperparameters(dict(opt))
        print("Running initialization for BaseModel")

        if IMG_DIRPATH in opt:
            # in training mode.
            # if in test mode, configLogging is not called.
            if TRAIN in running_modes:
                self.train_img_dirpath = osp.join(opt[IMG_DIRPATH], TRAIN)
                util.mkdir(self.train_img_dirpath)
            if VALID in running_modes and (
                len(opt[VALID_DATA].keys()) > 1 or opt[VALID_RATIO]
            ):
                self.valid_img_dirpath = osp.join(opt[IMG_DIRPATH], VALID)
                util.mkdir(self.valid_img_dirpath)

        self.opt = opt
        self.learning_rate = self.opt[LR]

        self.MODEL_WATCHED = False  # for wandb watching model
        self.global_valid_step = 0
        self.iogt = {}  # a dict, saving input, output and gt batch

        assert isinstance(running_modes, Iterable)
        self.logger_image_buffer = {k: [] for k in running_modes}

    def build_test_res_dir(self):
        assert self.opt[CHECKPOINT_PATH]
        modelpath = pathlib.Path(self.opt[CHECKPOINT_PATH])

        # only `test_ds` is supported when testing.
        ds_type = TEST_DATA
        runtime_dirname = f"{self.opt.runtime.modelname}_{modelpath.parent.name}_{modelpath.name}@{self.opt.test_ds.name}"
        dirpath = modelpath.parent / TEST_RESULT_DIRNAME

        if (dirpath / runtime_dirname).exists():
            if len(os.listdir(dirpath / runtime_dirname)) == 0:
                # an existing but empty dir
                pass
            else:
                try:
                    input_str = input(
                        f'[ WARN ] Result directory "{runtime_dirname}" exists. Press ENTER to overwrite or input suffix '
                        f"to create a new one:\n> New name: {runtime_dirname}."
                    )
                except Exception as e:
                    print(
                        f"[ WARN ] Excepion {e} occured, ignore input and set `input_str` empty."
                    )
                    input_str = ""
                if input_str == "":
                    print(f"[ WARN ] Overwrite result_dir: {runtime_dirname}")
                    pass
                else:
                    runtime_dirname += "." + input_str

        dirpath /= runtime_dirname
        util.mkdir(dirpath)
        print("TEST - Result save path:")
        print(str(dirpath))

        util.save_opt(dirpath, self.opt)
        return str(dirpath)

    @staticmethod
    def save_img_batch(batch, dirpath, fname, save_num=1):
        util.mkdir(dirpath)
        imgpath = osp.join(dirpath, fname)

        # If you want to visiual a single image, call .unsqueeze(0)
        assert len(batch.shape) == 4
        torchvision.utils.save_image(batch[:save_num], imgpath)

    def calc_and_log_losses(self, loss_lambda_map):
        logged_losses = {}
        loss = 0
        for loss_name, loss_weight in self.opt[RUNTIME][LOSS].items():
            if loss_weight:
                try:
                    current = loss_lambda_map[loss_name]()
                except KeyError:
                    continue
                if current != None:
                    current *= loss_weight
                    logged_losses[loss_name] = current
                    loss += current

        logged_losses[LOSS] = loss
        self.log_dict(logged_losses)
        return loss

    def log_images_dict(self, mode, input_fname, img_batch_dict, gt_fname=None):
        """
        log input, output and gt images to local disk and remote wandb logger.
        mode: TRAIN or VALID
        """
        if self.opt[DEBUG]:
            return

        global LOGGER_BUFFER_LOCK
        if LOGGER_BUFFER_LOCK and self.opt.logger == "wandb":
            return

        assert mode in [TRAIN, VALID]
        if mode == VALID:
            local_dirpath = self.valid_img_dirpath
            step = self.global_valid_step
            if self.global_valid_step == 0:
                print(
                    "WARN: Found global_valid_step=0. Maybe you foget to increase `self.global_valid_step` in `self.validation_step`?"
                )
        elif mode == TRAIN:
            local_dirpath = self.train_img_dirpath
            step = self.global_step

        if (
            (mode == TRAIN)
            and (step % self.opt[LOG_EVERY] == 0)
            or (mode == VALID)
        ):
            prefix = f"epoch{self.current_epoch}_step{step}_"
            ext = ".png"
            input_fname = osp.splitext(osp.basename(input_fname))[0]
            input_fname = prefix + input_fname + ext

            if gt_fname:
                gt_fname = osp.splitext(osp.basename(gt_fname))[0]
                gt_fname = prefix + gt_fname + ext

            # ****** public buffer opration ******
            LOGGER_BUFFER_LOCK = True
            for name, batch in img_batch_dict.items():
                if batch is None or batch is False:
                    continue

                # save local image:
                fname = input_fname
                if name == GT and gt_fname:
                    fname = gt_fname

                # save remote image:
                if self.opt.logger == "wandb":
                    self.add_img_to_buffer(mode, batch, mode, name, fname)
                else:
                    # tb logger
                    self.logger.experiment.add_image(
                        f"{mode}/{name}", batch[0], step
                    )

            # concatenate all images in buffer and save to local disk
            H, W = img_batch_dict[INPUT].shape[2:]
            concate_imgs = []
            for k, v in img_batch_dict.items():
                if v is not None:
                    if v.shape[1] == 3:
                        concate_imgs.append(
                            F.interpolate(
                                v,
                                size=(H, W),
                                mode="bilinear",
                                align_corners=False,
                            )
                        )
                    elif v.shape[1] == 1:
                        concate_imgs.append(
                            F.interpolate(
                                v.repeat(1, 3, 1, 1),
                                size=(H, W),
                                mode="bilinear",
                                align_corners=False,
                            )
                        )
                    else:
                        raise ValueError(
                            f"Only support 3-channel or 1-channel image, but got {v.shape[1]}"
                        )
                else:
                    raise ValueError(
                        f"None value in img_batch_dict with key {k}"
                    )
            all_images = torch.cat(
                [
                    torch.cat(concate_imgs[:5], dim=3),
                    torch.cat(concate_imgs[5:], dim=3),
                ],
                dim=2,
            )
            self.save_img_batch(
                batch=all_images,
                dirpath=local_dirpath,
                fname=input_fname,
                save_num=1,
            )

            if self.opt.logger == "wandb":
                self.commit_logger_buffer(mode)

            # self.buffer_img_step += 1
            LOGGER_BUFFER_LOCK = False
            # ****** public buffer opration ******

    def add_img_to_buffer(self, group_name, batch, *caption):
        if len(batch.shape) == 3:
            # when input is not a batch:
            batch = batch.unsqueeze(0)

        self.logger_image_buffer[group_name].append(
            wandb.Image(batch[0], caption="-".join(caption))
        )

    def commit_logger_buffer(self, groupname, **kwargs):
        assert self.logger
        self.logger.experiment.log(
            {groupname: self.logger_image_buffer[groupname]}, **kwargs
        )

        self.logger_image_buffer[groupname].clear()
