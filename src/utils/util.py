# -*- coding: utf-8 -*-
import logging
import os
import os.path as osp
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
import ipdb
import numpy as np
import omegaconf
import torch
import yaml
from matplotlib.image import imread
from PIL import Image
from skimage.metrics import structural_similarity as calc_ssim
from torch.autograd import Variable

from globalenv import *


def update_global_opt(global_opt, valued_opt):
    for k, v in valued_opt.items():
        global_opt[k] = v


def mkdir(dirpath):
    if not osp.exists(dirpath):
        print(f'Creating directory: "{dirpath}"')
        try:
            os.makedirs(dirpath)
        except:
            ipdb.set_trace()
        return


def rgb2gray(img):
    # img: [N, C, H, W]
    assert (
        len(img.shape) == 4
    ), f"The input image must in (N, C, H, W), but receive image with shape: {img.shape}."
    assert (
        img.shape[1] == 3
    ), f"The image must have 3 channels, but receive image with shape: {img.shape}."
    gray = (
        0.2989 * img[:, 0, :, :]
        + 0.5870 * img[:, 1, :, :]
        + 0.1140 * img[:, 2, :, :]
    )  # Y = 0.2989 R + 0.5870 G + 0.1140 B
    gray = gray.unsqueeze(1)  # [N, H, W] -> [N, 1, H, W]
    return gray


def cuda_tensor_to_ndarray(cuda_tensor):
    return cuda_tensor.clone().detach().cpu().numpy()


def tensor2pil(img):
    img = img.squeeze()  # * 0.5 + 0.5
    return Image.fromarray(
        img.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )


def save_opt(dirpath, opt):
    save_opt_fpath = dirpath / OPT_FILENAME

    with save_opt_fpath.open("w", encoding="utf-8") as f:
        yaml.dump(omega2dict(opt), f, default_flow_style=False)


def init_logging(mode, opt):
    assert mode == TRAIN

    log_dirpath = (
        ROOT_PATH / TRAIN_LOG_DIRNAME / opt[RUNTIME][MODELNAME] / opt[NAME]
    )
    img_dirpath = log_dirpath / IMAGES

    mkdir(log_dirpath)
    mkdir(img_dirpath)
    save_opt(log_dirpath, opt)

    return str(log_dirpath), str(img_dirpath)


def saveTensorAsImg(output, path, downsample_factor=False):
    output = output.squeeze(0)

    if len(output.shape) == 4:
        # a batch
        res = []
        for i in range(len(output)):
            res.append(
                saveTensorAsImg(output[i], f"{path}-{i}.png", downsample_factor)
            )
        return res

    # a single image
    assert len(output.shape) == 3
    outImg = cuda_tensor_to_ndarray(output.permute(1, 2, 0)) * 255.0
    outImg = outImg[:, :, [2, 1, 0]].astype(np.uint8)

    if downsample_factor:
        assert type(downsample_factor + 0.1) == float
        h = outImg.shape[0]
        w = outImg.shape[1]
        outImg = cv2.resize(
            outImg, (int(w / downsample_factor), int(h / downsample_factor))
        ).astype(np.uint8)

    cv2.imwrite(path, outImg)
    return outImg


def parse_config(opt, mode):
    def checkField(opt, name, raise_msg):
        try:
            assert name in opt
        except:
            raise RuntimeError(raise_msg)

    # make sure the model is implemented:
    modelname = opt[RUNTIME][MODELNAME]

    # check fields in runtime config is the same as template.
    # use `modelname.default.yaml` as template.
    runtime_config_dir = SRC_PATH.absolute() / CONFIG_DIR / RUNTIME
    template_yml_path = runtime_config_dir / f"{modelname}.default.yaml"
    print(f'Check runtime config: use "{template_yml_path}" as template.')
    assert template_yml_path.exists()

    pl_logger = logging.getLogger("lightning")
    pl_logger.propagate = False
    return opt


def omega2dict(opt):
    if type(opt) == omegaconf.DictConfig:
        return omegaconf.OmegaConf.to_container(opt)
    else:
        return opt


class ImageProcessing(object):
    @staticmethod
    def rgb_to_lab(img, is_training=True):
        """PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        :param img:
        :returns:
        :rtype:

        """
        img = img.permute(2, 1, 0)
        shape = img.shape
        img = img.contiguous()
        img = img.view(-1, 3)

        img = (img / 12.92) * img.le(0.04045).float() + (
            ((torch.clamp(img, min=0.0001) + 0.055) / 1.055) ** 2.4
        ) * img.gt(0.04045).float()

        rgb_to_xyz = Variable(
            torch.FloatTensor(
                [  # X        Y          Z
                    [0.412453, 0.212671, 0.019334],  # R
                    [0.357580, 0.715160, 0.119193],  # G
                    [0.180423, 0.072169, 0.950227],  # B
                ]
            ),
            requires_grad=False,
        ).type_as(img)

        img = torch.matmul(img, rgb_to_xyz)
        img = torch.mul(
            img,
            Variable(
                torch.FloatTensor([1 / 0.950456, 1.0, 1 / 1.088754]),
                requires_grad=False,
            ).type_as(img),
        )

        epsilon = 6 / 29

        img = (
            (img / (3.0 * epsilon**2) + 4.0 / 29.0) * img.le(epsilon**3).float()
        ) + (
            torch.clamp(img, min=0.0001) ** (1.0 / 3.0)
            * img.gt(epsilon**3).float()
        )

        fxfyfz_to_lab = Variable(
            torch.FloatTensor(
                [
                    [0.0, 500.0, 0.0],  # fx
                    [116.0, -500.0, 200.0],  # fy
                    [0.0, 0.0, -200.0],  # fz
                ]
            ),
            requires_grad=False,
        ).type_as(img)

        img = torch.matmul(img, fxfyfz_to_lab) + Variable(
            torch.FloatTensor([-16.0, 0.0, 0.0]), requires_grad=False
        ).type_as(img)

        img = img.view(shape)
        img = img.permute(2, 1, 0)

        """
        L_chan: black and white with input range [0, 100]
        a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
        [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
        """
        img[0, :, :] = img[0, :, :] / 100
        img[1, :, :] = (img[1, :, :] / 110 + 1) / 2
        img[2, :, :] = (img[2, :, :] / 110 + 1) / 2

        img[(img != img).detach()] = 0

        img = img.contiguous()

        return img

    @staticmethod
    def swapimdims_3HW_HW3(img):
        """Move the image channels to the first dimension of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 1, 2), 0, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 2, 3), 1, 3)

    @staticmethod
    def swapimdims_HW3_3HW(img):
        """Move the image channels to the last dimensiion of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)

    @staticmethod
    def load_image(img_filepath, normaliser):
        """Loads an image from file as a numpy multi-dimensional array

        :param img_filepath: filepath to the image
        :returns: image as a multi-dimensional numpy array
        :rtype: multi-dimensional numpy array

        """
        img = ImageProcessing.normalise_image(
            imread(img_filepath), normaliser
        )  # NB: imread normalises to 0-1
        return img

    @staticmethod
    def normalise_image(img, normaliser):
        """Normalises image data to be a float between 0 and 1

        :param img: Image as a numpy multi-dimensional image array
        :returns: Normalised image as a numpy multi-dimensional image array
        :rtype: Numpy array

        """
        img = img.astype("float32") / normaliser
        return img

    @staticmethod
    def compute_mse(original, result):
        """Computes the mean squared error between to RGB images represented as multi-dimensional numpy arrays.

        :param original: input RGB image as a numpy array
        :param result: target RGB image as a numpy array
        :returns: the mean squared error between the input and target images
        :rtype: float

        """
        return ((original - result) ** 2).mean()

    @staticmethod
    def compute_psnr(image_batchA, image_batchB, max_intensity=1.0):
        """Computes the average PSNR for a batch of input and output images
        could be used during training / validation

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """

        assert (
            image_batchA.shape == image_batchB.shape
        ), f"Input images with shape {image_batchA.shape} and {image_batchB.shape} are not the same."
        assert image_batchA.shape[1] == 3, "Input images must have 3 channels."

        num_images = image_batchA.shape[0]
        psnr_val = 0.0

        for imageA, imageB in zip(image_batchA, image_batchB):
            if torch.is_tensor(imageA):
                imageA, imageB = imageA.numpy().astype(
                    float
                ), imageB.numpy().astype(float)
            elif isinstance(imageA, np.ndarray):
                imageA, imageB = imageA.astype(float), imageB.astype(float)
            else:
                imageA, imageB = np.array(imageA).astype(float), np.array(
                    imageB
                ).astype(float)
            psnr_val += cv2.PSNR(imageA, imageB, max_intensity)

        return psnr_val / num_images

    @staticmethod
    def compute_ssim(image_batchA, image_batchB):
        """Computes the SSIM for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        assert (
            image_batchA.shape == image_batchB.shape
        ), f"Input images with shape {image_batchA.shape} and {image_batchB.shape} are not the same."
        assert image_batchA.shape[1] == 3, "Input images must have 3 channels."

        num_images = image_batchA.shape[0]
        ssim_val = 0.0

        for imageA, imageB in zip(image_batchA, image_batchB):
            ssim_val += calc_ssim(
                imageA,
                imageB,
                channel_axis=0,
                data_range=imageA.max() - imageA.min(),
                gaussian_weights=True,
                win_size=11,
            )

        return ssim_val / num_images
