<p align="center">
    <h1 align="center">Color Shift Estimation-and-Correction for Image Enhancement</h1>
    <p align="center">
        <a href="https://yiyulics.github.io/">Yiyu Li</a>
        ·
        <a href="https://kkbless.github.io/">Ke Xu</a>
        ·
        <a href="https://scholars.cityu.edu.hk/en/persons/gerhard-petrus-hancke(9e59c8eb-ba32-4075-97f7-e44e82367742).html">Gerhard Petrus Hancke</a>
        ·
        <a href="https://www.cs.cityu.edu.hk/~rynson/">Rynson W.H. Lau</a>
    </p>
</p>

<div align="center">


[![arxiv](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2405.17725)
[![cvf](https://img.shields.io/badge/Paper-CVF-%23357DBD)](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Color_Shift_Estimation-and-Correction_for_Image_Enhancement_CVPR_2024_paper.pdf)
[![LCDP](https://img.shields.io/badge/Dataset-LCDP-%23cda6c3)](https://github.com/onpix/LCDPNet/tree/main)
[![MSEC](https://img.shields.io/badge/Dataset-MSEC-%23cda6c3)](https://github.com/mahmoudnafifi/Exposure_Correction)
[![Pretrained Model](https://img.shields.io/badge/Pretrained-Model-%2380f69a)](https://drive.google.com/drive/folders/1SEQu3f2IdNnLlFH1OLUGyny5Xy-0TGzb?usp=sharing)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/color-shift-estimation-and-correction-for/image-enhancement-on-exposure-errors)](https://paperswithcode.com/sota/image-enhancement-on-exposure-errors?p=color-shift-estimation-and-correction-for)


</div>



**Abstract**: Images captured under sub-optimal illumination conditions may contain both over- and under-exposures.
Current approaches mainly focus on adjusting image brightness, which may exacerbate color tone distortion in under-exposed areas and fail to restore accurate colors in over-exposed regions.
We observe that over- and over-exposed regions display opposite color tone distribution shifts, which may not be easily normalized in joint modeling as they usually do not have "normal-exposed" regions/pixels as reference.
In this paper, we propose a novel method to enhance images with both over- and under-exposures by learning to estimate and correct such color shifts.
Specifically, we first derive the color feature maps of the brightened and darkened versions of the input image via a UNet-based network, followed by a pseudo-normal feature generator to produce pseudo-normal color feature maps.
We then propose a novel COlor Shift Estimation (COSE) module to estimate the color shifts between the derived brightened (or darkened) color feature maps and the pseudo-normal color feature maps.
The COSE module corrects the estimated color shifts of the over- and under-exposed regions separately.
We further propose a novel COlor MOdulation (COMO) module to modulate the separately corrected colors in the over- and under-exposed regions to produce the enhanced image.
Comprehensive experiments show that our method outperforms existing approaches.


## :mega: News
- [2024/04/18] Update Google Drive link for the paper and README.
- [2024/09/08] Release training and testing code, and model weights. Update paper link to arXiv and CVF Open Access version. Update README documentation.
- [2024/11/11] Add environment yaml file.


## :wrench: Installation
To get started, clone this project, create a conda virtual environment using Python 3.9 (or higher versions may do as well), and install the requirements:
```
git clone https://github.com/yiyulics/CSEC.git
cd CSEC

conda create -n csec python=3.9
conda activate csec

# Change the following line to match your CUDA version
# Refer to https://pytorch.org/get-started/previous-versions/#v1121
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

pip install pytorch_lightning==1.7.6
pip install -r requirements.txt
```


## :computer: Running the Code

### Evaluation

To evaluate the trained model, you'll need to do the following steps:
- Get the [pretrained model](https://drive.google.com/drive/folders/1SEQu3f2IdNnLlFH1OLUGyny5Xy-0TGzb?usp=sharing) (or you can use your own trained weights) and put them in the `pretrained/` folder.
- Modify the path to the test dataset in `src/config/ds/test.yaml` (if you don't need ground truth images for testing, just leave the `GT` value as `none`).
- Run the following command:
    ```
    python src/test.py checkpoint_path=/path/to/checkpoint/filename.ckpt
    ```
- Then under the folder `/path/to/checkpoint/`, a new folder named `test_result/` will be created, and all the final enhanced images (`*.png` images) will be saved in this folder. Other intermediate results of each image will also be saved in the subfolders of `test_result/` (e.g., `test_result/normal/` for pseudo-normal images, etc.)


### Training

To train your own model from scratch, you'll need to do the following steps:
- Prepare the training dataset. You can use the [LCDP dataset](https://github.com/onpix/LCDPNet/tree/main) or [MSEC dataset](https://github.com/mahmoudnafifi/Exposure_Correction) (or you can use your own paired data).
- Modify the path to the training dataset in `src/config/ds/train.yaml`.
- Modify the path to the validation dataset in `src/config/ds/valid.yaml` (if have any).
- Run the following command:
    ```
    python src/train.py name=your_experiment_name
    ```
- The trained models and intermediate results will be saved in the `log/` folder.

#### OOM Errors

You may need to reduce the batch size in `src/config/config.yaml` to avoid out of memory errors. If you do this, but want to preserve quality, be sure to increase the number of training iterations and decrease the learning rate by whatever scale factor you decrease batch size by.



## :postbox: Citation
If you find our work helpful, please cite our paper as:
```
@inproceedings{li_2024_cvpr_csec,
    title       =   {Color Shift Estimation-and-Correction for Image Enhancement},
    author      =   {Yiyu Li and Ke Xu and Gerhard Petrus Hancke and Rynson W.H. Lau},
    booktitle   =   {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year        =   {2024}
}
```
Should you have any questions, feel free to post an issue or contact me at [yiyuli.cs@my.cityu.edu.hk](mailto:yiyuli.cs@my.cityu.edu.hk).


## :sparkles: Acknowledgements
The project is largely based on [LCDPNet](https://github.com/onpix/LCDPNet.git). Many thanks to the project for their excellent contributions!


