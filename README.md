# Color Shift Estimation-and-Correction for Image Enhancement (CVPR 2024)

[Paper(arXiv)](https://arxiv.org/abs/2405.17725) | [Paper(CVF Open Access)](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Color_Shift_Estimation-and-Correction_for_Image_Enhancement_CVPR_2024_paper.pdf) | [Dataset(LCDP)](https://github.com/onpix/LCDPNet/tree/main) | [Dataset(MSEC)](https://github.com/mahmoudnafifi/Exposure_Correction) | [Pretrained Models](https://drive.google.com/drive/folders/1SEQu3f2IdNnLlFH1OLUGyny5Xy-0TGzb?usp=sharing)

This is the official implementation of the paper *"Color Shift Estimation-and-Correction for Image Enhancement"*. The code is implemented in PyTorch.


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


## :wrench: Installation
To get started, clone this project, create a conda virtual environment using Python 3.9 (or higher versions may do as well), and install the requirements:
```
git clone https://github.com/yiyulics/CSEC.git
cd CSEC
conda create -n csec python=3.9
conda activate csec
# Change the following line to match your environment
# Reference: https://pytorch.org/get-started/previous-versions/#v1121
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```


## :computer: Running the Code

### Evaluation

To evaluate the trained model, you'll need to do the following steps:
- Get the [pretrained models](https://drive.google.com/drive/folders/1SEQu3f2IdNnLlFH1OLUGyny5Xy-0TGzb?usp=sharing) (or you can use your own trained weights) and put them in the `pretrained/` folder.
- Modify the path to the test dataset in `src/config/ds/test.yaml` (if you don't need ground truth images for testing, just leave the `GT` value as `none`).
- Run the following command:
    ```
    python src/test.py checkpoint_path=path/to/checkpoint/filename.ckpt
    ```
- The results will be saved in the `test_result/` folder under `path/to/checkpoint/`.

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


