# Color Shift Estimation-and-Correction for Image Enhancement (CVPR 2024)

<div align="center">
    📜<a href="https://drive.google.com/file/d/1jZB2rW_I2WLTE5yNA4IZq9wb5p4NNOCR/view?usp=drive_link">Paper</a>(Google Drive)
</div>

<br/>

This is the official implementation of the paper "Color Shift Estimation-and-Correction for Image Enhancement". The code is implemented in PyTorch.


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


## :postbox: Cite Our Paper
If you find our work helpful, feel free to cite our paper as:
```
@inproceedings{li_2024_cvpr,
    title =        {Color Shift Estimation-and-Correction for Image Enhancement},
    author =       {Yiyu Li, Ke Xu, Gerhard Petrus Hancke, and Rynson W.H. Lau},
    booktitle =    {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year =         {2024}
}
```


