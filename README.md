# IBSD
PyTorch implementation of the paper "Information Bottleneck based Self-distillation: Boosting Lightweight Network for Real-world Super-Resolution"

## Introduction
>
> Most existing single-image super-resolution (SISR) methods focus on addressing predefined uniform degradations, such as bicubic. However, these methods often perform poorly in real-world scenarios due to complicated and varying realistic degradations. In this paper, we propose a novel information bottleneck-based self-distillation method (IBSD) to boost lightweight networks for real-world image super-resolution. The proposed IBSD leverages the principle of information bottleneck to guide SR networks to learn invariant correlations from low-resolution (LR) to high-resolution (HR) across various degradations, thereby improving their generalization capacity. Specifically, the target super-resolution network (\ie, student) is interpreted as a Markov chain, and the distillation process is carried out through two modules. Mutual information (MI) estimation networks are used to quantify the mutual information between adjacent nodes within the Markov chain. To enhance robustness against blur and noise in real-world scenarios, an auxiliary loss with a progressive soft target is employed to better identify what is effective for reconstruction in the high-frequency domain. Minimizing the mutual information while preserving task-relevant features can help remove information that reflects spurious correlations between specific degradations and reconstructed targets. Experiments conducted on real-world image super-resolution datasets demonstrate that our proposed method can significantly improve the performance of recent lightweight SR models without adding any extra inference complexity, and it outperforms existing self-distillation approaches.

## Code
### Dependencies
> * Python 3.6
> * PyTorch >= 1.1.0
> * pytorch_wavelets
> * skimage
> * tqdm
> * einops

### Testing
> 1. Clone this github repo.
> ```bash
> git clone https://github.com/hanzhu1121/IBSD.git
> cd IBSD
> ```
> 2. Download [pre-trained models](https://drive.google.com/drive/folders/17JzAB7rafavbmeJkDCtv8h94kQUV3wcY?usp=drive_link) to ```./models``` folder or use your pre-trained models
> 3. Change the ```test_dataroot``` argument in ```CDC_test.py``` to the place where images are located
> 4. Run ```CDC_test.py``` using script file ```test_models_pc.sh```.
> ```bash
> sh test_models_pc.sh cdc_x4_test ./CDC_test.py ./models/HGSR-MHR_X4_SubRegion_GW_283.pth 1
> ```
> 5. You can find the enlarged images in ```./results``` folder

### Training
> 1. Download [DRealSR dataset](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution). Then, change the ```dataroot``` and ```test_dataroot``` argument in ```./options/realSR_HGSR_MSHR.py``` to the place where images are located.
> 2. Run ```CDC_train_test.py``` using script file ```train_pc.sh```.
> ```bash
> sh ./train_pc.sh cdc_x4 ./CDC_train_test.py ./options/realSR_HGSR_MSHR.py 1
> ```
> 3. You can find the results in ```./experiments/CDC-X4``` if the ```exp_name``` argument in ```./options/realSR_HGSR_MSHR.py``` is ```CDC-X4```


## Acknowledgements
>
> This code is built on [CDC](https://arxiv.org/abs/2008.01928). We thank the authors for sharing their codes of CDC  [PyTorch version](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution).
