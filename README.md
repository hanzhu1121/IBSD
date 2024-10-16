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
> 2. Download [pre-trained models](https://drive.google.com/drive/folders/17JzAB7rafavbmeJkDCtv8h94kQUV3wcY?usp=sharing) to ```./Pretrained``` folder or use your pre-trained models
> 3. Change the ```test_dataroot``` argument in ```./src/test.py``` to the place where images are located
> 4. Run ```./src/test.py``` using script file ```test_models_pc.sh```.
> ```bash
> cd src
> sh test_models_pc.sh imdn_x4_IBSD_test ./test.py ../Pretrained/IMDN_X4_IBSD.pth 0
> ```
> 5. You can find the enlarged images in ```./results``` folder

### Training
Coming soon...
> The key code is available in ```./src/modules/model/ibnet.py```

## Acknowledgements
>
> This code is built on [CDC](https://arxiv.org/abs/2008.01928). We thank the authors for sharing their codes of CDC  [PyTorch version](https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution).
