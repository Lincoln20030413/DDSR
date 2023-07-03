# DDSR
PyTorch code for paper "Dual-Diffusion: Dual Conditional Denoising Diffusion Probabilistic Models for Blind Super-Resolution Reconstruction in RSIs", which can be seen at https://doi.org/10.48550/arXiv.2305.12170. 
The code is based on https://github.com/megvii-research/DCLS-SR/tree/master/codes

# The order of running code:
1. RRDB_LR encoder (The pretrained RRDB_LR encoder has been given in the foler"Pretrained rrdb_LR encoder")
2. Kernel Predictor
3. HR reconstructor

# Dataset
The datasets are shared at:
[https://pan.bnu.edu.cn/l/EFKfQK ](https://pan.baidu.com/s/1eD_mbFoNdPWfY8TCkjjfeA?pwd=j3vq )

To transform datasets to binary files for efficient IO, run:

python codes/scripts/create_lmdb.py

To generate LRblur/LR_up/Bicubic datasets paths, run:

python codes/scripts/generate_mod_blur_LR_bic.py
(You need to modify the file paths by yourself.)
