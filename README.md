# DDSR
PyTorch code for paper "Dual-Diffusion: Dual Conditional Denoising Diffusion Probabilistic Models for Blind Super-Resolution Reconstruction in RSIs".
The code is based on https://github.com/megvii-research/DCLS-SR/tree/master/codes


The order of running code:
1. RRDB_LR encoder
2. Kernel predictor
3. HR reconstructor
(Check framework for more details)

The datasets are from:
https://pan.bnu.edu.cn/l/EFKfQK 

To transform datasets to binary files for efficient IO, run:

python3 codes/scripts/create_lmdb.py

To generate LRblur/LR_up/Bicubic datasets paths, run:

python3 codes/scripts/generate_mod_blur_LR_bic.py
(You need to modify the file paths by yourself.)
