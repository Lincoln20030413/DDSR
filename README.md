# DDSR
PyTorch code for paper "Dual-Diffusion: Dual Conditional Denoising Diffusion Probabilistic Models for Blind Super-Resolution Reconstruction in RSIs", which can be seen at https://doi.org/10.48550/arXiv.2305.12170. 
The code is based on https://github.com/megvii-research/DCLS-SR/tree/master/codes

# The order of running code:
1. RRDB_LR encoder (The pretrained RRDB_LR encoder has been given in the folder "Pretrained rrdb_LR encoder")
2. Kernel Predictor
3. HR reconstructor

# Dataset
The link to the dataset: https://pan.baidu.com/s/1eD_mbFoNdPWfY8TCkjjfeA?pwd=j3vq

To transform datasets to binary files for efficient IO, run:

python codes/scripts/create_lmdb.py

To generate LRblur/LR_up/Bicubic datasets paths, run:

python codes/scripts/generate_mod_blur_LR_bic.py
(You need to modify the file paths by yourself.)

# Supplementary experiments
1. How good is the proposed HR reconstructor compared to non-blind methods when all those methods are given the same (predicted or true) kernel？
   ![image](https://github.com/Lincoln20030413/DDSR/assets/72965675/ab2bda51-b420-4512-a42d-076f6c5792cc)
   ![c26cbadd8186f509a09082d998bfd35](https://github.com/Lincoln20030413/DDSR/assets/72965675/cda2f357-ab3a-4ff6-b8f0-06c30568cfd7)
   ![4476c2cebe3eb361a36cd86b2145242](https://github.com/Lincoln20030413/DDSR/assets/72965675/3fba715b-941a-488c-9821-71f27a0935dd)

2. How good is the kernel predictor compared to the other blind methods' predictors
   ![7ce23fae28bb9622b2d7f772b8cdcbc](https://github.com/Lincoln20030413/DDSR/assets/72965675/9f366023-4435-4337-a8c0-171c9c822a86)
   
   ![f8f284bdeaa819e46975db3a026e8a1](https://github.com/Lincoln20030413/DDSR/assets/72965675/d4aa3bc5-64b9-4de7-b700-1ff22a0ad86e)





