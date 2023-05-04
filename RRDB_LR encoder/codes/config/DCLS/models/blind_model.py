import logging
import os
from collections import OrderedDict

import torchvision.utils as tvutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.lr_scheduler as lr_scheduler
import models.networks as networks
from models.modules.loss import CharbonnierLoss, CorrectionLoss

from .base_model import BaseModel
from utils import BatchBlur

from models.diffsr_modules import Unet, RRDBNet
from models.diffusion import GaussianDiffusion

from utils.matlab_resize_yuan import imresize

logger = logging.getLogger("base")


class B_Model(BaseModel):
    def build_model(self):
        hidden_size = 64
        dim_mults = '1|2|2|4'
        dim_mults = [int(x) for x in dim_mults.split('|')]
        rrdb_num_feat = 32
        denoise_fn = Unet(
            hidden_size, out_dim=3, cond_dim=rrdb_num_feat, dim_mults=dim_mults)
        # if hparams['use_rrdb']:
        #     rrdb = RRDBNet(3, 3, rrdb_num_feat, hparams['rrdb_num_block'],
        #                    rrdb_num_feat // 2)
        #     if hparams['rrdb_ckpt'] != '' and os.path.exists(hparams['rrdb_ckpt']):
        #         load_ckpt(rrdb, hparams['rrdb_ckpt'])
        # else:
        rrdb = None
        self.model = GaussianDiffusion(
            denoise_fn=denoise_fn,
            rrdb_net=rrdb,
            timesteps=100,
            loss_type='l1'
        )
        self.global_step = 0
        return self.model
    
    def __init__(self, opt):
        super(B_Model, self).__init__(opt)
        self.scale = opt['scale']
        self.model = self.build_model()
        self.model = self.model.to(self.device)
        # self.model.GPU()
        if opt["dist"]:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # # define network and load pretrained models
        # self.netG = networks.define_G(opt).to(self.device)
        # if opt["dist"]:
        #     self.netG = DistributedDataParallel(
        #         self.netG, device_ids=[torch.cuda.current_device()]
        #     )
        # else:
        #     self.netG = DataParallel(self.netG)
        # # print network
        # self.print_network()
        # self.load()

        # reformulate kernel and compute L1 loss
        self.cri_kernel = CorrectionLoss(scale=self.scale, eps=1e-20).to(self.device)

        if self.is_train:
            train_opt = opt["train"]
            print(train_opt)
            # self.netG.train()

            # self.l_pix_w = train_opt["pixel_weight"]

            # optimizers
            wd_G = train_opt["weight_decay_G"] if train_opt["weight_decay_G"] else 0

            params = list(self.model.named_parameters())
            # if not hparams['fix_rrdb']:
            #     params = [p for p in params if 'rrdb' not in p[0]]
            params = [p[1] for p in params]
            self.optimizer_G = torch.optim.Adam(
                [
                    {"params": params, "lr": 0.0002},
                ],
                weight_decay=wd_G,
                betas=(train_opt["beta1"],train_opt["beta2"]),
            )       
            # self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], momentum=0.9)
            self.optimizers = self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt["lr_scheme"] == "MultiStepLR":
                    self.schedulers.append(
                        torch.optim.lr_scheduler.StepLR(self.optimizer_G, 100000, gamma=0.5)
                    )  #decay_steps = 100000
            elif train_opt["lr_scheme"] == "CosineAnnealingLR_Restart":
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer,
                            train_opt["T_period"],
                            eta_min=train_opt["eta_min"],
                            restarts=train_opt["restarts"],
                            weights=train_opt["restart_weights"],
                        )
                    )
            else:
                print("MultiStepLR learning rate scheme is enough.")

            self.log_dict = OrderedDict()

    def init_model(self, scale=0.1):
        # Common practise for initialization.
        for layer in self.netG.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
                layer.weight.data *= scale  # for residual block
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, a=0, mode="fan_in")
                layer.weight.data *= scale
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, nn.BatchNorm2d):
                init.constant_(layer.weight, 1)
                init.constant_(layer.bias.data, 0.0)

    def feed_data(self, LR_img, GT_img=None, kernel=None, ker_map=None, lr_blured=None, lr=None, lr_up=None):
        self.var_L = LR_img.to(self.device)
        if not (GT_img is None):
            self.real_H = GT_img.to(self.device)
        if not (ker_map is None):
            self.real_ker_map = ker_map.to(self.device)
        if not (kernel is None):
            self.real_kernel = kernel.to(self.device)
        if not (lr_blured is None):
            self.lr_blured = lr_blured.to(self.device)
        if not (lr is None):
            self.lr = lr.to(self.device)
        if not (lr_up is None):
            self.lr_up = lr_up.to(self.device)


    def optimize_parameters(self, step):
        hr = self.real_H
        # hr = hr.to(self.device)
        # print(hr.device)
        lr = self.var_L
        # lr = lr.to(self.device)
        # print(lr.device)
        lr_up = self.lr_up
        # lr_up = lr_up.to(self.device)
        # print(lr_up.device)
        losses, _, _ = self.model(hr, lr, lr_up)
        total_loss = sum(losses.values())

        self.optimizer_G.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.netG.parameters(), .1)
        # torch.nn.utils.clip_grad_value_(self.netG.parameters(), .1)
        self.optimizer_G.step()


    def test(self):
        lr = self.var_L
        hr = self.real_H
        lr_up = self.lr_up 
        with torch.no_grad():
            self.fake_SR, _ = self.model.sample(lr, lr_up, hr.shape)


    # def test_x8(self):
    #     # from https://github.com/thstkdgus35/EDSR-PyTorch
    #     self.netG.eval()

    #     def _transform(v, op):
    #         # if self.precision != 'single': v = v.float()
    #         v2np = v.data.cpu().numpy()
    #         if op == "v":
    #             tfnp = v2np[:, :, :, ::-1].copy()
    #         elif op == "h":
    #             tfnp = v2np[:, :, ::-1, :].copy()
    #         elif op == "t":
    #             tfnp = v2np.transpose((0, 1, 3, 2)).copy()

    #         ret = torch.Tensor(tfnp).to(self.device)
    #         # if self.precision == 'half': ret = ret.half()

    #         return ret

    #     lr_list = [self.var_L]
    #     for tf in "v", "h", "t":
    #         lr_list.extend([_transform(t, tf) for t in lr_list])
    #     with torch.no_grad():
    #         sr_list = [self.netG(aug)[0] for aug in lr_list]
    #     for i in range(len(sr_list)):
    #         if i > 3:
    #             sr_list[i] = _transform(sr_list[i], "t")
    #         if i % 4 > 1:
    #             sr_list[i] = _transform(sr_list[i], "h")
    #         if (i % 4) % 2 == 1:
    #             sr_list[i] = _transform(sr_list[i], "v")

    #     output_cat = torch.cat(sr_list, dim=0)
    #     self.fake_H = output_cat.mean(dim=0, keepdim=True)
    #     self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["LQ"] = self.var_L.detach()[0].float().cpu()
        out_dict["SR"] = self.fake_SR.detach()[0].float().cpu()
        out_dict["GT"] = self.real_H.detach()[0].float().cpu()
        # out_dict["ker"] = self.fake_ker.detach()[0].float().cpu()
        out_dict["Batch_SR"] = (
            self.fake_SR.detach().float().cpu()
        )  # Batch SR, for train
        return out_dict

    # def print_network(self):
    #     s, n = self.get_network_description(self.netG)
    #     if isinstance(self.netG, nn.DataParallel) or isinstance(
    #         self.netG, DistributedDataParallel
    #     ):
    #         net_struc_str = "{} - {}".format(
    #             self.netG.__class__.__name__, self.netG.module.__class__.__name__
    #         )
    #     else:
    #         net_struc_str = "{}".format(self.netG.__class__.__name__)
    #     if self.rank <= 0:
    #         logger.info(
    #             "Network G structure: {}, with parameters: {:,d}".format(
    #                 net_struc_str, n
    #             )
    #         )
    #         logger.info(s)

    # def load(self):
    #     load_path_G = self.opt["path"]["pretrain_model_G"]
    #     if load_path_G is not None:
    #         logger.info("Loading model for G [{:s}] ...".format(load_path_G))
    #         self.load_network(load_path_G, self.netG, self.opt["path"]["strict_load"])

    def save(self, iter_label):
        self.save_network(self.model,  "G", iter_label)
