import functools
import torch
from torch import nn
import torch.nn.functional as F
from models.module_util import make_layer, initialize_weights
from models.commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from models.commons import ResnetBlock, Upsample, Block, Downsample


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, get_fea=False):
        feas = []
        x = (x + 1) / 2
        fea_first = fea = self.conv_first(x)
        for l in self.RRDB_trunk:
            fea = l(fea)
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        feas.append(fea)

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))

        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(0, 1)
        out = out * 2 - 1
        if get_fea:
            return out, feas
        else:
            return out


class Unet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), cond_dim=32, flag = "ker"):
        super().__init__()
        dims = [1, *map(lambda m: dim * m, dim_mults)] #
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0
        rrdb_num_block = 8
        sr_scale = 2


        self.cond_proj = nn.ConvTranspose2d(cond_dim * ((rrdb_num_block + 1) // 3),
                                            dim, sr_scale* 2, sr_scale,
                                            sr_scale // 2)

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=dim, groups=groups, flag = flag),
                ResnetBlock(dim_out, dim_out, time_emb_dim=dim, groups=groups, flag = flag),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups, flag = flag)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim, groups=groups, flag = flag)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim, groups=groups, flag = flag),
                ResnetBlock(dim_in, dim_in, time_emb_dim=dim, groups=groups, flag = flag),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            Block(dim, dim, groups=groups, flag = flag),
            nn.Conv2d(dim, 1, 1)
        )


    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x, time, cond, img_lr_up):
        t = self.time_pos_emb(time)
        
        t = self.mlp(t).to(x.device)
        

        h = []
        cond = self.cond_proj(torch.cat((cond[2::3]), dim=1))
        # print(cond.shape)
        cond = cond[:, :, 0:24, 0:24]
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            
            x = resnet(x, t)
            x = resnet2(x, t)
            

            if i == 0:
                x = x + cond

            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)

        x = self.mid_block2(x, t)
        for resnet, resnet2, upsample in self.ups:
            a = h.pop()

            x = torch.cat((x, a), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)
        


        return self.final_conv(x)

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
