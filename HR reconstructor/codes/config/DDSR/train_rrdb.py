import torch
import torch.nn.functional as F
from diffsr_modules import RRDBNet
from hparams import hparams
from trainer import Trainer

def build_model(self):
    hidden_size = hparams['hidden_size']
    self.model = RRDBNet(3, 3, hidden_size, hparams['num_block'], hidden_size // 2)
    return self.model

def build_optimizer(self, model):
    return torch.optim.Adam(model.parameters(), lr=hparams['lr'])

def build_scheduler(self, optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, 200000, 0.5)

def training_step(self, sample):
    img_hr = sample['img_hr']
    img_lr = sample['img_lr']
    p = self.model(img_lr)
    loss = F.l1_loss(p, img_hr, reduction='mean')
    return {'l': loss, 'lr': self.scheduler.get_last_lr()[0]}, loss

def sample_and_test(self, sample):
    ret = {k: 0 for k in self.metric_keys}
    ret['n_samples'] = 0
    img_hr = sample['img_hr']
    img_lr = sample['img_lr']
    img_sr = self.model(img_lr)
    img_sr = img_sr.clamp(-1, 1)
    for b in range(img_sr.shape[0]):
        s = self.measure.measure(img_sr[b], img_hr[b], img_lr[b], hparams['sr_scale'])
        ret['psnr'] += s['psnr']
        ret['ssim'] += s['ssim']
        ret['lpips'] += s['lpips']
        ret['lr_psnr'] += s['lr_psnr']
        ret['n_samples'] += 1
    return img_sr, img_sr, ret
def train(self):
    model = self.build_model()
    optimizer = self.build_optimizer(model)
    self.global_step = training_step = load_checkpoint(model, optimizer, hparams['work_dir'])
    self.scheduler = scheduler = self.build_scheduler(optimizer)
    scheduler.step(training_step)
    dataloader = self.build_train_dataloader()
    print(dataloader)
    # '''
    train_pbar = tqdm(dataloader, initial=training_step, total=float('inf'),
                    dynamic_ncols=True, unit='step')
                    
    for batch in train_pbar:
        if training_step % hparams['val_check_interval'] == 0:
            with torch.no_grad():
                model.eval()
                self.validate(training_step)
            save_checkpoint(model, optimizer, self.work_dir, training_step, hparams['num_ckpt_keep'])
        model.train()
        batch = move_to_cuda(batch)
        losses, total_loss = self.training_step(batch)
        optimizer.zero_grad()

        total_loss.backward()
        optimizer.step()
        training_step += 1
        scheduler.step(training_step)
        self.global_step = training_step
        if training_step % 100 == 0:
            self.log_metrics({f'tr/{k}': v for k, v in losses.items()}, training_step)
        train_pbar.set_postfix(**tensors_to_scalars(losses))
    
