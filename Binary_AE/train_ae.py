# file for running the training of the binaryae
import torch
import numpy as np
import copy
import time
import random
from torchvision.transforms.functional import hflip
from models.binaryae import BinaryGAN, BinaryAutoEncoder
from hparams import get_vqgan_hparams
from utils.data_utils import get_data_loaders, cycle
from utils.train_utils import EMA, visualize
from utils.log_utils import log, log_stats, save_model, save_stats, save_images, \
                             config_log, start_training_log
from utils.vqgan_utils import load_binaryae_from_checkpoint
import misc
import os
import torch.distributed as dist
import torchvision
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append('../Bernoulli_Diffusion')
from Bernoulli_Diffusion.guided_diffusion.image_datasets import load_data
from Bernoulli_Diffusion.guided_diffusion.bratsloader import BRATSDataset
#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
    binaryae = BinaryGAN(args).cuda()


    datal = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )

    train_loader = iter(datal)


    if args.ema:
        ema = EMA(args.ema_beta)
        ema_binaryae = copy.deepcopy(binaryae)
    else:
        ema_binaryae = None
    
   # if args.distributed:
       # binaryae = torch.nn.parallel.DistributedDataParallel(binaryae, device_ids=[args.gpu], find_unused_parameters=True)
    binaryae_without_ddp = binaryae

    optim = torch.optim.Adam(binaryae_without_ddp.ae.parameters(), lr=args.lr)
    d_optim = torch.optim.Adam(binaryae_without_ddp.disc.parameters(), lr=args.lr)

    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

    losses = np.array([])
    mean_losses = np.array([])
    val_losses = np.array([])
    recon_losses = np.array([])
    latent_ids = []
    fids = np.array([])
    best_fid = float('inf')

    start_step = 0
    log_start_step = 0
    eval_start_step = args.steps_per_eval
    if args.load_step > 0:
        start_step = args.load_step + 1  # don't repeat the checkpointed step
        binaryae, optim, d_optim, ema_binaryae, train_stats = load_binaryae_from_checkpoint(args, binaryae, optim, d_optim, ema_binaryae)

        # stats won't load for old models with no associated stats file
        if train_stats is not None:
            losses = train_stats["losses"]
            mean_losses = train_stats["mean_losses"]
            val_losses = train_stats["val_losses"]
            fids = train_stats["fids"]
            best_fid = train_stats["best_fid"]
            args.steps_per_log = train_stats["steps_per_log"]
            args.steps_per_eval = train_stats["steps_per_eval"]

            log('Loaded stats')
        else:
            if args.steps_per_eval:
                if args.steps_per_eval == 1:
                    eval_start_step = start_step
                else:
                    eval_start_step = start_step + args.steps_per_eval - start_step % args.steps_per_eval



    log(f"ae params: {(sum(p.numel() for p in binaryae_without_ddp.ae.parameters())/1e6)}M")
    log(f"disc params:{(sum(p.numel() for p in binaryae_without_ddp.disc.parameters()))/1e6}M")
    log(f"total params:{(sum(p.numel() for p in binaryae_without_ddp.ae.parameters())/1e6) + (sum(p.numel() for p in binaryae_without_ddp.disc.parameters()))/1e6}M")

    # for step in range(start_step, args.train_steps):
    step = start_step - 1
    epoch = -1
    while True:
        epoch += 1

       # train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            step += 1

            step_start_time = time.time()

            if isinstance(batch, list):
                x = batch[0]
            elif isinstance(batch, dict):
                x = batch['image_tensor']
            else:
                x = batch


            if args.horizontal_flip:
                if random.random() <= 0.5:
                    x = hflip(x)

            x = x.cuda()

            if args.amp:
                optim.zero_grad()
                with torch.cuda.amp.autocast():
                    x_hat, stats = binaryae(x, step)

                scaler.scale(stats['loss']).backward()
                scaler.step(optim)
                scaler.update()
            else:
                x_hat, stats = binaryae(x, step)
                optim.zero_grad()
                stats['loss'].backward()
                optim.step()
                print('took step')


            if step > args.disc_start_step:
                if args.amp:
                    d_optim.zero_grad()
                    with torch.cuda.amp.autocast():
                        stats = binaryae.module.disc_iter(x_hat.detach(), x, stats)
                    d_scaler.scale(stats['d_loss']).backward()
                    d_scaler.step(d_optim)
                    d_scaler.update()
                else:
                    d_optim.zero_grad()
                    stats = binaryae.module.disc_iter(x_hat, x, stats)
                    stats['d_loss'].backward()
                    d_optim.step()

            torch.cuda.synchronize()

            if step == 1500000:
                for param_group in optim.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                print('lr decay')
            
            # collect latent ids
            # log codebook usage
         #   if dist.get_rank() == 0:

            code = stats['latent_ids'].cpu().contiguous()
            code = code > 0.5
            latent_ids.append(code)
            if step % 200 == 0 and step > 0:
                latent_ids = torch.cat(latent_ids, dim=0).permute(1,0,2,3).reshape(args.codebook_size, -1)
                codesample_size = latent_ids.shape[1]
                latent_ids = latent_ids * 1.0

                latent_ids = latent_ids.sum(-1)

                odd_idx = ((latent_ids == 0) * 1.0).sum() + ((latent_ids == codesample_size) * 1.0).sum()
                    
                if int(args.codebook_size - odd_idx) != args.codebook_size:
                        log(f'Codebook size: {args.codebook_size}   Unique Codes Used in Epoch: {args.codebook_size - odd_idx}')
                latent_ids = []
                
            if step % args.steps_per_log == 0:
                losses = np.append(losses, stats['loss'].item())
                mean_loss = np.mean(losses)
                stats['loss'] = mean_loss
                step_time = time.time() - step_start_time
                stats['step_time'] = step_time
                mean_losses = np.append(mean_losses, mean_loss)
                recon_losses = np.append(recon_losses, stats['l1'])
                losses = np.array([])

                log_stats(step, stats)

            if args.ema and step % args.steps_per_update_ema == 0 and step > 0:
                ema.update_model_average(ema_binaryae, binaryae)



            if step % args.steps_per_checkpoint == 0:# and step > args.load_step:

                save_model(binaryae_without_ddp, 'binaryae', step, args.log_dir)
                save_model(optim, 'ae_optim', step, args.log_dir)
                save_model(d_optim, 'disc_optim', step, args.log_dir)
                if args.ema:
                    save_model(ema_binaryae, 'binaryae_ema', step, args.log_dir)

                train_stats = {
                    'losses': losses,
                    'mean_losses': mean_losses,
                    'val_losses': val_losses,
                    'fids': fids,
                    'best_fid': best_fid,
                    'steps_per_log': args.steps_per_log,
                    'steps_per_eval': args.steps_per_eval,
                }
                save_stats(args, train_stats, step)
            
            if step == args.train_steps:
                exit()

if __name__ == '__main__':
    args = get_vqgan_hparams()
    config_log(args.log_dir)
    log('---------------------------------')
    log(f'Setting up training for binaryae on {args.dataset}')
    start_training_log(args)
    main(args)
