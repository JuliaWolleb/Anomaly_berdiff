"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import sys
from torch.autograd import Variable
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
import blobfile as bf
import torch as th
os.environ['OMP_NUM_THREADS'] = '8'

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from visdom import Visdom
import numpy as np
viz = Visdom(port=8850)
loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='classification loss'))
val_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='validation loss'))
acc_window= viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='acc', title='accuracy'))

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.train_util import visualize
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict

import sys
sys.path.append('../BinaryLatentDiffusion')

from models.binaryae import BinaryAutoEncoder, Generator
from hparams import get_sampler_hparams
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, \
    get_sampler, get_online_samples, get_online_samples_guidance, get_samples_test, get_samples_temp, get_samples_loop


def main():
    dist_util.setup_dist()
    logger.configure()
    print('we are in the right function')
    H = get_sampler_hparams()
    print('H load dir', H.ae_load_dir)
    H.ae_load_step = 20000
    H.data_dir = './data/chexpert/training'
    H.ae_load_dir = '../BinaryLatentDiffusion/logs/binaryae_custom128'
    H.n_channels = 1
    H.sampler = 'bld'
    H.dataset = 'chexpert'
    H.amp = True
    H.norm_first = True
    H.codebook_size = 32
    H.nf = 32
    H.img_size = 256


    ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['encoder', 'quantize', 'generator'],
        remove_component_from_key=False
    )

    bergan = BinaryAutoEncoder(H)
    bergan.load_state_dict(ae_state_dict, strict=True)
    bergan = bergan.cuda()
    del ae_state_dict
    device = th.device("cuda:0")
    print('device', device)


    print('got until here, I was able to load binaryAE')

    args, unknown = create_argparser().parse_known_args()
    print(args)
    # Namespace(foo='BAR')
    print(unknown)
    # ['spam']
    #  args=args2.parse_args()
    print('args', args.dataset)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys()),
    )
    model.to(dist_util.dev())
    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion, maxt=1000
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        resume_step=15000
        if dist.get_rank() == 0:
            logger.log(
                f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            model.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=False, initial_lg_loss_scale=16.0
    )


    logger.log("creating data loader...")

    if args.dataset == 'brats':
        ds = BRATSDataset(args.data_dir, test_flag=False)
        datal = th.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True)
        data = iter(datal)

    elif args.dataset == 'chexpert':
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
        print('dataset is chexpert')



    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        print('resume_checkpoint', resume_step)
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"optchexclass{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")


    def forward_backward_log(data_loader, step, prefix="val"):
        if args.dataset=='brats':
            batch, extra, labels,_ , _ = next(data_loader)
            print('IS BRATS')

        elif  args.dataset=='chexpert':
          #  print('we are about to load data')
            batch, extra = next(data_loader)
        #    print('batch', batch.shape)
         #   print('extra', extra)
            labels = extra["y"].to(dist_util.dev())
           # print('IS CHEXPERT')


        batch = batch.to(dist_util.dev())
        labels= labels.to(dist_util.dev())
        code = bergan(batch, code_only=True).detach()
      #  viz.image(visualize(code[0, 0, ...]), opts=dict(caption="clean input code"))
       # viz.image(visualize(batch[0, 0, ...]), opts=dict(caption="org input image"))

        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], dist_util.dev())
            code = diffusion.q_sample(code, t)#put noise on batch
          #  viz.image(visualize(code[0, 0, ...]), opts=dict(caption="noisy input code _" + str(t)))


        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, code, labels, t)
        ):
          
            sub_batch = Variable(sub_batch, requires_grad=True)
            logits = model(sub_batch, timesteps=sub_t)
         
            loss = F.cross_entropy(logits, sub_labels, reduction="none")
            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@2"] = compute_top_k(
                logits, sub_labels, k=2, reduction="none"
            )
            log_loss_dict(diffusion, sub_t, losses)

            loss = loss.mean()
            if prefix=="train":
                viz.line(X=th.ones((1, 1)).cpu() * step, Y=th.Tensor([loss]).unsqueeze(0).cpu(),
                     win=loss_window, name='loss_cls',
                     update='append')

            else:

                output_idx = logits[0].argmax()
                print('outputidx', output_idx)
                output_max = logits[0, output_idx]
                print('outmax', output_max, output_max.shape)
                output_max.backward()
                saliency, _ = th.max(sub_batch.grad.data.abs(), dim=1)
                print('saliency', saliency.shape)
                viz.heatmap(visualize(saliency[0, ...]))
              #  viz.image(visualize(sub_batch[0, 0,...]))
              #  viz.image(visualize(sub_batch[0, 1, ...]))
                th.cuda.empty_cache()


            if loss.requires_grad and prefix=="train":
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

        return losses

    correct=0; total=0
    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        losses = forward_backward_log(data, step + resume_step)
        #try:
         #   losses = forward_backward_log(data, step + resume_step)
       # except:
        #    data = iter(datal)
          #  losses = forward_backward_log(data, step + resume_step)

        correct+=losses["val_acc@1"].sum()
        total+=args.batch_size
        if step%100==0:
            acctrain=correct/total
            correct=0
            total=0
            viz.line(X=th.ones((1, 1)).cpu() * step, Y=th.Tensor([acctrain]).unsqueeze(0).cpu(),
                     win=acc_window, name='train_acc',
                     update='append')

        mp_trainer.optimize(opt)
          
        if not step % args.log_interval:
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"chexclass{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"optchexclass{step:06d}.pt"))

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)



def create_argparser():
    defaults = dict(
        data_dir="../BinaryLatentDiffusion/data/chexpert/train",
        val_data_dir="",
        noised=True,
        schedule_sampler="uniform",
        lr=1e-4,
        amp=False,
        img_channels=4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        iterations=150000,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,  # -1 disables microbatches
       # ema_rate=0.9999,  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="./results/chexclass015000.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='chexpert',
        ae_load_dir = '../BinaryLatentDiffusion/logs/binaryae_custom128',
        ae_load_step= 2000,
        sampler="bld",
        codebook_size=32,
        nf=32,
        img_size=256,
        latent_shape=[1, 128, 128],
        n_channels = 1,
        ch_mult=[1,2]
    )
    H = get_sampler_hparams()
    print('got H', H.img_size)
    defaults.update(classifier_and_diffusion_defaults())
    print('got until here2')
    #defaults.update(H)
    print('defaults', defaults.values())
    print('got until here22')

    parser = argparse.ArgumentParser()
    print('created parser')
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
