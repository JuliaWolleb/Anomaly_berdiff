"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import sys
sys.path.append("..")
sys.path.append(".")
import numpy as np
import torch
import torch.distributed as dist
from guided_diffusion.train_util import visualize

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import sys
import argparse
import torch as th
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8850)

sys.path.append('../BinaryLatentDiffusion')

from models.binaryae import BinaryAutoEncoder, Generator
from hparams import get_sampler_hparams
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, \
    get_sampler, get_online_samples, get_online_samples_guidance, get_samples_test, get_samples_temp, get_samples_loop


def main():

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
    device = torch.device("cuda:0")
    print('device', device)
    print('got until here, I was able to load binaryAE')


    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale


    args, unknown = create_argparser().parse_known_args()
    print(args)
    print(unknown)
    print('args', args.dataset)

    dist_util.setup_dist()
    print('device',dist_util.dev() )
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    print('loaded model', args.model_path)
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        # if args.class_cond:
        #     classes = th.randint(
        #         low=1, high=2, size=(args.batch_size,), device=dist_util.dev()
        #     )
        #     print('classes', classes)
        #     model_kwargs["y"] = classes
        print('start sampling loop')
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 32 , 128, 128),
           # clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        print('sample', sample.shape)
      #  viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="generated sample code 0"))

        reconstruction,_,_ = bergan(sample, code_only=False, code=sample)
        print('did reconstruction', reconstruction.shape)
        viz.image(visualize(reconstruction[0, 0, ...]), opts=dict(caption="generated reconstruction"))

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        s=torch.tensor(sample)
        torch.save(s, './tensor.pt')

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                torch.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    # if args.class_cond:
    #     label_arr = np.concatenate(all_labels, axis=0)
    #     label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        # if args.class_cond:
        #     np.savez(out_path, arr, label_arr)
    #    else:
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
       # clip_denoised=True,
        num_samples=1,
        batch_size=1,
        data_dir='/home/juliawolleb/PycharmProjects/Python_Tutorials/BinaryLatentDiffusion/data/chexpert',
        use_ddim=True,
        model_path="./results/latentchex630000.pt",
        use_fp16=False,
        img_channels=4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        microbatch=-1,  # -1 disables microbatches
        # ema_rate=0.9999,  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,
        resume_checkpoint='',
        fp16_scale_growth=1e-3,
        dataset='brats',
        ae_load_dir='../BinaryLatentDiffusion/logs/binaryae_custom128',
        ae_load_step=2000,
        sampler="bld",
        codebook_size=32,
        nf=32,
        img_size=256,
        latent_shape=[1, 128, 128],
        n_channels=1,
        ch_mult=[1, 2]
    )
    H = get_sampler_hparams()
    print('got H', H.img_size)
    defaults.update(model_and_diffusion_defaults())
    print('got until here2')
    # defaults.update(H)
    print('defaults', defaults.values())
    print('got until here22')
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
