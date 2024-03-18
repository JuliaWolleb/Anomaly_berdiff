"""
Train a diffusion model on images.
"""
import sys
import argparse
import torch as th
sys.path.append("..")
sys.path.append(".")
sys.path.append('./Binary_AE')
sys.path.append('./Bernoulli_Diffusion')
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
viz = Visdom(port=8852)

import torch


from models.binaryae import BinaryAutoEncoder, Generator
from hparams import get_sampler_hparams
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, \
    get_sampler, get_online_samples, get_online_samples_guidance, get_samples_test, get_samples_temp, get_samples_loop


def main2():

    print('we are in the right function')
    H = get_sampler_hparams()
    print('got hparams', H)
    print('H load dir', H.ae_load_dir)
    print('H latent shape', H.latent_shape)
    H.norm_first = True


    ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['encoder', 'quantize', 'generator'],
        remove_component_from_key=False
    )

    bergan = BinaryAutoEncoder(H)
    bergan.load_state_dict(ae_state_dict, strict=True)
    bergan = bergan.cuda()
    del ae_state_dict
    device = torch.device("cuda:1")
    print('device', device)


    print('got until here, I was able to load binaryAE', H.ae_load_dir, H.ae_load_step)
    #args= create_argparser().parse_args()

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
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print('device', dist_util.dev())

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    logger.log("creating data loader...")

    print('datadir', args.data_dir)
    datal = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        autoencoder=bergan,
        data=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=0.999,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset
    ).run_loop()

def create_argparser():
    defaults = dict(
        data_dir='/home/juliawolleb/PycharmProjects/Python_Tutorials/BerDiff/diffusion-anomaly-berdiff/data/brats/train_healthy',
        schedule_sampler="uniform",
        lr=1e-4,
        amp=False,
        img_channels=4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        log_interval=100,
        save_interval=10000,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='chexpert',
        ae_load_dir = '../BinaryLatentDiffusion/logs/binaryae_brats',
        ae_load_step= 00000,
        sampler="bld",
        codebook_size=64,
        nf=32,
        img_size=256,
        latent_shape=[1, 64, 64],
        n_channels = 4,
        ch_mult=[1,2,2],
        mean_type = "epsilon"
    )
    H = get_sampler_hparams()
    print('got H', H.img_size)
    defaults.update(model_and_diffusion_defaults())
    print('defaults', defaults.values())
    parser = argparse.ArgumentParser()
    print('created parser')
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main2()
