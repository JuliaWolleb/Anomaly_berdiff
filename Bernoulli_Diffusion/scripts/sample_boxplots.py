"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from skimage import exposure
import sys
sys.path.append("..")
sys.path.append(".")
import csv
import numpy as np
import nibabel
import torch
import torch.distributed as dist
from guided_diffusion.train_util import visualize
import scipy
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms.functional as F
from evaluation import apply_2d_median_filter,filter_2d_connected_components
torch.manual_seed(0)
import random
random.seed(0)

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

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()




def main():


    print('we are in the right function')
    H = get_sampler_hparams()
    print('got hparams')
    print('image size', H.img_size)
    print('H load dir', H.ae_load_dir)
    print('H data dir', H.data_dir)
    print('H n channels', H.n_channels)

    #H.n_channels=4
    H.sampler = 'bld'
    H.data_dir = "./data/brats/val_diseased_with_labels"
  #  H.data_dir = "./data/brats/val_healthy"
    sample_healthy=False
    sample_diseased=True
    H.dataset='chexpert'

    H.img_size=256

    H.sampler = 'bld'
    H.dataset = 'chexpert'
    H.amp = True
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

    device = torch.device("cuda:0")
    print('device', device)

    print('got until here, I was able to load binaryAE')
    args, unknown = create_argparser().parse_known_args()
    print(args)
    print(unknown)
    print('args', args.dataset)

    datal = load_data(
        data_dir=H.data_dir,
        batch_size=1,
        image_size=H.image_size,
    )
    print('dataset is brats')
    val_loader = iter(datal)

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
    List_mask=[]
    all_labels = []
    mask_list=[]
    all_dice=0
    non_zerodice = 0
    nz=0
    k=0
    while k < args.num_samples:
        k+=1
        data, out = next(val_loader)
        number=out["name"]
        print('number', number, 'k', k)
        batch=data[:,:4,...]
        label=data[:,-1:,...]
        mask = np.zeros_like(batch[0])
        N = 1  #we have 4 channels
        for j in range(N):
            mask = np.logical_or(mask, batch[0, j])
        mask = scipy.ndimage.morphology.binary_fill_holes(mask)*1
        mask2= mask[None,...]
        print('mask2', mask2.shape)
        Cmask= F.resize(torch.tensor(mask2[:,:1,...]), size = (32,32))
        Cmask[Cmask<0.5]=0
        Cmask[Cmask>= 0.5] = 1

        Cmask = Cmask.repeat(1, 128, 1, 1)


    #    viz.image(visualize(data[0, 4, ...]), opts=dict(caption="GT label"))
       # viz.image(visualize(batch[0, 1, ...]), opts=dict(caption="img input 1"))
      #  viz.image(visualize(batch[0, 2, ...]), opts=dict(caption="img input 2"))
      #  viz.image(visualize(batch[0, 3, ...]), opts=dict(caption="img input 3"))

        #viz.image(visualize(Cmask[0, 0, ...]), opts=dict(caption="cmask"))


        model_kwargs = {}
        # if args.class_cond:
        #     classes = th.randint(
        #         low=1, high=2, size=(args.batch_size,), device=dist_util.dev()
        #     )
        #     print('classes', classes)
        #     model_kwargs["y"] = classes
        img=batch.cuda()
        code = bergan(img, code_only=True).detach()
       # viz.image(visualize(code[0, 0, ...]), opts=dict(caption="code 0"))

        sample_fn = (
           diffusion.p_sample_loop_anomaly if not args.use_ddim else diffusion.ddim_sample_loop_anomaly
        )
        sample, Mask = sample_fn(
            model,
            (args.batch_size, 128 , 32, 32),
            code,
            Cmask,
           # clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        if sample_diseased:
            M = np.array(Mask.sum().cpu()).astype(int)

            mask_list.append(str(M))
            with open('dummylist_diseased', 'w') as f:
                # using csv.writer method from CSV package
                writer = csv.writer(f)
                writer.writerow(mask_list)
            continue
        if sample_healthy:
            M=np.array(Mask.sum().cpu()).astype(int)

            mask_list.append(str(M))

            with open('dummylist_healthy', 'w') as f:

                # using csv.writer method from CSV package
                writer = csv.writer(f)
                writer.writerow(mask_list)
            continue
      #  viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="generated sample code 0"))

    print('done')

def create_argparser():
    defaults = dict(
       # clip_denoised=True,
        num_samples=500,
        batch_size=1,
        data_dir="./data/brats/train_healthy",#'/home/juliawolleb/PycharmProjects/Python_Tutorials/BinaryLatentDiffusion/data/chexpert',
        use_ddim=True,
        model_path='./results/latentbrats_1283232healthy170000.pt',   #'./results/latentbrats_646464healthy120000.pt',  #'./results/latentOCT_1283232healthy300000.pt', # './results/latentbrats_1283232healthy110000.pt',#'./results/latentbrats_646464healthy070000.pt',
                                                            #"./results/latentbrats_32healthy290000_epsilon.pt", #'./results/latentbrats_64healthy280000.pt',latentbrats_1283232healthy020000
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
        ae_load_dir='binaryae_brats_custom128_64ch.pt',#'../BinaryLatentDiffusion/logs/binaryae_brats_custom32ch_epsilon',
        ae_load_step=260000,
        sampler="bld",
        codebook_size=64,
        nf=32,
        img_size=256,
        latent_shape=[1, 128, 128],
        n_channels=4,
        ch_mult=[1, 2],
        mean_type="epsilon"
    )
    H = get_sampler_hparams()
    print('got H', H.img_size)
    defaults.update(model_and_diffusion_defaults())
    print('got until here2')
    print('nchaneels', H.n_channels)
    # defaults.update(H)
    print('defaults', defaults.values())
    print('got until here22')
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
