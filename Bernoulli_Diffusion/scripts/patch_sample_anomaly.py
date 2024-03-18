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
import numpy as np
import nibabel
import torch
import torch.distributed as dist
from guided_diffusion.train_util import visualize
import scipy
from guided_diffusion import dist_util, logger
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
from guided_diffusion.script_util_gauss import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

import torchvision.transforms.functional as F
from evaluation import apply_2d_median_filter,filter_2d_connected_components
torch.manual_seed(1)
import random
random.seed(1)

import sys
import argparse
import torch as th
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler

from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8852)

sys.path.append('../BinaryLatentDiffusion')

from models.binaryae import BinaryAutoEncoder, Generator
from hparams import get_sampler_hparams
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, \
    get_sampler, get_online_samples, get_online_samples_guidance, get_samples_test, get_samples_temp, get_samples_loop

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def visualize2(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)*2-1
    return normalized_img



def main():


    print('we are in the right function')

    data_dir = "./data/brats/val_diseased_with_labels"
 #   data_dir = "./data/OCT/DRUSEN"



    device = torch.device("cuda:0")
    print('device', device)

    args, unknown = create_argparser().parse_known_args()
    print(args)
    print(unknown)
    print('args', args.dataset)

    datal = load_data(
        data_dir=data_dir,
        batch_size=1,
        image_size=256,
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
    all_labels = []
    mask_list=[]
    all_dice=0
    non_zerodice = 0
    nz=0
    while len(all_images) * args.batch_size < args.num_samples:
        data, out = next(val_loader)
        number=out["name"]
        print('number', number)
     #   if number[0] != 'DRUSEN-349021-1':
      #      continue
        batch=data[:,:4,...]
        label=data[:,-1:,...]
      #  batch=visualize2(data)
       # mask = np.zeros_like(batch[0])
        N = 4  #we have 4 channels

        #m_resized = transform(torch.tensor(mask2))






        #viz.image(visualize(data[0, 4, ...]), opts=dict(caption="GT label"))
      #  viz.image(visualize(batch[0, 3, ...]), opts=dict(caption="img input 1"))

      #  viz.image(visualize(batch[0, 0, ...]), opts=dict(caption="img input 3"))

        #viz.image(visualize(Cmask[0, 0, ...]), opts=dict(caption="cmask"))


        model_kwargs = {}
        # if args.class_cond:
        #     classes = th.randint(
        #         low=1, high=2, size=(args.batch_size,), device=dist_util.dev()
        #     )
        #     print('classes', classes)
        #     model_kwargs["y"] = classes
        img=batch.to(dist_util.dev())#.cuda()

       # viz.image(visualize(code[0, 0, ...]), opts=dict(caption="code 0"))

        sample_fn = (
           diffusion.p_sample_loop_patch# if not args.use_ddim else diffusion.ddim_sample_loop_anomaly
        )
        start.record()
        sample, _,_ = sample_fn(
            model,
            (1,4 , 256, 256),
            img,
            #Cmask,
           # clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
     #   viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="generated sample code 0"))
        reconstruction=sample.cpu()
        end.record()
        torch.cuda.synchronize()
        print('elapsed time', start.elapsed_time(end))


        reconstruction=torch.clamp(reconstruction,0,1).detach().cpu()
       # reconstructed_org=torch.clamp(reconstructed_org,0,1)

        #reconstruction_mask=torch.clamp(reconstructed_mask,0,1)

       # xx, _, _ = bergan(sample, code_only=False)
       # print('xx', xx.shape)

     #   viz.image(visualize(reconstruction[0, 0, ...]), opts=dict(caption="generated reconstruction0"))
     #   viz.image(visualize(reconstruction[0, 1, ...]), opts=dict(caption="generated reconstruction1"))
      #  viz.image(visualize(reconstruction[0, 2, ...]), opts=dict(caption="generated reconstruction2"))
      #  viz.image(visualize(reconstruction[0, 3, ...]), opts=dict(caption="generated reconstruction3"))

      #  viz.image(visualize(reconstruction[1, 0, ...]), opts=dict(caption="generated reconstruction0"))
       # viz.image(visualize(reconstruction[1, 1, ...]), opts=dict(caption="generated reconstruction1"))
      #  viz.image(visualize(reconstruction[1, 2, ...]), opts=dict(caption="generated reconstruction2"))
        #viz.image(visualize(reconstruction[1, 3, ...]), opts=dict(caption="generated reconstruction3"))



        final_img=th.zeros(1,4,256,256)
        for r in range(4):
            if r == 0:
                final_img[0, :, :128, :128]=reconstruction[0,:, :128, :128]

            if r == 1:
                final_img[0, :, 128:, :128] = reconstruction[1, :, 128:, :128]
            if r == 2:
                final_img[0, :, :128, 128:] = reconstruction[2, :, :128, 128:]

            if r == 3:
                final_img[0, :, 128:, 128:] = reconstruction[3, :, 128:, 128:]

        viz.image(visualize(final_img[0, 3, ...]), opts=dict(caption="final_img 0"))
       # viz.image(visualize(final_img[0, 1, ...]), opts=dict(caption="final_img 1"))
        #viz.image(visualize(final_img[0, 2, ...]), opts=dict(caption="final_img 2"))
      #  viz.image(visualize(final_img[0, 3, ...]), opts=dict(caption="final_img 3"))


        #viz.image(reconstruction[0, 1, ...], opts=dict(caption="generated reconstruction1"))
      #  viz.image(reconstruction[0, 2, ...], opts=dict(caption="generated reconstruction2"))
       # viz.image(reconstruction[0, 3, ...], opts=dict(caption="generated reconstruction3"))
        difference=torch.abs(final_img[0,...].cpu()-batch[0,...].cpu()).sum(dim=0)
        viz.heatmap(difference, opts=dict(caption="median mse 0", colormap='Jet'))
       # diff = torch.abs(final_img[0,...].cpu() - batch[0,...].cpu()).square()

        final_imag=torch.cat((batch[0,...], final_img[0,...], difference[None,...]), dim=0)
        print('final image', final_imag.shape)

        final_imag = nibabel.Nifti1Image(np.array(final_imag), affine=np.eye(4))
        #name = os.path.join("./Reconstruction_patch_simplex4/", number[0] + '.nii.gz')
       # nibabel.save(final_imag, name)


      #  DSC = dice_score(torch.tensor(diffs_thresholded), label.cpu())

      #  all_dice+=DSC
      #  if DSC>0:
       #     non_zerodice+=DSC
       #     nz+=1
      #  print('dice', DSC)

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        s=torch.tensor(sample)
        torch.save(s, './tensor.pt')

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        # if args.class_cond:
        #     gathered_labels = [
        #         torch.zeros_like(classes) for _ in range(dist.get_world_size())
        #     ]
        #     dist.all_gather(gathered_labels, classes)
        #     all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
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
    print('mean dice', all_dice/args.num_samples)
   # print('mean dice nonzero', non_zerodice / nz)
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
       # clip_denoised=True,
        num_samples=1038,
        batch_size=1,
        data_dir="./data/brats/train_healthy",#'/home/juliawolleb/PycharmProjects/Python_Tutorials/BinaryLatentDiffusion/data/chexpert',
        use_ddim=True,
        model_path='./results/patch_global_simplex050000.pt' ,  #./results/OCT_patch_global070000.pt',   #'./results/latentbrats_646464healthy120000.pt',  #'./results/latentOCT_1283232healthy300000.pt', # './results/latentbrats_1283232healthy110000.pt',#'./results/latentbrats_646464healthy070000.pt',
                                                            #"./results/latentbrats_32healthy290000_epsilon.pt", #'./results/latentbrats_64healthy280000.pt', patch_global_simplex020000OCT_patch_global070000.pt
        use_fp16=False,
        img_channels=8,
        weight_decay=0.0,
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


    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
