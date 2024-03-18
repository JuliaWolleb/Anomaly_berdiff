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
import scipy
import torch
from torchvision.utils import save_image

torch.manual_seed(0)
import random
random.seed(0)
from sklearn.metrics import auc, roc_curve
import sys

sys.path.append("..")
sys.path.append(".")
from skimage.measure import regionprops, label
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from visdom import Visdom
viz = Visdom(port=8852)

sys.path.append('../BinaryLatentDiffusion')

from hparams import get_sampler_hparams

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def PSNR(recon, real):
    se = (real - recon).square()
    mse = torch.mean(se, dim=list(range(len(real.shape))))
    psnr = 20 * torch.log10(torch.max(real) / torch.sqrt(mse))
    return psnr.detach().cpu().numpy()


def SSIM(real, recon):
    return ssim(real.detach().cpu().numpy(), recon.detach().cpu().numpy(), channel_axis=2)

def apply_2d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    for i in range(4):
        volume[i,...] = scipy.ndimage.filters.median_filter(volume[i,...], (kernelsize, kernelsize))
    return volume

def filter_2d_connected_components(volume):
    sz = None
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    cc_volume = label(volume, connectivity=2)
    props = regionprops(cc_volume)
    for prop in props:
        if prop['filled_area'] <= 7:
            volume[cc_volume == prop['label']] = 0

    if sz is not None:
        volume = np.reshape(volume, [sz[0], sz[1], sz[2], sz[3]])
    return volume


def IoU(real, recon):
    import numpy as np
    real = real.cpu().numpy()
    recon = recon.cpu().numpy()
    intersection = np.logical_and(real, recon)
    union = np.logical_or(real, recon)
    return np.sum(intersection) / (np.sum(union) + 1e-8)


def precision(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FP = ((real_mask == 1) & (recon_mask == 0))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FP)).float() + 1e-6)



def recall(real_mask, recon_mask):
    TP = ((real_mask == 1) & (recon_mask == 1))
    FN = ((real_mask == 0) & (recon_mask == 1))
    return torch.sum(TP).float() / ((torch.sum(TP) + torch.sum(FN)).float() + 1e-6)


def FPR(real_mask, recon_mask):
    FP = ((real_mask == 1) & (recon_mask == 0))
    TN = ((real_mask == 0) & (recon_mask == 0))
    return torch.sum(FP).float() / ((torch.sum(FP) + torch.sum(TN)).float() + 1e-6)


def ROC_AUC(real_mask, square_error):
    if type(real_mask) == torch.Tensor:
        return roc_curve(real_mask.detach().cpu().numpy().flatten(), square_error.detach().cpu().numpy().flatten())
    else:
        return roc_curve(real_mask.flatten(), square_error.flatten())


def AUC_score(fpr, tpr):
    return auc(fpr, tpr)

import argparse
import os

import sys
from sklearn.metrics import roc_auc_score
#from metrics import dc, jc, hd95, hd1, getHausdorff
from scipy.spatial.distance import directed_hausdorff
sys.path.append("..")
sys.path.append(".")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import nibabel as nib
from skimage.filters import threshold_otsu

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def hd(u,v):
    a=directed_hausdorff(u, v)[0]
    b=directed_hausdorff(u, v)[0]
    c=max(a,b)
    print('abc', a,b,c)
    return hd95


#x=[0,250,500,750,1000,1500]
#y1=[0,0,0.61,0,0.53,0.51]
import matplotlib.pyplot as plt
#plt.plot(x, y1,'bo-')
#viz.matplot(plt)



PathDicomstripped = "./Reconstruction_DRUSEN_300_0.8/"

#PathDicomstripped = "./Reconstruction_LDM_OCT/"
#PathDicomstripped = "/data/OCT/autoddpm_results/val_results"
#PathDicomstripped = "/home/juliawolleb/PycharmProjects/Python_Tutorials/BerDiff/AnoDDPM/AnoDDPM/reconstructions_simplex_OCT"
PathInput = "./data/OCT/DRUSEN"

k=0
tot=0
totj=0
tothd=0
b=0
f=0
B=0
total_auc=0


def main():
    all_dice=0
    non_zerodice=0
    nz=0
    k=0
    for dirName, subdirList, fileList in os.walk(PathDicomstripped):  # used to be dicomstripped
        s = dirName.split("/", -1)
        print('s', s[-1], s[-2])
        #     if 't1n_3d' in subdirList:
        #         path=os.path.join(dirName, 't1n_3d'))
        for filename in fileList:
            s = filename.split("_", 1)
            number = s[0]
         #   number = s[0].split(".", 1)[0]
            print('number', number)
          #  print('number', number)
            if '228939' not in number:# != 'DRUSEN-349021-1':
                continue

            path = os.path.join(dirName, filename)
          #  print('dirName', dirName, 'filename', filename, path)

            output = nib.load(path)
            output = th.from_numpy(np.asarray(output.dataobj).astype(dtype='float32'))
          #  output=output[None]
            print('output', output.shape)
           # inputpath=os.path.join(PathInput, filename)
           # input = nib.load(inputpath)
           # input = th.from_numpy(np.asarray(input.dataobj).astype(dtype='float32'))
           # print('inputpath')
          #  output[output==1]=0
          #  viz.image(visualize(output[0, 0, 25:, 10:-5]).unsqueeze(dim=0), opts=dict(caption="in" + str(number)))
          #  viz.image(visualize(output[0, 1, 25:, 10:-5]).unsqueeze(dim=0), opts=dict(caption="out" + str(number)))
          #  diff = (output[0, 0, 25:, 10:-5].cpu() - output[0, 1, 25:, 10:-5].cpu()).square()

            viz.image(visualize(output[0, 0,:-10,10:]).unsqueeze(dim=0), opts=dict(caption="in" + str(number)))
            viz.image(visualize(output[0, 1, :-10,10:]).unsqueeze(dim=0), opts=dict(caption="out" + str(number)))
            diff = (output[0, 0, :-10,10:].cpu() - output[0, 1, :-10,10:].cpu()).square()
          #  viz.image(visualize(output[ 0,0, ...]), opts=dict(caption=str(number) + "LDM"))
            #viz.image(visualize(output[0, 1, ...]), opts=dict(caption=str(number) + "LDM"))
            #save_image(output[0,1,  ...], '../Figures_paper/OCT/349021/' + str(number) + 'LDM.png')
         #   viz.image(visualize(output[0, 1, ...]), opts=dict(caption="pred1"))
         #   viz.image(visualize(output[0, 2, ...]), opts=dict(caption="diff"))

           # save_image(output[0, 0:1, ...], '../Figures_paper/OCT/349021/' + str(number) + 'org.png')
          #  save_image(output[0, 1:2, ...], '../Figures_paper/OCT/349021/' + str(number) + 'ano.png')
           # save_image(output[0, 2:3, ...], '../Figures_paper/OCT/400_07/' + str(number) + 'OCT_diff.png')

            #  th.save(diff, './diffmap_000255/ours')
           # diff = np.array(output[0, 4, ...].cpu())
          #  diff = torch.abs(output[0,0, ...] - output[0,1, ...].cpu()).square()
            viz.heatmap(np.flipud(np.array(diff.cpu())),
                        opts=dict(caption="diffpred" + str(number), colormap='Jet'))



            k+=1

    print('mean dice', all_dice / k)
    print('mean dice nonzero', non_zerodice / nz)

def create_argparser():
    defaults = dict(
       # clip_denoised=True,
        num_samples=1083,
        batch_size=1,
        data_dir="./data/brats/train_healthy",#'/home/juliawolleb/PycharmProjects/Python_Tutorials/BinaryLatentDiffusion/data/chexpert',
        use_ddim=True,
        model_path= './results/latentbrats_1283232healthy110000.pt',#'./results/latentbrats_646464healthy070000.pt',
                                                            #"./results/latentbrats_32healthy290000_epsilon.pt", #'./results/latentbrats_64healthy280000.pt',
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
