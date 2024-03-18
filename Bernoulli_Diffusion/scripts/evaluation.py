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
from torcheval.metrics import BinaryAUPRC
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import precision_recall_curve
import sys
metric = BinaryAUPRC()
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

        volume[i,...] = scipy.ndimage.median_filter(volume[i,...], (kernelsize, kernelsize))
    return volume

def filter_2d_connected_components(volume):
    sz = None
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    cc_volume = label(volume, connectivity=2)
    props = regionprops(cc_volume)
    for prop in props:
        if prop['filled_area'] <= 15:
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

def PRC(real_mask, square_error):
    if type(real_mask) == torch.Tensor:
        return precision_recall_curve(real_mask.detach().cpu().numpy().flatten(), square_error.detach().cpu().numpy().flatten())
    else:
        return precision_recall_curve(real_mask.flatten(), square_error.flatten())



def AUC_score(fpr, tpr):
    return auc(fpr, tpr)

import argparse
import os
from visdom import Visdom
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



Dice_list=[]
PSNR_list=[]
AUC_list=[]
PRC_list=[]
#PathDicomstripped = "/home/juliawolleb/PycharmProjects/Python_Tutorials/BerDiff/AnoDDPM/AnoDDPM/reconstructions_simplex/"
#PathDicomstripped = ("./Reconstruction_300_0.8/")
#PathDicomstripped = "./fake/"
PathDicomstripped = "./Reconstruction_patch_simplex4/"
#PathDicomstripped = "/data/Bratssliced/val_reconstruction_ddpm"
#PathDicomstripped = "./reconstructions_patch_ddpm/"
PathInput = "./data/brats/val_diseased_with_labels"
#PathInput = "./data/brats/val_healthy"

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
    total_auc=0
    total_psnr=0
    total_ssim=0

    nz=0
    k=0
    print('path',PathDicomstripped )
    for dirName, subdirList, fileList in os.walk(PathDicomstripped):  # used to be dicomstripped
        s = dirName.split("/", -1)
        #print('s', s[-1], s[-2])
        #     if 't1n_3d' in subdirList:
        #         path=os.path.join(dirName, 't1n_3d'))
        for filename in fileList:
            s = filename.split(".", 1)
            number = s[0]
            s = filename.split("_", 1)
            number = s[0]

           # if 'nii.gz' not in number:
              #  continue
            #
           # print('number', number)
         #   if number != '002177':
            #    continue

            path = os.path.join(dirName, filename)

            output = nib.load(path)
            try:
                output = th.from_numpy(np.asarray(output.dataobj).astype(dtype='float32'))
            except:
                continue
           # print('output', output.shape)
            output=output[None,...]
         #   print('output', output.shape)
         #   viz.image(visualize(output[0, 4, ...]), opts=dict(caption=str(number) + "output0"))
          #  viz.image(visualize(output[0, 5, ...]), opts=dict(caption=str(number) + "output1"))
         #   viz.image(visualize(output[0, 6, ...]), opts=dict(caption=str(number) + "output2"))
         #   viz.image(visualize(output[0, 5, ...]), opts=dict(caption=str(number) + "output1"))

          #  viz.image(visualize(output[0, 2, ...]), opts=dict(caption=str(number) + "input2"))
         #   viz.image(visualize(output[0, 1, ...]), opts=dict(caption=str(number) + "input1"))
        #    viz.image(visualize(output[0, 2, ...]), opts=dict(caption=str(number) + "input2"))
          #  viz.image(visualize(output[0, 3, ...]), opts=dict(caption=str(number) + "input3"))
            mse = (output[0, 4:8, ...].cpu() - output[0, 0:4, ...].cpu()).square()
            D = np.flipud(np.array(mse.sum(dim=0)))
       #     viz.heatmap(D, opts=dict(caption="sum", colormap='Jet'))


           # output=output[None,...]
            inputpath=os.path.join(PathInput, number)#+'.nii.gz')
           # print('inputpath')
            input = nib.load(inputpath)
            input = th.from_numpy(np.asarray(input.dataobj).astype(dtype='float32'))[None,...]
            label=input[0, 4, ...]
            n = number.split(".", 1)
            number = n[0]
         #   output=output[None,...]
           # print('output', output.shape)
          #  number='005203'
            # #

           # viz.image(visualize(input[0, 2, ...]), opts=dict(caption="input0"))
          #  viz.image(visualize(input[0, 3, ...]), opts=dict(caption="input3"))
            # viz.image(visualize(input[0, 4, ...]), opts=dict(caption="gt"))
            #
         #   save_image(output[0, 4:5, ...], '../Figures_paper/' + str(number) + '/ano0.png')
         #   save_image(output[0, 5:6, ...], '../Figures_paper/' + str(number) + '/ano1.png')
        #    save_image(output[0, 6:7, ...], '../Figures_paper/' + str(number) + '/ano2.png')
         #   save_image(output[0, 7:8, ...], '../Figures_paper/' + str(number) + '/ano3.png')

        #    save_image(output[0, 0:1, ...], '../Figures_paper/' + str(number) + '/auto0.png')
         #   save_image(output[0, 1:2, ...], '../Figures_paper/' + str(number) + '/auto1.png')
         #   save_image(output[0, 2:3, ...], '../Figures_paper/' + str(number) + '/auto2.png')
          #  save_image(output[0, 3:4, ...], '../Figures_paper/' + str(number) + '/auto3.png')
          #  save_image(label, '../Figures_paper/' + str(number) + '/GT.png')


            #  th.save(diff, './diffmap_000255/ours')
            #diff = np.array(output[0, 4, ...].cpu())


            compute_ours=False
            Otsu = False
            if compute_ours:
                diff = (torch.abs(visualize(output[0,  ...]) - visualize(input[0, :4, ...]).cpu())).square()
                D = np.flipud(np.array(diff.sum(dim=0)))
              #  viz.heatmap(D, opts=dict(caption="sum", colormap='Jet'))
                diff = torch.from_numpy(apply_2d_median_filter(np.array(diff).squeeze(), kernelsize=5))
            #OURS
                for i in range(4):
                    p2 = np.percentile(diff[i,...], 1)
                    p98 = np.percentile(diff[i,...], 99)
                    diff[i,...] = torch.clamp(diff[i,...], min=p2, max=p98)

             #   diff = torch.from_numpy(apply_2d_median_filter(np.array(diff).squeeze(), kernelsize=5))
                #sum=diff.sum(dim=0)
              #  save_image(visualize(sum), '../Figures_paper/' + str(number) + '/berdiff_difference.png')

               # sum=scipy.ndimage.filters.median_filter(sum.numpy(), 5)
                diff=visualize(diff.clone().detach())


                if Otsu:
                    sum=diff.sum(dim=0)
                    D2 = np.flipud(np.array(diff.sum(dim=0)))
                 #   viz.heatmap(D2, opts=dict(caption="sum", colormap='Jet'))
                   # plt.imshow(np.flipud(D2), cmap='jet')
                   # viz.matplot(plt)

                    print('sum', sum.shape)
                    T= threshold_otsu(np.array(diff))
                    print('T', T)
                    mse=(diff > T).float()
                  #  sum_mse = (sum>T)
                    mse = torch.tensor(mse)
                    sum_mse = mse.sum(dim=0)

                    sum_mse[sum_mse > 0] = 1
                    pred = filter_2d_connected_components(np.squeeze(np.array(sum_mse)))
                  #  viz.heatmap(sum, opts=dict(caption="median mse 2"))
           #     viz.heatmap(diff[0,...], opts=dict(caption="median mse 2", colormap='Jet'))
             #   viz.heatmap(diff[0, ...], opts=dict(caption="median mse 2", colormap='Jet'))
               # viz.heatmap(diff[2, ...], opts=dict(caption="median mse 2", colormap='Jet'))
            #    viz.heatmap(diff[3, ...], opts=dict(caption="median mse 2", colormap='Jet'))
                else:
                 #   viz.heatmap(diff[0, ...], opts=dict(caption="median mse 2", colormap='Jet'))
                 #   viz.heatmap(diff[2, ...], opts=dict(caption="median mse 2", colormap='Jet'))
                   # viz.heatmap(diff[3, ...], opts=dict(caption="median mse 2", colormap='Jet'))

                    mse = (diff > 0.5).float()
                 #   mse = (sum > 0.5).float()
                    #mse= filter_2d_connected_components(np.squeeze(np.array(mse)))
                    mse =mse.clone().detach()
                    sum_mse = mse.sum(dim=0)

                    sum_mse[sum_mse > 0] = 1
                    pred = filter_2d_connected_components(np.squeeze(np.array(sum_mse)))
                   # viz.image(pred)
            else:

            # THEIRS
                mse = (output[0,4:8,...].cpu() - input[0,:4,  ...].cpu()).square()
                D = np.flipud(np.array(mse.sum(dim=0)))
             #   viz.heatmap(D, opts=dict(caption="sum", colormap='Jet'))
                mse = torch.from_numpy(apply_2d_median_filter(np.array(mse ).squeeze(), kernelsize=5))
                mse = mse.sum(dim=0)
                p2 = np.percentile(mse, 1)
                p98 = np.percentile(mse, 99)
                mse= torch.clamp(mse, min=p2, max=p98)

                #   diff = torch.from_numpy(apply_2d_median_filter(np.array(diff).squeeze(), kernelsize=5))
                # sum=diff.sum(dim=0)
                #  save_image(visualize(sum), '../Figures_paper/' + str(number) + '/berdiff_difference.png')

                # sum=scipy.ndimage.filters.median_filter(sum.numpy(), 5)
                mse = visualize(mse.clone().detach())
                prob=mse.clone().detach()
             #   viz.heatmap(mse, opts=dict(caption="median mse 2", colormap='Jet'))
                mse = (mse > 0.5).float()
                pred = filter_2d_connected_components(np.squeeze(np.array(mse)))




            DSC = dice_score(torch.tensor(pred), label.cpu())
            Dice_list.append(DSC)
            #   DSC = dice_score(th.tensor(T), label.cpu())
            all_dice += DSC
            if DSC > 0:
                non_zerodice += DSC
                nz += 1
            k+=1
            psnr = PSNR(output[0 ,4:8,...].cpu(), input[0,:4,  ...].cpu())

            total_psnr += psnr

            if np.isnan(psnr)==True:
                print('psnr nan', psnr)
                continue
            else:

                PSNR_list.append(psnr)
            try:
                fpr_simplex, tpr_simplex, _ = ROC_AUC(label.cpu().to(torch.uint8), prob)
                auc = AUC_score(fpr_simplex, tpr_simplex)
                AUC_list.append(auc)
            except:
                continue
          #  total_auc += auc
           #
            prc=0
            metric.update(prob.flatten(), label.cpu().to(torch.uint8).flatten())
            prc =metric.compute()
            metric.reset()
            PRC_list.append(prc)

          #  fpr_simplex2, tpr_simplex2, _ = PRC(label.cpu().to(torch.uint8), prob)
           # prc2 = AUC_score(fpr_simplex2, tpr_simplex2)

     #       print('PRC', prc)
#

            if k%100==0:
                print('k', k)
            #prc = binary_auprc(prob, label.cpu().to(torch.uint8))
         #   prc = AUC_score(fpr_simplex, tpr_simplex)








    print('mean dice', all_dice / k)

   #print('mean ssim', total_ssim/k)
    print('mean PSNR', total_psnr/k)
    print('mean AUC', total_auc/k)
    D_std=np.std(np.array(Dice_list))
    D_mean = np.mean(np.array(Dice_list))
    P_std = np.std(np.array(PSNR_list))
    P_mean = np.mean(np.array(PSNR_list))
    A_std=np.std(np.array(AUC_list))
    A_mean = np.mean(np.array(AUC_list))
    PRC_std = np.std(np.array(PRC_list))
    PRC_mean = np.mean(np.array(PRC_list))
    print('Dice', D_mean, D_std)
    print('PSNR', P_mean, P_std)
    print('auroc', A_mean, A_std)
    print('PRC', PRC_mean, PRC_std)
    print('folder', PathDicomstripped)

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
