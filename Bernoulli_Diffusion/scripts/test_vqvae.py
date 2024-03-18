import os
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.nn import L1Loss
from tqdm import tqdm
from visdom import Visdom
viz = Visdom(port=8850)
from generative.networks.nets import VQVAE
import sys
sys.path.append(".")
from guided_diffusion.image_datasets import load_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

# # OLD IMOLEMENTATION
# model = VQVAE(
#     spatial_dims=2,
#     in_channels=1,
#     out_channels=1,
#     num_channels=(256, 256),
#     num_res_channels=256,
#     num_res_layers=2,
#     downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
#     upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
#     num_embeddings=256,
#     embedding_dim=32)


# NEW IMOLEMENTATION
model = VQVAE(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(256, 256, 256),
    num_res_channels=256,
    num_res_layers=2,
    downsample_parameters=((2, 4, 1 , 1), (2, 4, 1, 1),(2, 4, 1, 1)),
    upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0),(2, 4, 1, 1, 0)),
    num_embeddings=256,
    embedding_dim=128)


model.to(device)
#model_path='./VAE/models/0.0059407130333439095-Epo95.pt'
#model_path='./VAE/models/0.014410071386630512-OCT-Epo80.pt'
model_path='./VAE/models/0.018745080581393795-OCT32-Epo100.pt'
model.load_state_dict(torch.load(model_path))

print("loaded model from file"+str(model_path))


epoch_recon_loss_list = []
epoch_quant_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

batch_size=1


#data_dir = "./data/brats/val_diseased_with_labels"  #"./data/brats/val_diseased_with_labels"  "./data/OCT/DRUSEN"
data_dir =  "./data/OCT/DRUSEN"

datal = load_data(
    data_dir=data_dir,
    batch_size=1,
    image_size=256,
)

k=0
num_samples=10
val_loader = iter(datal)
while k < num_samples:
        data, out = next(val_loader)
        number=out["name"]
        print('number', number)
        batch=data[:,:1,...]
    #    label=data[:,-1:,...]

        images =batch.to(device)#. batch[0].squeeze(dim=0).permute(2, 0, 1).unsqueeze(dim=1).to(device)
        print('images', images.shape)

        # model outputs reconstruction and the quantization error
        reconstruction, quantization_loss = model(images=images)
        print('reconstruction', reconstruction.shape, quantization_loss.shape)
        viz.image(visualize(images[0, 0, :, :]).unsqueeze(dim=0))
        viz.image(visualize(reconstruction[0, 0, :, :]).unsqueeze(dim=0))


