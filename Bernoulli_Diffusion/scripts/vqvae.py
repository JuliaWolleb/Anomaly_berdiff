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
viz = Visdom(port=8852)
from generative.networks.nets import VQVAE
import sys
sys.path.append(".")
from guided_diffusion.image_datasets import load_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


 #OLD IMOLEMENTATION

# model   = VQVAE(
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
#
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

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
l1_loss = L1Loss()


n_epochs = 101
val_interval = 10
epoch_recon_loss_list = []
epoch_quant_loss_list = []
val_recon_epoch_loss_list = []
intermediary_images = []
n_example_images = 4

batch_size=24


#data_dir = "./data/brats/train_healthy"
data_dir = "./data/OCT/NORMAL"
datal = load_data(
    data_dir=data_dir,
    batch_size=24,
    image_size=256,
)

epoch=0
for epoch in range(n_epochs):
    print('epoch', epoch)
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(enumerate(datal), total=760, ncols=110)  #enumerate(datal)
  #  progress_bar = enumerate(tqdm(datal)) # enumerate(datal)pro
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:

        images =batch[0].to(device)#. batch[0].squeeze(dim=0).permute(2, 0, 1).unsqueeze(dim=1).to(device)

        optimizer.zero_grad(set_to_none=True)

        # model outputs reconstruction and the quantization error
        reconstruction, quantization_loss = model(images=images)
       # z = model.encode(images)
       # print('latentspace', z.shape)

        recons_loss = l1_loss(reconstruction.float(), images.float())

        loss = recons_loss + quantization_loss

        loss.backward()
        optimizer.step()

        epoch_loss += recons_loss.item()

        #progress_bar.set_postfix(
          #  {"recons_loss": epoch_loss / (step + 1), "quantization_loss": quantization_loss.item() / (step + 1)}
      #  )

        if step % 100 == 0:
            viz.image(images[0, 0, :, :].unsqueeze(dim=0))
            viz.image(reconstruction[0, 0, :, :].unsqueeze(dim=0))

    if epoch %20== 0:
            print("Save model ...")
            torch.save(model.state_dict(), './VAE/models/' + str(epoch_loss/(step+1)) + '-OCT32-Epo' + str(epoch) + '.pt')

    epoch_recon_loss_list.append(epoch_loss / (step + 1))
    epoch_quant_loss_list.append(quantization_loss.item() / (step + 1))

