# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -

# # 2D Latent Diffusion Model
#
# In this tutorial, we will walk through the process of using the MONAI Generative Models package to generate synthetic data using Latent Diffusion Models (LDM)  [1, 2]. Specifically, we will focus on training an LDM to create synthetic X-ray images of hands from the MEDNIST dataset.
#
# [1] - Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
#
# [2] - Pinaya et al. "Brain imaging generation with latent diffusion models" https://arxiv.org/abs/2209.07162
#

# ### Set up environment

# !python -c "import monai" || pip install -q "monai-weekly[tqdm]"
# !python -c "import matplotlib" || pip install -q matplotlib
# %matplotlib inline

# ### Setup imports

# +
import os
import shutil
import tempfile
from visdom import Visdom
viz = Visdom(port=8850)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from generative.networks.nets import VQVAE

from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")
print_config()
import sys
sys.path.append(".")
from guided_diffusion.image_datasets import load_data

# -

# ### Set deterministic training for reproducibility

set_determinism(42)

# ### Setup a data directory and download dataset
# Specify a MONAI_DATA_DIRECTORY variable, where the data will be downloaded. If not specified a temporary directory will be used.

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img



# ## Prepare training set data loader

image_size = 256

#data_dir = "./data/brats/train_healthy"
data_dir = "./data/OCT/NORMAL"
datal = load_data(
    data_dir=data_dir,
    batch_size=24,
    image_size=256,
)


# ## Autoencoder KL

device = torch.device("cuda")

# OLD IMOLEMENTATION
# autoencoderkl  = VQVAE(
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
autoencoderkl = VQVAE(
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

autoencoderkl .to(device)
#ae_path='./VAE/models/0.0059407130333439095-Epo95.pt'
#ae_path='./VAE/models/0.014410071386630512-OCT-Epo80.pt'
ae_path='./VAE/models/0.018745080581393795-OCT32-Epo100.pt'

autoencoderkl.load_state_dict(torch.load(ae_path))

print("loaded ae from file"+str(ae_path))


# ## Diffusion Model

# unet = DiffusionModelUNet(
#     spatial_dims=2,
#     in_channels=32,
#     out_channels=32,
#     num_res_blocks=2,
#     num_channels=(128, 256, 512),
#     attention_levels=(False, True, True),
#     num_head_channels=(0, 256, 512),
# )

# NEW IMOLEMENTATION
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=128,
    out_channels=128,
    num_res_blocks=2,
    num_channels=(128, 256, 512),
    attention_levels=(False, True, True),
    num_head_channels=(0, 256, 512),
)
#model_path='./VAE/models/Unet-Epo180.pt'
model_path='./VAE/models/OCT-Unet32-Epo100.pt'

unet.load_state_dict(torch.load(model_path))
print('loaded unet', model_path)

scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)
# -

# ### Scaling factor
#
# As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
#
# _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
#
#
# with torch.no_grad():
#     with autocast(enabled=True):
#         z = autoencoderkl.encode_stage_2_inputs(check_data["image"].to(device))
#         print('z',z.shape )

#print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 #/ torch.std(z)
# -

# We define the inferer using the scale factor:

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# ### Train diffusion model
#
# It takes about ~80 min to train the model.

# +
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

unet = unet.to(device)
n_epochs = 202
val_interval = 40
epoch_losses = []
val_losses = []
scaler = GradScaler()

for epoch in range(n_epochs):
    unet.train()
    print('epoch', epoch)

    autoencoderkl.eval()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(datal), total=759, ncols=110)  #759
    # enumerate(datal)
    #  progress_bar = enumerate(tqdm(datal)) # enumerate(datal)pro
    progress_bar.set_description(f"Epoch {epoch}")


    for step, batch in progress_bar:

        optimizer.zero_grad(set_to_none=True)

        images =batch[0].to(device)#. batch[0].squeeze(dim=0).permute(2, 0, 1).unsqueeze(dim=1).to(device)

        with autocast(enabled=True):

            z= autoencoderkl.encode(images)
          #  decode = autoencoderkl.decode_stage_2_outputs

          #  output1 = decode(z)
           # viz.image(visualize(output1[0, 0, :, :]).unsqueeze(dim=0))
           # viz.image(visualize(images[0, 0, :, :]).unsqueeze(dim=0))

            noise = torch.randn_like(z).to(device)
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            noise_pred = inferer(
                inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps, autoencoder_model=autoencoderkl
            )
         #   print('noise.shape', noise_pred.shape, noise.shape)
        #    viz.image(visualize(noise_pred[0, 0, :, :]).unsqueeze(dim=0))
         #   viz.image(visualize(noise[0, 0, :, :]).unsqueeze(dim=0))
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

    epoch_losses.append(epoch_loss / (step + 1))
    if epoch %25== 0:
            print("Save model ...")
            torch.save(unet.state_dict(), './VAE/models/'  + 'OCT-Unet32-Epo' + str(epoch) + '.pt')


progress_bar.close()


# ### Clean-up data directory

if directory is None:
    shutil.rmtree(root_dir)