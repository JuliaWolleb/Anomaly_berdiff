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
viz = Visdom(port=8852)
import matplotlib.pyplot as plt
import numpy as np
import torch
import nibabel
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader, Dataset
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from generative.networks.nets import VQVAE
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
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

#data_dir = "./data/brats/val_diseased_with_labels"
data_dir = "./data/OCT/DRUSEN"
datal = load_data(
    data_dir=data_dir,
    batch_size=1,
    image_size=256,
)


device = torch.device("cuda")


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
#
# ### Define diffusion model and scheduler
#
# In this section, we will define the diffusion model that will learn data distribution of the latent representation of the autoencoder. Together with the diffusion model, we define a beta scheduler responsible for defining the amount of noise tahat is added across the diffusion's model Markov chain.

# +
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


#model_path='./VAE/models/Unet-Epo200.pt'
model_path='./VAE/models/OCT-Unet32-Epo200.pt'

#model_path='./VAE/models/OCT-Unet-Epo400.pt'
unet.load_state_dict(torch.load(model_path))
print("loaded unet from file"+str(model_path))

scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)
# -

scale_factor = 1 #/ torch.std(z)
# -

# We define the inferer using the scale factor:

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# ### Train diffusion model
#
# It takes about ~80 min to train the model.


unet = unet.to(device)


epoch_losses = []
val_losses = []
scaler = GradScaler()
k=0
num_samples=1
val_loader = iter(datal)

unet.eval()
scheduler.set_timesteps(num_inference_steps=1000)
noise = torch.randn((1, 3, 16, 16))
noise = noise.to(device)

while k < num_samples:
        data, out = next(val_loader)
        number=out["name"]
        print('number', number)
        if number[0]!= 'DRUSEN-349021-1':
            continue
        batch=data[:,:4,...]
        label=data[:,-1:,...]
       # batch = data
        images =batch.to(device)#. batch[0].squeeze(dim=0).permute(2, 0, 1).unsqueeze(dim=1).to(device)
        viz.image(visualize(images[0, 0, :, :]).unsqueeze(dim=0))
        print('images', images.shape)

        start.record()

        z = autoencoderkl.encode(images)

        #decode = autoencoderkl.decode_stage_2_outputs

       # output1 = decode(z)
       # viz.image(visualize(output1[0, 0, :, :]).unsqueeze(dim=0))

       # print('z', z.shape, z.max(), z.min())
       # print('inputoutput', batch.min(), batch.max(),output1.min(), output1.max() )
      #  z=torch.clamp(z,0,1)#.detach().cpu()

        timesteps=torch.tensor(300).long()
        #TODO: add noise
        noise = torch.randn_like(z).to(device)
        noisy_z = scheduler.add_noise(z, noise, timesteps)  # add t steps of noise to the input image
        indices = list(range(timesteps))[::-1]

        #scheduler.set_timesteps(num_inference_steps=250)
        # with autocast(enabled=True):
        #        decoded = inferer.sample(
        #            input_noise=noisy_z, diffusion_model=unet, scheduler=scheduler, autoencoder_model=autoencoderkl)
        # print('decoded', decoded.shape)
        # viz.image(decoded[0, 0, :, :].unsqueeze(dim=0))

        current_img = noisy_z
        progress_bar = tqdm(range(timesteps))  # go back and forth L timesteps
        for i in progress_bar:  # go through the noising process
            with autocast(enabled=False):
                t = timesteps - i
                with torch.no_grad():
                    model_output = unet(current_img, timesteps=torch.Tensor((t,)).to(current_img.device))
            current_img, _ = scheduler.step(model_output, t, current_img)
          #  if i%30==0:
           #     output = decode(current_img)
            #    output = torch.clamp(output, 0, 1)  # .detach().cpu()
           #     viz.image(output[0, 0, :, :].unsqueeze(dim=0))
            torch.cuda.empty_cache()

        decode = autoencoderkl.decode_stage_2_outputs

        output = decode(current_img)

        end.record()
        torch.cuda.synchronize()
        print('elapsed time', start.elapsed_time(end))
        continue
        output = torch.clamp(output, 0, 1)  # .detach().cpu()
        print('output', output.shape, output.max(), output.min())
        viz.image(visualize(output[0, 0, 25:, 10:-5]).unsqueeze(dim=0))
        viz.image(visualize(images[0, 0, 25:, 10:-5]).unsqueeze(dim=0))
        diff=(output[0, 0, 25:, 10:-5].cpu()-images[0, 0, 25:, 10:-5].cpu()).square()
        viz.heatmap(diff.cpu(), opts=dict(caption="median mse 0", colormap='Jet'))

        final_imag = torch.cat((images, output), dim=1).detach().cpu()
        print('final image', final_imag.shape)
        final_img = nibabel.Nifti1Image(np.array(final_imag), affine=np.eye(4))
     #   name = os.path.join("./Reconstruction_LDM_OCT/", number[0] + '.nii.gz')
     #   nibabel.save(final_img, name)



progress_bar.close()



# ### Clean-up data directory

if directory is None:
    shutil.rmtree(root_dir)