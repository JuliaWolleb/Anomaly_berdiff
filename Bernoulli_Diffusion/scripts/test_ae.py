import torch
import sys
from visdom import Visdom
sys.path.append('./BinaryLatentDiffusion')
sys.path.append('./Bernoulli_Diffusion')

sys.path.append('.')
viz = Visdom(port=8850)
from Binary_AE.models.binaryae import BinaryAutoEncoder, Generator
from Binary_AE.hparams import get_sampler_hparams

from Bernoulli_Diffusion.guided_diffusion.image_datasets import load_data

from Binary_AE.utils.data_utils import get_data_loaders
from Binary_AE.utils.sampler_utils import retrieve_autoencoder_components_state_dicts, \
    get_sampler, get_online_samples, get_online_samples_guidance, get_samples_test, get_samples_temp, get_samples_loop
from Binary_AE.utils.train_utils import  visualize


def autoencoder(image):
    H = get_sampler_hparams()


    ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['encoder', 'quantize', 'generator'],
        remove_component_from_key=False
    )

    bergan = BinaryAutoEncoder(H)
    bergan.load_state_dict(ae_state_dict, strict=True)
    bergan = bergan.cuda()
    del ae_state_dict


    datal = load_data(
        data_dir=H.data_dir,
        batch_size=1,
        image_size=H.image_size,
    )
    val_loader=iter(datal)

    batch, out = next(val_loader)

    epoch = -1

    with torch.no_grad():
     while True:
        epoch += 1

        for i, (data, out) in enumerate(val_loader):


            img = data.cuda()

            with torch.no_grad():
                code = bergan(img, code_only=True).detach()  #binary latent representation
                xx, _,_ = bergan(img, code_only=False)

                reconstruction, _, _ = bergan(img, code_only=False, code=code)  #reconstructed image
                reconstruction = torch.clamp(reconstruction, 0, 1)

            #plot the results using visdom
            viz.image(visualize(img[0, 0, ...]), opts=dict(caption="input to AE "))
            viz.image(visualize(reconstruction[0, 0, ...]), opts=dict(caption="reconstruction "))
            viz.image(visualize(code[0, 0, ...]), opts=dict(caption="generated code"))
        return xx, code



if __name__ == '__main__':
    autoencoder(0)

