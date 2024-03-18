import torch
import sys
from visdom import Visdom
sys.path.append('./BinaryLatentDiffusion')
sys.path.append('./Bernoulli_Diffusion')

sys.path.append('.')
viz = Visdom(port=8851)
from Binary_AE.models.binaryae import BinaryAutoEncoder, Generator
from Binary_AE.hparams import get_sampler_hparams
from Bernoulli_Diffusion.guided_diffusion.bratsloader import BRATSDataset

from Bernoulli_Diffusion.guided_diffusion.image_datasets import load_data

from Binary_AE.utils.data_utils import get_data_loaders
from Binary_AE.utils.sampler_utils import retrieve_autoencoder_components_state_dicts, \
    get_sampler, get_online_samples, get_online_samples_guidance, get_samples_test, get_samples_temp, get_samples_loop
from Binary_AE.utils.train_utils import EMA, NativeScalerWithGradNormCount, visualize
from Binary_AE.utils.log_utils import log, log_stats, config_log, start_training_log, \
    save_stats, load_stats, save_model, load_model, save_images, \
    MovingAverage


def autoencoder(image):
    print('we are in the right function')
    H = get_sampler_hparams()

    print('got hparams', H)
    print('image size', H.img_size)
    print('H load dir', H.ae_load_dir)
    print('H latent shape', H.latent_shape)
    print('H n channels', H.n_channels)

    H.ae_load_step = 00000
    H.sampler = 'bld'
    H.data_dir = './data/brats/training'#"./data/OCT/NORMAL"#./data/brats/train_healthy"
    print('codebooksize', H.codebook_size)

    H.latent_shape='1,128,128'



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
    print('datadir', H.data_dir)

    datal = load_data(
        data_dir=H.data_dir,
        batch_size=1,
        image_size=H.image_size,
    )
    print('dataset is brats')
    val_loader=iter(datal)

    batch, out = next(val_loader)
    print('got batch', batch.shape)


    epoch = -1

    with torch.no_grad():
     while True:
        epoch += 1

        for i, (data, out) in enumerate(val_loader):

            print('data', data.shape)
            img = data.cuda()

            print('img', img.shape)

            with torch.no_grad():
                code = bergan(img, code_only=True).detach()
                print('code', code.shape)
                xx, _,_ = bergan(img, code_only=False)

                print('xx', xx.shape)

                b, c, h, w = code.shape
                x = code.view(b, c, -1).permute(0, 2, 1).contiguous()
                print('code', code.shape, x.shape, code.max(), code.min())

                reconstruction, _, _ = bergan(img, code_only=False, code=code)
                print('reconsturction', reconstruction.shape)
                reconstruction = torch.clamp(reconstruction, 0, 1)


            print('code', code.shape, x.shape,  code.max(), code.min())


            viz.image(visualize(img[0, 0, ...]), opts=dict(caption="input to AE 1"))
            viz.image(visualize(reconstruction[0, 0, ...]), opts=dict(caption="reconstruction 1"))

            viz.image(visualize(code[0, 0, ...]), opts=dict(caption="generated code 1"))

        return xx, code






if __name__ == '__main__':
    autoencoder(0)

