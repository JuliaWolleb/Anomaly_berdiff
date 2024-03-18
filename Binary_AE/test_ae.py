import torch
import numpy as np
import copy
import time
import os
import pdb
from visdom import Visdom
viz = Visdom(port=8850)
from tqdm import tqdm
from models.binaryae import BinaryAutoEncoder, Generator
from hparams import get_sampler_hparams
from utils.data_utils import get_data_loaders
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts, \
    get_sampler, get_online_samples, get_online_samples_guidance, get_samples_test, get_samples_temp, get_samples_loop
from utils.train_utils import EMA, NativeScalerWithGradNormCount, visualize
from utils.log_utils import log, log_stats, config_log, start_training_log, \
    save_stats, load_stats, save_model, load_model, save_images, \
    MovingAverage
import misc
import torch.distributed as dist
from utils.lr_sched import adjust_lr, lr_scheduler


def main(H, vis):
    misc.init_distributed_mode(H)

    ae_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['encoder', 'quantize', 'generator'],
        remove_component_from_key=False
    )
    quanitzer_and_generator_state_dict = retrieve_autoencoder_components_state_dicts(
        H,
        ['quantize', 'generator'],
        remove_component_from_key=True
    )

    embedding_weight = quanitzer_and_generator_state_dict.pop(
        'embed.weight')
    if H.deepspeed:
        embedding_weight = embedding_weight.half()
    embedding_weight = embedding_weight.cuda()
    generator = Generator(H)
    generator.load_state_dict(quanitzer_and_generator_state_dict, strict=False)
    generator = generator.cuda()
    #del quanitzer_and_generator_state_dict


    bergan = BinaryAutoEncoder(H)
    bergan.load_state_dict(ae_state_dict, strict=True)
    bergan = bergan.cuda()
    del ae_state_dict

    sampler = get_sampler(H, bergan.quantize.embed.weight).cuda()
    print('got sampler')
    device = torch.device("cuda:0")
    print('device', device)
    if H.ema:
        ema_sampler = copy.deepcopy(sampler)

    start_step = 0

    scaler = NativeScalerWithGradNormCount(H.amp, H.init_scale)

    if H.load_step == -1:
        fs = os.listdir(os.path.join(H.load_dir, 'saved_models'))
        fs = [f for f in fs if f.startswith('bld_ema')]
        fs = [int(f.split('.')[0].split('_')[-1]) for f in fs]
        load_step = np.max(fs)
        print('Overriding loadstep with %d' % load_step)
        H.load_step = load_step


    if H.load_step > 0:
        start_step = H.load_step + 1


        print('device', device)
        print('loading sampler')

        allow_mismatch = H.allow_mismatch
        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir, device=device,
                             allow_mismatch=allow_mismatch).cuda()
        if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
            try:
                ema_sampler = load_model(
                    ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, device=device,
                    allow_mismatch=allow_mismatch)
            except Exception:
                ema_sampler = copy.deepcopy(sampler_without_ddp)

        if not allow_mismatch:
            if H.load_optim:
                optim = load_model(
                    optim, f'{H.sampler}_optim', H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch)
                for param_group in optim.param_groups:
                    param_group['lr'] = H.lr


        if not H.reset_step:
            if not H.reset_scaler:
                try:
                    scaler.load_state_dict(
                        torch.load(os.path.join(H.load_dir, 'saved_models', f'absorbingbnl_scaler_{H.load_step}.th')))
                except Exception:
                    print('Failing to load scaler.')
        else:
            H.load_step = 0


        if H.reset_step:
            start_step = 0
    print('about to get loader')
    train_loader, val_loader = get_data_loaders(
        H.dataset,
        H.img_size,
        H.batch_size,
        get_val_dataloader=True,
        custom_dataset_path=H.path_to_data,
        num_workers=4,
        distributed=False,
        random=True,
        args=H,
    )

    log(f"Sampler params total: {(sum(p.numel() for p in sampler.parameters()) / 1e6)}M")

    # for step in range(start_step, H.train_steps):
    H.train_steps = H.train_steps * H.update_freq
    H.warmup_iters = H.warmup_iters * H.update_freq
    H.steps_per_log = H.steps_per_log * H.update_freq

    step = start_step - 1


    epoch = -1
    print('valloader', val_loader, len(val_loader))

    with torch.no_grad():
     while True:
        epoch += 1

        for i, data in enumerate(val_loader):
            step += 1

            img = data[0].cuda()
            img=img[:,:1,...]

            viz.image(visualize(img[0, 0, ...]), opts=dict(caption="img input 0"))
            viz.image(visualize(img[1, 0, ...]), opts=dict(caption="img input 1"))
            label = data[1].cuda()

            with torch.no_grad():
                code = bergan(img, code_only=True).detach()
                xx, _,_ = bergan(img, code_only=False)


            viz.image(visualize(img[0, 0, ...]), opts=dict(caption="input to AE 1"))
            viz.image(visualize(xx[0, 0, ...]), opts=dict(caption="generated output 1"))
            viz.image(visualize(code[0, 0, ...]), opts=dict(caption="generated code 1"))

        return xx





if __name__ == '__main__':
    H = get_sampler_hparams()
    H.n_channels=1

    print('saving steps', H.steps_per_save_output, H.steps_per_checkpoint)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    print('sampler', H.sampler)
    start_training_log(H)
    main(H, None)
