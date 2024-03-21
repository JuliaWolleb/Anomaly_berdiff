import io
import argparse
import blobfile as bf

import torch as th

from guided_diffusion import binomial_diffusion as bd
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel, EncoderUNetModel



def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    print('got defaults here')
    return dict(
        image_size=32,
        img_channels=128,
        num_channels=128,
        num_res_blocks=2,
        num_heads=1,
        num_heads_upsample=-1,
        attention_resolutions="16",
        dropout=0.0,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing=[1000],
        ltype="bce",
        mean_type="epsilon",
        rescale_timesteps=True,
        use_checkpoint=False,
        use_scale_shift_norm=False
    )



def create_model_and_diffusion(
    image_size,
    img_channels,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    ltype,
    mean_type,
    rescale_timesteps,
    use_checkpoint,
    use_scale_shift_norm,
):
   # print('imgchannels', img_channels)
    model = create_model(
        image_size,
        img_channels,
        num_channels,
        num_res_blocks,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    
    diffusion = create_binomial_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        ltype=ltype,
        mean_type=mean_type,
        rescale_timesteps=rescale_timesteps,
        timestep_respacing=timestep_respacing,
    )
    print('diffusion is binomial diffusion')

    return model, diffusion


def create_model(
    image_size,
    img_channels,
    num_channels,
    num_res_blocks,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    print('img_channels', img_channels)
    if image_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    out_channels=img_channels#=1
    
    
    model = UNetModel(
        in_channels=img_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

    return model

def create_binomial_diffusion(
    *,
    steps=1000,
    noise_schedule="linear",
    ltype="mix",
    mean_type="xstart",
    rescale_timesteps=False,
    timestep_respacing="",
):
    betas = bd.get_named_beta_schedule(noise_schedule, steps)
    if ltype == "rescale_kl":
        loss_type =bd.LossType.RESCALED_KL
    elif ltype == "kl":
        loss_type = bd.LossType.KL
    elif ltype == "bce":
        loss_type = bd.LossType.BCE
    elif ltype == "mix":
        loss_type = bd.LossType.MIX
    else:
        raise NotImplementedError(f"unknown LossType: {ltype}")
    if not timestep_respacing:
        timestep_respacing = [steps]
    if mean_type == "xstart":
        model_mean = bd.ModelMeanType.START_X
    elif mean_type == "epsilon":
        model_mean = bd.ModelMeanType.EPSILON
    elif mean_type == "previous":
        model_mean = bd.ModelMeanType.PREVIOUS_X
    else:
        raise NotImplementedError(f"unknown ModelMeanType: {mean_type}")

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
    

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    mpigetrank=0
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    return th.load(io.BytesIO(data), **kwargs)

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")



def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=128,
        img_channels=32,
        num_channels=128,
        num_res_blocks=2,
        attention_resolutions="16"

    )


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(model_and_diffusion_defaults())
    return res


def create_classifier_and_diffusion(
    image_size,
    img_channels,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    ltype,
    mean_type,
    rescale_timesteps,
    use_checkpoint,
    use_scale_shift_norm,
):
    print('timestepresp2', timestep_respacing)


    classifier = create_classifier(
    image_size,
    img_channels,
    num_channels,
    num_res_blocks,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    )

    diffusion = create_binomial_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        ltype=ltype,
        mean_type=mean_type,
        rescale_timesteps=rescale_timesteps,
        timestep_respacing=timestep_respacing,
    )
    print('diffusion is binomial diffusion')
    return classifier, diffusion


