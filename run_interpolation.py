import warnings
import argparse
import os
import numpy as np
import torch
import cv2

import stylegan2
from stylegan2 import utils
from run_generator import _add_shared_arguments


#----------------------------------------------------------------------------

_description = """StyleGAN2 generator.
Run 'python %(prog)s <subcommand> --help' for subcommand help."""


def get_arg_parser():
    parser = argparse.ArgumentParser(
        description=_description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    range_desc = 'NOTE: This is a single argument, where list ' + \
        'elements are separated by "," and ranges are defined as "a-b". Only integers are allowed.'

    parser.add_argument(
        '--batch_size',
        help='Batch size for generator. Default: %(default)s',
        type=int,
        default=1,
        metavar='VALUE'
    )

    parser.add_argument(
        '--seeds',
        help='List of random seeds for generating images. ' + range_desc,
        type=utils.range_type,
        required=True,
        metavar='RANGE'
    )


    parser.add_argument(
        '--interpolation_step',
        help='Number of interpolated images in-between 2 samples',
        type=int,
        default=50,
        metavar='VALUE'
    )

    parser.add_argument(
        '--animate_interpolation',
        help='Generate a video of interpolated images or not (False - do not generate)',
        type=bool,
        default=False,
        metavar='VALUE'
    )

    parser.add_argument(
        '--animation_fps',
        help='FPS of animated interpolation',
        type=int,
        default=24,
        metavar='VALUE'
    )

    parser.add_argument(
        '--animation_filename',
        help='Filename of animated interpolation',
        type=str,
        default='interpolated.mp4',
        metavar='VALUE'
    )

    _add_shared_arguments(parser)

    return parser


def interpolate(G, args):
    latent_size, label_size = G.latent_size, G.label_size
    device = torch.device(args.gpu[0] if args.gpu else 'cpu')
    if device.index is not None:
        torch.cuda.set_device(device.index)
    G.to(device)
    if args.truncation_psi != 1:
        G.set_truncation(truncation_psi=args.truncation_psi)
    if len(args.gpu) > 1:
        warnings.warn(
            'Noise can not be randomized based on the seed ' + \
            'when using more than 1 GPU device. Noise will ' + \
            'now be randomized from default random state.'
        )
        G.random_noise()
        G = torch.nn.DataParallel(G, device_ids=args.gpu)
    else:
        noise_reference = G.static_noise()

    noise_tensors = None
    if noise_tensors is not None:
        G.static_noise(noise_tensors=noise_tensors)
    def gen_latent(seed):
        return torch.from_numpy(np.random.RandomState(seed).randn(latent_size))

    def interpolate_generator(seed, step):
        if len(args.gpu) <= 1:
            noise_tensors = [[] for _ in noise_reference]
            for i, ref in enumerate(noise_reference):
                noise_tensors[i].append(torch.from_numpy(np.random.RandomState(seed).randn(*ref.size()[1:])))
            noise_tensors = [
                torch.stack(noise, dim=0).to(device=device, dtype=torch.float32)
                for noise in noise_tensors
            ]
        else:
            noise_tensors = None

        latent1 = gen_latent(seed)
        latent2 = gen_latent(seed + 1)
        d_latents = (latent2 - latent1) / float((step - 1))
        for i in range(step):
            yield latent1 + i * d_latents, noise_tensors

    progress = utils.ProgressWriter(len(args.seeds) * args.interpolation_step)
    progress.write('Generating images...', step=False)

    if args.interpolation_step:
        fourcc_ = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(filename=os.path.join(args.output, args.animation_filename),
                                       fourcc=fourcc_,
                                       fps=args.animation_fps,
                                       apiPreference=cv2.CAP_ANY,
                                       frameSize=(512,512))

    for seed in args.seeds:
        for i, (latent, noise_tensors) in enumerate(interpolate_generator(seed, args.interpolation_step)):
            latents = torch.stack([latent], dim=0).to(device=device, dtype=torch.float32)

            if noise_tensors is not None:
                G.static_noise(noise_tensors=noise_tensors)

            with torch.no_grad():
                generated = G(latents, labels=None)
            images = utils.tensor_to_PIL(
                generated, pixel_min=args.pixel_min, pixel_max=args.pixel_max)
            for img in images: # args.seeds[i: i + args.batch_size]
                img.save(os.path.join(args.output, f'{seed}_{i}.png'))
                if args.interpolation_step:
                    img = np.array(img)
                    # Convert RGB to BGR
                    img = img[:, :, ::-1].copy()
                    video_writer.write(img)

                progress.step()

    progress.write('Done!', step=False)
    progress.close()


# ----------------------------------------------------------------------------


def main():
    args = get_arg_parser().parse_args()
    assert os.path.isdir(args.output) or not os.path.splitext(args.output)[-1], \
        '--output argument should specify a directory, not a file.'
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    G = stylegan2.models.load(args.network)
    G.eval()

    assert isinstance(G, stylegan2.models.Generator), 'Model type has to be ' + \
        'stylegan2.models.Generator. Found {}.'.format(type(G))

    interpolate(G, args)




if __name__ == '__main__':
    main()