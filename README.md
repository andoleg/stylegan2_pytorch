## Tattoo Generation Porject
### About
This is an art project aimed to generate tattoo sketches. Project is based on PyTorch Implementation of StyleGAN2.

### Participants
Developed by Innopolis University students:
* [Andriyanchenko Oleg](https://github.com/andoleg)
* [Daniel Ginzburg](https://github.com/thedownhill)

### Dataset
Dataset was scrapped from [freepik.com](freepik.com) using custom written scrapper (`./scrapper.py`). It was partially cleaned during scrapping using image tag words. Final revision and cleaning was done by hand with following criteria:
* Tattoo unrelated image
* Multiple sketches on a single image
* Text on image

Overall around ~3.6k images were gathered initially and ~2.6 images remained after cleaning.
Link to dataset: [Google Drive](https://drive.google.com/file/d/1Dxv_ZZyilD7zeSHe0zFHdoIoWdogSRd-/view?usp=sharing)

### Training
Training was done on 16Gb Tesla V100 GPU. We trained 2 main models:
* Generator that outputs 512x512 images with initial latent vector of size 512
* Generator that outputs 128x128 images with initial latent vector of size 512

Trained models can be accessed here (Link will appear in near future).

### Results
Generated 512x512 images:\
<img src="https://github.com/andoleg/stylegan2_pytorch/blob/master/results_generated/6_0.png" width="450"/> <img src="https://github.com/andoleg/stylegan2_pytorch/blob/master/results_generated/18_0.png" width="450"/> 

### Interpolation Between Generated Images
For convenience, we created a script (`run_interpolation.py`) that allows you to create interpolation frames between different images. Script allows to create a video of smooth interpolation over created images.
Arguments are the same as in `run_generator.py` plus:
* `animate_interpolation` - write a video if `True`
* `animation_filename` - filename to output video
* `animation_fps` - fps of the video
* `animation_frame_size` - frame size of images written to video

Example of running the script:
`python --network=../path/to/Gs.pth --seeds=1-2 --truncation_psi=1.0 --output ./interpolated --animate_interpolation True`

Our resulted interpolation over generated tattoo sketches: [Google Drive](https://drive.google.com/file/d/1Sdv8WRpIAJSeD8ZHJELqfKOCFunb8Bl3/view?usp=sharing)

# StyleGAN2 readme.md
## StyleGAN2 &mdash; Pytorch Implementation
### About
This is an unofficial port of the StyleGAN2 architecture and training procedure from [the official Tensorflow implementation](https://github.com/NVlabs/stylegan2) to Pytorch. Pretrained Tensorflow models can be converted into Pytorch models.

This model is built to be runnable for 1d, 2d and 3d data.

### Installation
1. Clone this repository: `git clone https://github.com/Tetratrio/stylegan2_pytorch.git`
2. (Recommended) Create a virtualenv: `virtualenv .venv && source .venv/bin/activate`
3. Install requirements: `pip install -r requirements.txt`

### Using pre-trained networks
Networks can be converted from their tensorflow version by running `run_convert_from_tf.py`. The networks have to be of type StyleGAN2, the baseline StyleGAN is not supported (config a-d). The network weights can be automatically downloaded if you specify `--download=NAME` where `NAME` is one of the following:

```
car-config-e
car-config-f
cat-config-f
church-config-f
ffhq-config-e
ffhq-config-f
horse-config-f
car-config-e-Gorig-Dorig
car-config-e-Gorig-Dresnet
car-config-e-Gorig-Dskip
car-config-e-Gresnet-Dorig
car-config-e-Gresnet-Dresnet
car-config-e-Gresnet-Dskip
car-config-e-Gskip-Dorig
car-config-e-Gskip-Dresnet
car-config-e-Gskip-Dskip
ffhq-config-e-Gorig-Dorig
ffhq-config-e-Gorig-Dresnet
ffhq-config-e-Gorig-Dskip
ffhq-config-e-Gresnet-Dorig
ffhq-config-e-Gresnet-Dresnet
ffhq-config-e-Gresnet-Dskip
ffhq-config-e-Gskip-Dorig
ffhq-config-e-Gskip-Dresnet
ffhq-config-e-Gskip-Dskip
```
Alternatively, specify a file directly with `--input=FILE`.

### Generating images
```.bash
# Train a network or convert a pretrained one.
# Example of converting pretrained ffhq model:
python run_convert_from_tf.py --download ffhq-config-f --output G.pth D.pth Gs.pth

# Generate ffhq uncurated images (matches paper Figure 12)
python run_generator.py generate_images --network=Gs.pth --seeds=6600-6625 --truncation_psi=0.5

# Generate ffhq curated images (matches paper Figure 11)
python run_generator.py generate_images --network=Gs.pth --seeds=66,230,389,1518 --truncation_psi=1.0

# Example of converting pretrained car model:
python run_convert_from_tf.py --download car-config-f --output G_car.pth D_car.pth Gs_car.pth

# Generate uncurated car images (matches paper Figure 12)
python run_generator.py generate_images --network=Gs_car.pth --seeds=6000-6025 --truncation_psi=0.5

# Generate style mixing example (matches style mixing video clip)
python run_generator.py style_mixing_example --network=Gs.pth --row_seeds=85,100,75,458,1500 --col_seeds=55,821,1789,293 --truncation_psi=1.0
```
The results are placed in `<RUNNING_DIR>/results/*.png`. You can change the location with `--output`. For example, `--output=~/my-stylegan2-results`.

### Projecting images to latent space
```.bash
# Train a network or convert a pretrained one.
# Example of converting pretrained ffhq model:
python run_convert_from_tf.py --download ffhq-config-f --output G.pth D.pth Gs.pth

# Project generated images
python run_projector.py project_generated_images --network=Gs.pth --seeds=0,1,5

# Project real images
python run_projector.py project_real_images --network=Gs.pth --data-dir=path/to/image_folder
```

### Training and Evaluating
When specifying a location of your images it should be a root directory where images are located in the root directory or any subdirectories. If conditioning networks with labels, classes are interpreted as the subdirectory of the root directory that the image was loaded from.

Note that a GPU with less than 16 GB memory will not be able to train on images of size 1024x1024.
See `python run_training.py --help` for info on training. This script is unique compared to the other runnable python files in that the settings can be specified beforehand in a yaml file. Just pass the yaml file as the first positional argument. Any following keyword arguments will take precedent over the same argument if found in the yaml file.
Example:
```yaml
# Here is an example training config file for 512x512 images.
channels: [32, 64, 128, 256, 512, 512, 512, 512]
tensorboard_log_dir: 'runs/stylegan2_512x512'
tensorboard_image_interval: 500
checkpoint_dir: 'checkpoints/stylegan2_512x512'
checkpoint_interval: 10000
data_dir: /path/to/dataset/images512x512/
gpu: [0, 1, 2, 3]
```
This training setup can be ran by `python run_training.py my_settings.yaml --gpu 0 --resume`.
In this example we did override one of the options, the `--gpu` option which was set to 4 devices in the settings file. But since the keyword arguments take precedent over the config file we will instead train only on one GPU device. We also added the argument `--resume` which will look in the checkpoint directory if we specified one and attempt to load the latest checkpoint before continuing to train. If no checkpoints are available, the training is started from scratch.

See `python run_metrics.py --help` for info on metric evaluation (metric evaluation can also be performed during training at specified intervals). This script supports multi-GPU metric evaluations.
Metrics will be appended to the file `metrics.json` if it exists. Identical metrics will be overwritten by the one that was last evaluated.

### Missing features
+ Precision-Recall metric
+ Multi-GPU evaluation of metrics during training

### Code information
Base functionality is tested but there might be settings that raise exceptions as the repository contains a couple thousand lines of code. If you find one, please open an issue or pull request.

### License
~5% of the code has extracts that may have been copied from the official repository. For this code, the [NVIDIA license](https://github.com/NVlabs/stylegan2/blob/master/LICENSE.txt) is applied. For everything else, the MIT license is applied.
