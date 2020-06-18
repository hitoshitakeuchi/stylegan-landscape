## System requirements

* Both Linux and Windows are supported.
* 64-bit Python 3.6 installation.
* Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.10.0 or newer with GPU support.
* One or more high-end NVIDIA GPUs with at least 11GB of DRAM.
* NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.3.1 or newer.

## Get the trained models

To actually run the trained GAN, you need the pretrained models.

These are available in this link and put this pkl file in `cache` folder.

https://drive.google.com/file/d/1fgmuI9aMH2Kge02UaUYMx3T_lzGSWTYK/view?usp=sharing

## Run

```bash
python encode_images.py img/ generated_images/ latent/
```

```bash
python morphing.py
```

## Reference

This code uses the official implementation of StyleGAN and StyleGAN Encoder.

https://github.com/NVlabs/stylegan

https://github.com/Puzer/stylegan-encoder
