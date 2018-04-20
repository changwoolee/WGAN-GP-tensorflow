# WGAN-GP-tensorflow

Tensorflow implementation of paper ["Improved Training of Wasserstein GANs"](https://arxiv.org/abs/1704.00028).

![gif](https://thumbs.gfycat.com/VerifiableHonoredHind-size_restricted.gif)

* 0 epoch

![epoch0](http://cfile24.uf.tistory.com/image/99DE3E355AD971992E9F3C)

* 25 epoch

![img](http://cfile29.uf.tistory.com/image/99274A355AD9719925FEF4)

* 50 epoch

![epoch50](http://cfile23.uf.tistory.com/image/9927653B5AD971B537B169)

* 100 epoch

![img](http://cfile8.uf.tistory.com/image/996E113B5AD971CB1010F7)

* 150 epoch

![img](http://cfile28.uf.tistory.com/image/9999403C5AD971DB2483C5)

## Prerequisites

- Python 2.7 or 3.5
- Tensorflow 1.3+
- SciPy
- Aligned&Cropped celebA dataset([download](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0))
- (Optional) moviepy (for visualization)

## Usage

* Download aligned&cropped celebA dataset([link](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADSNUu0bseoCKuxuI5ZeTl1a/Img?dl=0)) and unzip at ./data/img_align_celeba

* Train:

  ```
  $ python main.py --train
  ```

  Or you can set some arguments like:

  ```
  $ python main.py --dataset=celebA --max_epoch=50 --learning_rate=1e-4 --train
  ```

* Test:

  ```
  $ python main.py
  ```

## Acknowledge

Based on the implementation [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow), [LynnHo/DCGAN-LSGAN-WGAN-WGAN-GP-Tensorflow](https://github.com/LynnHo/DCGAN-LSGAN-WGAN-WGAN-GP-Tensorflow), [shekkizh/WassersteinGAN.tensorflow](https://github.com/shekkizh/WassersteinGAN.tensorflow) and [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training).
