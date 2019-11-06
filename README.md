# DCGAN

Minimalistic tf 2.0 implementation of DCGAN with support for distributed training on multiple GPUs.
Code compatibility for python>=3.6

## Dataset

`python download_celebA.py 0B7EVK8r0v71pZjFTYXZWM3FlRnM CelebA.zip`

## Run Code

For training:

`python dcgan.py --train`

For Generating new samples:

`python dcgan.py --generate`
