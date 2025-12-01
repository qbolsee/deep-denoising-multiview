# Deep Denoising for Multiview Depth Cameras

This is the code for the paper titled [Deep Denoising for Multiview Depth Cameras](https://ieeexplore.ieee.org/abstract/document/9805795).

## Run

```
usage: run.py [-h] -i INPUT_DIR -o OUTPUT_DIR [-f] [-t THRESHOLD] [-s1] [-s2] [-n NUMBER] [--alpha ALPHA] [--beta BETA]
```

## Requirements

This code is designed to run on Python >= 3.10, with the following dependencies:

```
sklearn==0.0
tensorflow-gpu==1.13.2
tqdm>=4.0.0
numpy>=1.18.0
opencv-contrib-python>=3.4.0.0
h5py>=2.9.0
matplotlib>=3.0.0
plyfile>=0.7.0
```

## Citation

Bolsée, Q., Denis, L., Darwish, W., Kaashki, N. N., & Munteanu, A. (2022). Deep denoising for multiview depth cameras. IEEE Transactions on Instrumentation and Measurement, 71, 1-12.

## License

The software is provided under the MIT License.
