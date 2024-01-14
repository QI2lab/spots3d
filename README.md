# spots3d
GPU accelerated spot finding and localization for both skewed (diSPIM, lattice, OPM) and standard geometry (widefield, confocal) 3D microscopy data.

Install
-------
Create a python>=3.10 and CUDA >=11.2, <12.0 environment. We will only use conda (or mambda if you prefer) for the intial CUDA installation.

For example if you are using conda,
```
conda create -n spots3d python=3.10
conda activate spots3d
conda install -c conda-forge cudatoolkit=11.8
```

Then, build and install the qi2lab branch of [Gpufit](https://github.com/QI2lab/Gpufit).

Finally, install SPOTS3D and cupy extras
```
pip install spots3d@git+https://git@github.com/qi2lab/spots3d@main#egg=spots3d'
python -m cupyx.tools.install_library --cuda 11.x --library cutensor
```

An optional, but recommended napari plugin for spot localization, that uses SPOTS3D and the qi2lab branch of Gpufit, is found at [napari-spot-detection](https://github.com/QI2lab/napari-spot-detection).
