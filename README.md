# spots3d
GPU accelerated spot finding and localization for both skewed (diSPIM, lattice, OPM) and standard geometry (widefield, confocal) 3D microscopy data.

Install
-------
Create a python>=3.10 and CUDA >=11.2, <12.0 environment. We will use conda (or mamba if you prefer) for the CUDA and CuPy installation.

For Linux OS:
```
conda create -n spots3d python=3.10
conda activate spots3d
conda install -c conda-forge -c rapidsai -c nvidia cudatoolkit=11.8 cupy cucim=24.02 cuda-version=11.8
python -m cupyx.tools.install_library --cuda 11.x --library cutensor
```

For Windows OS:
```
conda create -n spots3d python=3.10
conda activate spots3d
conda install -c conda-forge -c nvidia cudatoolkit=11.8 cupy cuda-version=11.8
python -m cupyx.tools.install_library --cuda 11.x --library cutensor
```

On Windows OS, the required skimage portion of cuCIM will be built and installed during spots3d installation.

Then, build and install the qi2lab branch of [Gpufit](https://github.com/QI2lab/Gpufit).

Finally, install spots3d
```
pip install spots3d@git+https://git@github.com/qi2lab/spots3d@main#egg=spots3d
```

An optional, but recommended napari plugin for spot localization, that uses SPOTS3D and the qi2lab branch of Gpufit, is found at [napari-spot-detection](https://github.com/QI2lab/napari-spot-detection).
