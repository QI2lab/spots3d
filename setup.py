from setuptools import setup, find_packages

required_pkgs = ['numpy',
                'pandas',
                'dask',
                'zarr',
                'joblib',
                'tqdm',
                'clij2-fft',
                'numba',
                'psfmodels',
                'localize_psf @ git+https://git@github.com/qi2lab/localize-psf@master#egg=localize_psf',
                ]

# check what platform we are on
if sys.platform == 'win32':
  required_pkgs.append(['cucim @ git+https://github.com/rapidsai/cucim.git@v23.02.00#egg=cucim&subdirectory=python/cucim'])

# extras
extras = {}

setup(
    name='spots3d',
    version="0.1.1",
    description="SPOTS3D",
    long_description="Class interface to qi2lab localization codes for finding and localizing spots in 3D images.",
    author="qi2lab",
    author_email="",
    packages=find_packages(include=["spots3d", "spots3d.*"]),
    python_requires='>=3.10',
    install_requires=required_pkgs,
    extras_require=extras)
