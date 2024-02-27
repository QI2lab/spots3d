"""
Image processing functions for OPM MERFISH data
"""

import numpy as np
from numba import njit, prange
import dask.array as da
from zarr.core import Array as zarr_array
from tqdm.dask import TqdmCallback
from tqdm import tqdm 
import gc
from functools import partial
import warnings
from typing import Union, Sequence, Tuple, Dict
from numpy.typing import NDArray
from cupy.typing import ArrayLike

# our packages
from localize_psf import localize
import localize_psf.rois as roi_fns
from localize_psf.fit_psf import gaussian3d_psf_model
import spots3d._skeweddatatools as skeweddatatools

# GPU
# cupy imports
_cupy_available = True
try:
    import cupy as cp # type: ignore
except ImportError:
    cp = np
    _cupy_available = False
else:
    from cupy.fft.config import get_plan_cache
    from cupyx.scipy import ndimage

# Deconvolution imports
_opencl_avilable = True
try:
    from clij2fft.richardson_lucy import richardson_lucy_nc as clij2_decon # type: ignore
    from clij2fft.libs import getlib # type: ignore
    DECON_LIBRARY = 'clij2'
except ImportError:
    _opencl_avilable = False
    DECON_LIBRARY = 'none'

def return_gpu_status() -> Tuple[bool,bool,str]:
    """
    Return flags for what GPU libraries are available for computation

    Parameters
    ----------
    None

    Returns
    -------
    cupy_available : bool
        Status of cupy library
    opencl_avilable : bool
        Status of clij2 library
    DECON_LIBRARY : str
        Name of deconvolution library ('clij2' or 'none')
    """

    return _cupy_available, _opencl_avilable, DECON_LIBRARY

def convert_data_to_dask(data: np.ndarray,
                         chunk_size: Sequence[int],
                         psf_size: Sequence[int]) -> Tuple[da.Array,Sequence[int]]:
    """
    Convert data from numpy array to dask array using provided scan chunk size

    Parameters
    ----------
    data : np.ndarray
        Data to convert
    chunk_size : Sequence[int]
        chunk sizes
    psf_size : Sequence[int]
        size of PSF
    
    Returns
    -------
    dask_data : da.Array
        Data as dask array
    overlap_depth : Sequence[int]
        overlap dimensions
    """

    if isinstance(data, np.ndarray):
        if data.shape[0]<chunk_size[0]:
            dask_data = da.from_array(data,
                                    chunks=(data.shape[0],chunk_size[1],chunk_size[2]))
            overlap_depth = (0,0,0)
        else:
            dask_data = da.from_array(data,
                                    chunks=(chunk_size[0],chunk_size[1],chunk_size[2]))
            overlap_depth = (psf_size[0]//2,0,0)
    elif isinstance(data, zarr_array):
        if data.shape[0]<chunk_size[0]:
            dask_data = da.from_zarr(data,
                                    chunks=(data.shape[0],chunk_size[1],chunk_size[2]))
            overlap_depth = (0,0,0)
        else:
            dask_data = da.from_zarr(data,
                                    chunks=(chunk_size[0],chunk_size[1],chunk_size[2]))
            overlap_depth = (psf_size[0],0,0)
    elif isinstance(data, da.Array):
        if data.shape[0]<chunk_size[0]:
            dask_data = da.rechunk(data,
                                    chunks=(data.shape[0],chunk_size[1],chunk_size[2]))
            overlap_depth = (0,0,0)
        else:
            dask_data = da.rechunk(data,
                                    chunks=(chunk_size[0],chunk_size[1],chunk_size[2]))
            overlap_depth = (psf_size[0],0,0)
    else:
        raise ValueError("data of unsupported type")
    
    del data
    gc.collect()

    return dask_data, overlap_depth

def clij_lr(image: np.ndarray,
            psf: np.ndarray,
            iterations: int = 30,
            tv_tau: float = .01,
            lib = None) -> np.ndarray:
    """
    Lucy-Richardson non-circulant with total variation deconvolvution 
    using clij2-fft (OpenCL)

    Parameters
    ----------
    image : np.ndarray
        Image to be deconvolved
    psf : np.ndarray
        Point spread function
    iterations : int
        Number of iterations
    tau : float
        Total variation parameter
    lib: Optional
        pre-initialized libfft

    Returns
    -------
    result: np.ndarray
        Deconvolved image
    """

    result = clij2_decon(image.astype(np.float32),
                         psf.astype(np.float32),
                         iterations,
                         tv_tau,
                         lib=lib)

    return result.astype(np.uint16)

def deconvolve(image: Union[np.ndarray, da.Array],
               psf: Union[np.ndarray, da.Array],
               decon_params: Dict,
               overlap_depth: Sequence[int]) -> np.ndarray:
    """
    Deconvolve current image in blocks using Dask and GPU. 
    Will not run if no GPU present.

    Parameters
    ----------
    image : np.ndarray
        Image data
    psf : np.ndarray
        Point spread function
    decon_params : Dict
        Deconvolution parameters
    overlap_depth : Sequence[int]
        Overlap padding

    Returns
    -------
    deconvolved_image : np.ndarray
        Deconolved image
    """

    iterations = decon_params['iterations']
    tv_tau = decon_params['tv_tau']
    lib = getlib()
    lr_dask_func = partial(clij_lr,psf=psf,iterations=iterations,tv_tau=tv_tau,lib=lib)
    decon_setup = True

    if decon_setup:
        dask_decon = da.map_overlap(lr_dask_func,
                                    image,
                                    depth=overlap_depth,
                                    boundary='reflect',
                                    trim=True,
                                    meta=np.array((), dtype=np.uint16))

        with TqdmCallback(desc="Deconvolve",leave=False):
            deconvolved_image = dask_decon.compute(scheduler='single-threaded')

        # clean up RAM
        del dask_decon, lr_dask_func, lib
        gc.collect()
    else:
        warnings.warn('GPU libraries not loaded, deconvolution not run.')

    return deconvolved_image.astype(np.uint16)

def perform_DoG_cartesian(image: Union[NDArray,ArrayLike],
                          kernel_small: Sequence[float],
                          kernel_large: Sequence[float],
                          pixel_size: float,
                          scan_step:  float) -> np.ndarray:
    """
    Perform difference of gaussian filter on cartesian image chunk

    Parameters
    ----------
    image : array
        Image to filter
    kernel_small : Sequence[float]
        Small Gaussian kernel sigmas (z,y,x)
    kernel_large : Sequence[float]
        Large Gaussian kernel sigmas (z,y,x)
    pixel_size : float
        Effective camera pixel size
    scan_step : float
        Spacing between z-planes

    Returns
    -------
    filtered : array
        Filtered image
    """
    kernel_small = localize.get_filter_kernel(kernel_small,
                                              [scan_step,
                                              pixel_size,
                                              pixel_size],
                                              sigma_cutoff = 2)
    kernel_large = localize.get_filter_kernel(kernel_large,
                                              [scan_step,
                                              pixel_size,
                                              pixel_size],
                                              sigma_cutoff = 2)
    
       
    if _cupy_available:
        image_cp = cp.asarray(image,dtype=cp.float32)
        image_hp = localize.filter_convolve(image_cp, kernel_small.astype(cp.float32))
        image_lp = localize.filter_convolve(image_cp, kernel_large.astype(cp.float32))
        image_filtered = cp.asnumpy(image_hp - image_lp)
        image_filtered[image_filtered<0.0]=0.0
        

        del image_cp, image_lp, image_hp
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    else:
        image_hp = localize.filter_convolve(image, kernel_small)
        image_lp = localize.filter_convolve(image, kernel_large)
        image_filtered = image_hp - image_lp
     
    return image_filtered.astype(np.float32)

def perform_DoG_skewed(image: np.ndarray,
                       kernel_small: Sequence[float],
                       kernel_large: Sequence[float],
                       pixel_size: float,
                       scan_step :  float,
                       theta : float) -> np.ndarray:
    """
    Perform difference of gaussian filter on skewed image chunk

    Parameters
    ----------
    image : array
        Image to filter
    kernel_small : Sequence[float]
        Small Gaussian kernel sigmas (z,y,x)
    kernel_large : Sequence[float]
        Large Gaussian kernel sigmas (z,y,x)
    pixel_size : float
        Effective camera pixel size
    scan_step : float
        Spacing between oblique planes
    theta: float
        Oblique angle in degrees

    Returns
    -------
    filtered : array
        Filtered image
    """

    kernel_small = skeweddatatools.get_filter_kernel_skewed(kernel_small, 
                                                            pixel_size, 
                                                            scan_step, 
                                                            theta,
                                                            sigma_cutoff=2)
    kernel_large = skeweddatatools.get_filter_kernel_skewed(kernel_large,
                                                            pixel_size,
                                                            scan_step,
                                                            theta, 
                                                            sigma_cutoff=2)

    if _cupy_available:
        image_cp = cp.asarray(image,dtype=cp.float32)
        image_hp = localize.filter_convolve(image_cp, kernel_small).astype(cp.float32)
        image_lp = localize.filter_convolve(image_cp, kernel_large).astype(cp.float32)
        image_filtered = cp.asnumpy(image_hp - image_lp)

        del image_cp, image_lp, image_hp
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    else:
        image_hp = localize.filter_convolve(image, kernel_small)
        image_lp = localize.filter_convolve(image, kernel_large)
        image_filtered = image_hp - image_lp
     
    return image_filtered.astype(np.float32)

def DoG_filter(image: Union[np.ndarray, da.Array],
               DoG_filter_params: Dict,
               overlap_depth: Sequence[int],
               image_params : Dict) -> np.ndarray:
    """
    Run difference of gaussian filter using Dask and GPU (if available).

    Parameters
    ----------
    image : Union[np.ndarray, da.core.Array]
        Image to filter
    DoG_filter_params : Dict
        DoG filtering parameters
    overlap_depth : Sequence[int]
        Size of overlap for dask map_overlap
    image_params : Dict
        {'pixel_size','scan_step','theta'}. Camera pixel size (µm), 
        spacing between oblique planes or z-planes(µm), and oblique angle (deg).

    Returns
    -------
    fitered_image : np.ndarray
        DoG filtered image
    """

    kernel_small = [DoG_filter_params['sigma_small_z'],
                    DoG_filter_params['sigma_small_y'],
                    DoG_filter_params['sigma_small_x']]

    kernel_large = [DoG_filter_params['sigma_large_z'],
                    DoG_filter_params['sigma_large_y'],
                    DoG_filter_params['sigma_large_x']]
    
    # check if we are using skewed or cartesian data
    if image_params['theta'] != 0:
        DoG_dask_func = partial(perform_DoG_skewed,
                                kernel_small=kernel_small,
                                kernel_large=kernel_large,
                                pixel_size=image_params['pixel_size'],
                                scan_step=image_params['scan_step'],
                                theta=image_params['theta'])
    else:
        DoG_dask_func = partial(perform_DoG_cartesian,
                                kernel_small=kernel_small,
                                kernel_large=kernel_large,
                                pixel_size=image_params['pixel_size'],
                                scan_step=image_params['scan_step'])

    if image.ndim == 3:
        if image_params['theta'] != 0:
            dask_dog_filter = da.map_overlap(DoG_dask_func,
                                            image,
                                            depth=overlap_depth,
                                            boundary='reflect',
                                            trim=True,
                                            meta=np.array((), dtype=np.float32))
        else:
            dask_dog_filter = da.map_overlap(DoG_dask_func,
                                            image,
                                            depth=[0,128,128],
                                            boundary='reflect',
                                            trim=True,
                                            meta=np.array((), dtype=np.float32))

        if _cupy_available:
            with TqdmCallback(desc="DoG filter"):
                filtered_image = dask_dog_filter.compute(scheduler='single-threaded')
            cp.clear_memo()
            cp._default_memory_pool.free_all_blocks()
        else:
            with TqdmCallback(desc="DoG filter"):
                filtered_image = dask_dog_filter.compute()

        del kernel_small, kernel_large, DoG_dask_func, dask_dog_filter
        gc.collect()
    elif image.ndim==4:
        print('\nDoG filter for each bit')
        print('-----------------------')
        filtered_image = np.zeros_like(image,dtype=np.float32)
        for bit_idx in tqdm(range(image.shape[0]),desc='Bit'):
            dask_dog_filter = da.map_overlap(DoG_dask_func,
                                            image[bit_idx,:],
                                            depth=overlap_depth,
                                            boundary='reflect',
                                            trim=True,
                                            meta=np.array((), dtype=np.float32))

            if _cupy_available:
                with TqdmCallback(desc="DoG filter",leave=False):
                    filtered_image[bit_idx,:] = dask_dog_filter.compute(scheduler='single-threaded')
                cp.clear_memo()
                cp._default_memory_pool.free_all_blocks()
            else:
                with TqdmCallback(desc="DoG filter",leave=False):
                    filtered_image[bit_idx,:] = dask_dog_filter.compute()
    else:
        raise ValueError(f"image.ndim = {image.ndim:d}, but must be either 3 or 4")

    return np.flip(filtered_image.astype(np.float32),axis=0)

def identify_candidates(image: Union[np.ndarray, da.Array],
                        coords: Sequence[np.ndarray],
                        dxy_min: float,
                        dz_min: float,
                        threshold: float,
                        pixel_size: float,
                        scan_step: float,
                        theta: float,
                        block_info: Dict = None) -> np.ndarray:
    """
    Identify and collapse nearby spot candidates in image chunk

    Parameters
    ----------
    image : Union[np.ndarray, da.core.Array]
        Image to filter
    coords : Sequence[np.ndarray]
        A Tuple of coordinates (z, y, x), where z, y, and x are broadcastable 
        to the same shape as image data.
    dxy_min : float
        Minimum spot size in xy
    dz_min : float
        Minimum spot size in z
    threshold: float
        Threshold for spot candidate amplitude
    pixel_size : float
        Effective camera pixel size
    scan_step : float
        Spacing between oblique planes or z-planes
    theta: float
        Oblique angle in degrees
    block_info: Dict
        map_overlap chunk metadata

    Returns
    -------
    candidate_results : np.ndarray
        Guesses for centers
    """

    z, y, x = coords

    # ---------------------------------------------------
    # useful code for diagnosing array-location due to
    # using an expanded array
    # ---------------------------------------------------
    # pprint(block_info[0])
    # print("========================")
    #chunk_location = block_info[0]['chunk-location'][0]
    x_min = block_info[0]['array-location'][2][0] 
    x_max = block_info[0]['array-location'][2][1]
    y_min = block_info[0]['array-location'][1][0]
    y_max = block_info[0]['array-location'][1][1]
    z_min = block_info[0]['array-location'][0][0]
    z_max = block_info[0]['array-location'][0][1]
    # print(x_min,x_max,y_min,y_max,z_min,z_max)
    # print(x.shape)
    # print(y.shape)
    # print(z.shape)
    x = x[:,:,x_min:x_max]
    y = y[z_min:z_max,y_min:y_max,:]
    z = z[:,y_min:y_max,:]

    if theta != 0:
        footprint = skeweddatatools.get_skewed_footprint((dz_min, dxy_min, dxy_min), 
                                                        pixel_size, 
                                                        scan_step, 
                                                        theta)
    else:
        footprint = localize.get_max_filter_footprint((dz_min, dxy_min, dxy_min), 
                                                      (scan_step, pixel_size, pixel_size))

    if _cupy_available:
        image_cp = cp.asarray(image)
        footprint_cp = cp.asarray(footprint)
        threshold_cp = cp.asarray(threshold)
        x_cp = cp.asarray(x)
        y_cp = cp.asarray(y)
        z_cp = cp.asarray(z)
        centers_guess_inds_cp, amps_cp = localize.find_peak_candidates(image_cp, 
                                                                       footprint_cp,
                                                                       threshold_cp)

        if theta != 0:
            # convert to xyz coordinates
            xc_cp = x_cp[0, 0, centers_guess_inds_cp[:, 2]]
            yc_cp = y_cp[centers_guess_inds_cp[:, 0], centers_guess_inds_cp[:, 1], 0]
            # z-position is determined by the y'-index in oblque acquired image
            zc_cp = z_cp[0, centers_guess_inds_cp[:, 1], 0] 
        else:
            # Alexis: make sure we can skip this step and define directly center_guess
            # convert to xyz coordinates
            xc_cp = x_cp[0, 0, centers_guess_inds_cp[:, 2]]
            yc_cp = y_cp[0, centers_guess_inds_cp[:, 1], 0]
            # z-position is determined by the y'-index in oblque acquired image
            zc_cp = z_cp[centers_guess_inds_cp[:, 0], 0, 0] 
        centers_guess_cp = cp.stack((zc_cp, yc_cp, xc_cp), axis=1)

        inds_cp = cp.ravel_multi_index(centers_guess_inds_cp.transpose(), 
                                       image_cp.shape)
        weights_cp = image_cp.ravel()[inds_cp]

        centers_guess_inds = cp.asnumpy(centers_guess_inds_cp)
        centers_guess = cp.asnumpy(centers_guess_cp)
        weights = cp.asnumpy(weights_cp)
        amps = cp.asnumpy(amps_cp)

        del image_cp, footprint_cp, threshold_cp, centers_guess_inds_cp, amps_cp
        del x_cp, y_cp, z_cp, xc_cp, yc_cp, zc_cp, centers_guess_cp, inds_cp, weights_cp
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()

    else:
        centers_guess_inds, amps = localize.find_peak_candidates(image, 
                                                                 footprint,
                                                                 threshold)

        if theta != 0:
            # convert to xyz coordinates
            xc = x[0, 0, centers_guess_inds[:, 2]]
            yc = y[centers_guess_inds[:, 0], centers_guess_inds[:, 1], 0]
            # z-position is determined by the y'-index in oblque acquired image
            zc = z[0, centers_guess_inds[:, 1], 0]
        else:
            # Alexis: make sure we can skip this step and define directly center_guess
            # convert to xyz coordinates
            xc = x[0, 0, centers_guess_inds[:, 2]]
            yc = y[0, centers_guess_inds[:, 1], 0]
            # z-position is determined by the y'-index in oblque acquired image
            zc = z[centers_guess_inds[:, 0], 0, 0]
        centers_guess = np.stack((zc, yc, xc), axis=1)

        inds = np.ravel_multi_index(centers_guess_inds.transpose(), image.shape)
        weights = image.ravel()[inds]
       
    # collapse neaby candidates to average position
    # remark: do we want to use the same values for minimum spot size and separation distances?
    centers_comb, inds_comb = localize.filter_nearby_peaks(centers_guess, 
                                                           dxy_min,
                                                           dz_min,
                                                           weights=weights,
                                                           mode="average",
                                                           nmax = np.inf)

    # extract raw amplitude for fitting routine
    amps = amps[inds_comb]
    
    zc_comb = centers_comb[:,0]
    yc_comb = centers_comb[:,1]
    xc_comb = centers_comb[:,2]

    candidates = np.stack((zc_comb,yc_comb,xc_comb,amps), axis=1)

    return np.transpose(candidates)

    # useful for diagnostic on raw data
    # centers_guess_inds[:,0] = centers_guess_inds[:,0]+z_min
    # centers_guess_inds[:,1] = centers_guess_inds[:,1]
    # centers_guess_inds[:,2] = centers_guess_inds[:,2]
    #return centers_guess_inds

def find_candidates(image: Union[np.ndarray, da.Array],
                    find_candidates_params: Dict,
                    coords: Tuple[np.ndarray,np.ndarray,np.ndarray],
                    image_params : Dict) -> np.ndarray:
    """
    Identify spot candidates using Dask and GPU (if available).

    Parameters
    ----------
    image : array
        Image to filter
    find_candidates_params : Dict
        Parameters for candidate finding.
    coords : Tuple[np.ndarray,np.ndarray,np.ndarray]
        A Tuple of coordinates (z, y, x), where z, y, and x are broadcastable 
        to the same shape as image data.
    image_params : Dict
        {'pixel_size','scan_step','theta'}. Camera pixel size, 
        spacing between oblique planes, and oblique angle.

    Returns
    -------
    candidate_results : np.ndarray
        Candidate spot centers and amplitudes
    """ 

    dxy_min = find_candidates_params['min_spot_xy']
    dz_min = find_candidates_params['min_spot_z']

    # To correctly address z,y,x indices, should be able to use chunk indices
    # https://docs.dask.org/en/stable/generated/dask.array.map_blocks.html
    # this requires "block_info" argument in function call.
    find_candidates_dask_func = partial(identify_candidates,
                                        coords=coords,
                                        dxy_min=dxy_min,
                                        dz_min=dz_min,
                                        threshold=find_candidates_params['threshold'],
                                        pixel_size=image_params['pixel_size'],
                                        scan_step=image_params['scan_step'],
                                        theta=image_params['theta'],
                                        block_info=None)

    # need to be careful here because output size is ragged!
    # https://blog.dask.org/2021/07/02/ragged-output
    # because we will again collapse spots, it's ok to have a few degenerate
    # candidates at the chunk edges 
    dask_find_candidates = da.map_blocks(find_candidates_dask_func,
                                         image,
                                         drop_axis=[1],
                                         meta=np.array((), dtype=np.float32))

    if _cupy_available:
        with TqdmCallback(desc="Finding candidates"):
            candidate_spots = dask_find_candidates.compute(scheduler='single-threaded')
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    else:
        with TqdmCallback(desc="Finding candidates"):
            candidate_spots = dask_find_candidates.compute()

    del dxy_min, dz_min, dask_find_candidates
    gc.collect()

    return np.transpose(candidate_spots)

def fit_candidate_spots(image: Union[np.ndarray, da.Array],
                        spot_candidates: np.ndarray,
                        coords: Sequence[np.ndarray],
                        fit_candidate_spots_params: Dict,
                        image_params: Dict,
                        return_rois: bool = False,
                        fit_model = gaussian3d_psf_model()) -> Sequence[np.ndarray]:

    """
    Fit candidates in image data. The strategy for this function is a bit different. 
    We manually process in chunks of 'n_spots_to_fit' candidates at a time. 
    The 'n_spots_to_fit' candidates are extracted as ROIs, fit with the GPU in a chunk, 
    then the next 'n_spots_to_fit' candidates, etc... until no candidates remain.

    This avoids having complicated chunking logic with Dask for the different sized 
    arrays of image data, coordinates, and spot locations.

    Parameters
    ----------
    image : Union[np.ndarray, da.core.Array]
        Image data
    spot_candidates : np.ndarray
        Guesses for spot candidate location and amplitudes (cz, cy, cx, amp)
    coords : Tuple[np.ndarray,np.ndarray,np.ndarray]
        A Tuple of coordinates (z, y, x), where z, y, and x are broadcastable 
        to the same shape as image data.
    image_params : Dict
        {'pixel_size','scan_step','theta'}. Camera pixel size, 
        spacing between oblique planes, and oblique angle.
    return_rois: bool
        True/False return ROIs
    fit_model: gpufit model
        GPUfit model for fitting

    Returns
    -------
    init_params : np.ndarray
        Initial guess for fitting
    fit_params : np.ndarray
        Results from fitting
    fit_states : np.ndarray
        Fitting states
    chi_sqrs : np.ndarray
        Mean squared errors
    niters : np.ndarray
        Number of iterations
    fit_t : np.ndarray
        T-test results
    fit_results : np.ndarray
        Summarized result fitting matrix
    """

    pixel_size = image_params['pixel_size']
    scan_step = image_params['scan_step']
    theta = image_params['theta']

    z, y, x = coords

    spots_to_fit = fit_candidate_spots_params['n_spots_to_fit']
    roi_size = [fit_candidate_spots_params['roi_z'],
                fit_candidate_spots_params['roi_y'],
                fit_candidate_spots_params['roi_x']]

    spot_idx_start = 0

    centers_guess = spot_candidates[:,0:3]
    # amps = spot_candidates[:,3]   # use max(roi) now

    if theta != 0:
        roi_size_pix = skeweddatatools.get_skewed_roi_size(roi_size,
                                                            pixel_size,
                                                            scan_step,
                                                            theta,
                                                            ensure_odd=True)
    else: 
        roi_size_pix = roi_fns.get_roi_size(roi_size, 
                                            [scan_step, pixel_size, pixel_size], 
                                            ensure_odd=True)

    pbar = tqdm(total=centers_guess.shape[0]//spots_to_fit,desc="Fitting spots")
    while spot_idx_start < centers_guess.shape[0]:
        if spot_idx_start+spots_to_fit >= centers_guess.shape[0]:
            spot_idx_stop = centers_guess.shape[0]
        else:
            spot_idx_stop = spot_idx_start+spots_to_fit
        spot_idx_range = np.arange(spot_idx_start, spot_idx_stop)
        # amps_chunk = amps[spot_idx_range]  # use max(roi) now
        centers_guess_chunk = centers_guess[spot_idx_range, :]

        # cut rois out
        image_np = image.compute()

        # todo: note that roi_centers[1] can be different by a few pixels from before
        # I have some ideas how to solve this ... but for the moment it is probably close enough
        if theta != 0:
            roi_centers = skeweddatatools.get_nearest_pixel(centers_guess_chunk, pixel_size, scan_step, theta)
        else:
            roi_centers = localize.get_nearest_pixel(centers_guess_chunk, (scan_step, pixel_size, pixel_size))
        rois = roi_fns.get_centered_rois(roi_centers,
                                         roi_size_pix,
                                         min_vals=np.array([0, 0, 0], dtype=int),
                                         max_vals=np.asarray(image_np.shape, dtype=int))

        if theta != 0:
            max_seps = np.array([np.inf, 0.5 * roi_size_pix[1]])

            # get maximum possible ROI size
            # todo: actually is smaller than this when accounting for mask
            max_roi_size = np.max(np.prod(rois[:, 1::2] - rois[:, ::2], axis=1))
            
            img_rois, coords_rois, roi_sizes = skeweddatatools.prepare_rois(image_np,
                                                                            coords,
                                                                            rois,
                                                                            centers_guess_chunk,
                                                                            max_seps,
                                                                            max_roi_size)
            # todo: should trim ROI's down to actual maximum size now
        else:
            img_rois, coords_rois, roi_sizes = localize.prepare_rois(image_np,
                                                                     coords,
                                                                     rois)

        # estimate parameters for each ROI
        init_params_chunk = fit_model.estimate_parameters(img_rois,
                                                          coords_rois,
                                                          num_preserved_dims=1)

        # fit ROI's
        fit_results = localize.fit_rois(img_rois,
                                        coords_rois,
                                        roi_sizes,
                                        init_params=init_params_chunk,
                                        estimator="LSE",
                                        model=fit_model)

        fit_params_chunk = fit_results["fit_params"]
        fit_states_chunk = fit_results["fit_states"]
        fit_states_key = fit_results["fit_states_key"]
        chi_sqrs_chunk = fit_results["chi_sqrs"]
        niters_chunk = fit_results["niters"]
        fit_t_chunk = fit_results["fit_time"]

        if spot_idx_start == 0:
            init_params = np.zeros(shape=(centers_guess.shape[0],
                                   init_params_chunk.shape[1]),
                                   dtype=init_params_chunk.dtype)
            fit_params = np.zeros(shape=(centers_guess.shape[0],
                                  fit_params_chunk.shape[1]),
                                  dtype=fit_params_chunk.dtype)
            fit_states = np.zeros(shape=(centers_guess.shape[0],),
                                  dtype=fit_states_chunk.dtype)
            chi_sqrs = np.zeros(shape=(centers_guess.shape[0],),
                                dtype=chi_sqrs_chunk.dtype)
            niters = np.zeros(shape=(centers_guess.shape[0],),
                              dtype=niters_chunk.dtype)
            fit_t = []

        init_params[spot_idx_start:spot_idx_stop,:] = init_params_chunk
        fit_params[spot_idx_start:spot_idx_stop,:] = fit_params_chunk
        fit_states[spot_idx_start:spot_idx_stop] = fit_states_chunk
        chi_sqrs[spot_idx_start:spot_idx_stop] = chi_sqrs_chunk
        niters[spot_idx_start:spot_idx_stop] = niters_chunk
        fit_t.append(fit_t_chunk)

        del img_rois, coords_rois, centers_guess_chunk
        del fit_params_chunk, fit_states_chunk, chi_sqrs_chunk, niters_chunk
        del fit_t_chunk
        del init_params_chunk
        if not return_rois:
            del rois
            rois = None

        gc.collect()

        spot_idx_start = spot_idx_stop + 1

        pbar.update(1)

    pbar.close()

    return init_params, fit_params, fit_states, fit_states_key, chi_sqrs, niters, np.array(fit_t), rois

def filter_localizations(fit_params : np.ndarray, 
                         init_params : np.ndarray, 
                         coords: Sequence[np.ndarray],
                         spot_filter_params : Dict,
                         return_values : bool = False) -> Tuple[np.ndarray]:
    """
    Filter spots based on fit results. Because this does not require image data, 
    we process all spots at once without chunking.

    Parameters
    ----------
    fit_params : np.ndarray
        Fitting results: 
        amplitude, x, y, z, sigma_xy, sigma_z, background
    init_params: np.ndarray
        Initial parameters per spot
    coords: Tuple[np.ndarray,np.ndarray,np.ndarray]
        A Tuple of coordinates (z, y, x), where z, y, and x are broadcastable 
        to the same shape as image data.
    spot_filter_params : Dict
        {'fit_dist_max_err','min_spot_sep','sigma_bounds','amp_min',
         'dist_boundary_min'}. Maximum distances between fit and initial guess center,
         assume points separated by less than this distance come from one spot, 
         exclude fits with sigmas that fall outside these ranges, lower bound on amplitude,
         and minimum distance from image boundary.
    
    Returns
    -------
    to_keep : np.ndarray
        Array of boolean values. True for accepted, False for rejected for each spot
    conditions : np.ndarray
        Results of filtering
    condition_names : Dict
        Names of conditions
    """

    # todo: might make life easier to replace this with localize.py filter object. See e.g. get_param_filter()
    z, y, x = coords

    fit_dist_max_err = spot_filter_params['fit_dist_max_err']
    min_spot_sep = spot_filter_params['min_spot_sep']
    sigma_bounds = spot_filter_params['sigma_bounds']
    amp_min = spot_filter_params['amp_min']
    dist_boundary_min = spot_filter_params['dist_boundary_min']
    sr_min = spot_filter_params['min_sigma_ratio']
    sr_max = spot_filter_params['max_sigma_ratio']

    centers_guess = np.concatenate((init_params[:, 3][:, None], 
                                    init_params[:, 2][:, None], 
                                    init_params[:, 1][:, None]), axis=1)
    centers_fit = np.concatenate((fit_params[:, 3][:, None], 
                                  fit_params[:, 2][:, None], 
                                  fit_params[:, 1][:, None]), axis=1)

    # ###################################################
    # only keep points if size and position were reasonable
    # ###################################################
    dz_min, dxy_min = dist_boundary_min

    in_bounds = skeweddatatools.point_in_trapezoid(centers_fit, coords)

    zmax, zmin = skeweddatatools.get_trapezoid_zbound(centers_fit[:, 1], coords)
    far_from_boundary_z = np.logical_and(centers_fit[:, 0] > zmin + dz_min,\
                                         centers_fit[:, 0] < zmax - dz_min)

    ymax, ymin = skeweddatatools.get_trapezoid_ybound(centers_fit[:, 0], coords)
    far_from_boundary_y = np.logical_and(centers_fit[:, 1] > ymin + dxy_min,\
                                         centers_fit[:, 0] < ymax - dxy_min)

    xmin = np.min(x)
    xmax = np.max(x)
    far_from_boundary_x = np.logical_and(centers_fit[:, 2] > xmin + dxy_min,\
                                         centers_fit[:, 2] < xmax - dxy_min)

    in_bounds = np.logical_and.reduce((in_bounds, 
                                       far_from_boundary_x, 
                                       far_from_boundary_y, 
                                       far_from_boundary_z))

    # maximum distance between fit center and guess center
    z_err_fit_max, xy_fit_err_max = fit_dist_max_err
    center_close_to_guess_xy = np.sqrt((centers_guess[:, 2] - fit_params[:, 1])**2 +
                                       (centers_guess[:, 1] - fit_params[:, 2])**2) <= xy_fit_err_max
    center_close_to_guess_z = np.abs(centers_guess[:, 0] - fit_params[:, 3]) <= z_err_fit_max

    # sigma ratios
    sigma_ratios = fit_params[:, 5] / fit_params[:, 4]

    # maximum/minimum sigmas AND combine all conditions
    (sz_min, sxy_min), (sz_max, sxy_max) = sigma_bounds
    conditions = np.stack((in_bounds, center_close_to_guess_xy, center_close_to_guess_z,
                            fit_params[:, 4] <= sxy_max, fit_params[:, 4] >= sxy_min,
                            fit_params[:, 5] <= sz_max, fit_params[:, 5] >= sz_min,
                            fit_params[:, 0] >= amp_min,
                            sigma_ratios <= sr_max, sigma_ratios >= sr_min), axis=1)


    # understand as "point excluded because *not* in_bounds" or "z_size *not* big enough"
    condition_names = ["in_bounds", "center_close_to_guess_xy", 
                       "center_close_to_guess_z", "xy_size_small_enough", 
                       "xy_size_big_enough", "z_size_small_enough",
                       "z_size_big_enough", "amp_high_enough",
                       "sigma_ratio_small_enough", "sigma_ratio_big_enough"]

    to_keep_temp = np.logical_and.reduce(conditions, axis=1)

    # ###################################################
    # check for unique points
    # ###################################################

    dz, dxy = min_spot_sep
    if np.sum(to_keep_temp) > 0:

        # only keep unique center if close enough
        _, unique_inds = localize.filter_nearby_peaks(centers_fit[to_keep_temp],
                                                      dxy, 
                                                      dz, 
                                                      mode="keep-one")

        # unique mask for those in to_keep_temp
        is_unique = np.zeros(np.sum(to_keep_temp), dtype=bool)
        is_unique[unique_inds] = True

        # get indices of non-unique points among all points
        not_unique_inds_full = np.arange(len(to_keep_temp), 
                                         dtype=int)[to_keep_temp][np.logical_not(is_unique)]

        # get mask in full space
        unique = np.ones(len(fit_params), dtype=bool)
        unique[not_unique_inds_full] = False
    else:
        unique = np.ones(len(fit_params), dtype=bool)

    conditions = np.concatenate((conditions, np.expand_dims(unique, axis=1)), axis=1)
    condition_names += ["unique"]
    to_keep = np.logical_and(to_keep_temp, unique)
    
    if return_values:
        dist_to_guess_xy = np.sqrt((centers_guess[:, 2] - fit_params[:, 1])**2 +
                                    (centers_guess[:, 1] - fit_params[:, 2])**2)
        dist_to_guess_z = np.abs(centers_guess[:, 0] - fit_params[:, 3])
        
        filter_values = np.stack((fit_params[:, 0], fit_params[:, 4], 
                                  fit_params[:, 5], sigma_ratios, dist_to_guess_xy, 
                                  dist_to_guess_z), axis=1)
        filter_names = ['amplitude', 'sigma xy', 'sigma z', 'sigmas ratio',
                        'dist to guess xy', 'dist to guess z']
        return to_keep, conditions, condition_names, filter_values, filter_names

    return to_keep, conditions, condition_names

@njit(parallel=True)
def deskew(data: np.ndarray,
           pixel_size: float = .115,
           scan_step: float = .400,
           theta: float = 30.0) -> np.ndarray:
    """
    Perform parallelized orthogonal interpolation into a uniform pixel size grid.
    
    Parameters
    ----------
    data : np.ndarray
        Image stack of uniformly spaced oblique image planes
    pizel_size: float 
        Effective camera pixel size in microns
    scan_step: float 
        Spacing between oblique planes in microns
    theta : float 
        Oblique angle in degrees
    
    Returns
    -------
    deskewed_image : np.ndarray
        Image stack of deskewed oblique planes on uniform grid
    """

    # unwrap parameters 
    [num_images,ny,nx]=data.shape     # (pixels)

    # change step size from physical space (nm) to camera space (pixels)
    pixel_step = scan_step/pixel_size    # (pixels)

    # calculate the number of pixels scanned during stage scan 
    scan_end = num_images * pixel_step  # (pixels)

    # calculate properties for final image
    final_ny = np.int64(np.ceil(scan_end+ny*np.cos(theta*np.pi/180))) # (pixels)
    final_nz = np.int64(np.ceil(ny*np.sin(theta*np.pi/180)))          # (pixels)
    final_nx = np.int64(nx)                                           # (pixels)

    # create final image
    # (time, pixels,pixels,pixels - data is float32)
    deskewed_image = np.zeros((final_nz, final_ny, final_nx),dtype=np.float32)  

    # precalculate trig functions for scan angle
    tantheta = np.float32(np.tan(theta * np.pi/180)) # (float32)
    sintheta = np.float32(np.sin(theta * np.pi/180)) # (float32)
    costheta = np.float32(np.cos(theta * np.pi/180)) # (float32)

    # perform orthogonal interpolation

    # loop through output z planes
    # defined as parallel loop in numba
    # http://numba.pydata.org/numba-doc/latest/user/parallel.html#numba-parallel
    for z in prange(0,final_nz):
        # calculate range of output y pixels to populate
        y_range_min=np.minimum(0,np.int64(np.floor(np.float32(z)/tantheta)))
        y_range_max=np.maximum(final_ny,np.int64(np.ceil(scan_end+np.float32(z)/tantheta+1)))

        # loop through final y pixels
        # defined as parallel loop in numba
        # http://numba.pydata.org/numba-doc/latest/user/parallel.html#numba-parallel
        for y in prange(y_range_min,y_range_max):

            # find the virtual tilted plane that intersects the interpolated plane 
            virtual_plane = y - z/tantheta

            # find raw data planes that surround the virtual plane
            plane_before = np.int64(np.floor(virtual_plane/pixel_step))
            plane_after = np.int64(plane_before+1)

            # continue if raw data planes are within the data range
            if ((plane_before>=0) and (plane_after<num_images)):
                
                # find distance of a point on the  interpolated plane to plane_before 
                # and plane_after
                l_before = virtual_plane - plane_before * pixel_step
                l_after = pixel_step - l_before
                
                # determine location of a point along the interpolated plane
                za = z/sintheta
                virtual_pos_before = za + l_before*costheta
                virtual_pos_after = za - l_after*costheta

                # determine nearest data points to interpoloated point in raw data
                pos_before = np.int64(np.floor(virtual_pos_before))
                pos_after = np.int64(np.floor(virtual_pos_after))

                # continue if within data bounds
                if ((pos_before>=0) and (pos_after >= 0) and\
                    (pos_before<ny-1) and (pos_after<ny-1)):
                    
                    # determine points surrounding interpolated point on virtual plane 
                    dz_before = virtual_pos_before - pos_before
                    dz_after = virtual_pos_after - pos_after

                    # compute final image plane using orthogonal interpolation
                    deskewed_image[z,y,:] = (l_before * dz_after *\
                                             data[plane_after,pos_after+1,:] +\
                                             l_before * (1-dz_after) *\
                                             data[plane_after,pos_after,:] +\
                                             l_after * dz_before *\
                                             data[plane_before,pos_before+1,:] +\
                                             l_after * (1-dz_before) *\
                                             data[plane_before,pos_before,:])/pixel_step

    return deskewed_image.astype(np.uint16)

def replace_hot_pixels(darkfield_image: Union[NDArray,ArrayLike], 
                       data: Union[NDArray,ArrayLike]) -> Union[NDArray,ArrayLike]:
    """
    Replace hot pixels with mean values surrounding it

    Parameters
    ----------
    darkfield_image: Union[NDArray,ArrayLike]
        darkfield image
    data: Union[NDArray,ArrayLike]
        ND data [broadcast_dim,z,y,x]
        
    Returns
    -------
    corrected_data: Union[NDArray,ArrayLike]
        corrected data
    """

    # threshold darkfield_image to generate bad pixel matrix
    if _cupy_available:
        hot_pixels_cp = cp.squeeze(cp.asarray(darkfield_image))
        hot_pixels_cp[hot_pixels_cp<=16] = 0
        hot_pixels_cp[hot_pixels_cp>16]=1
        hot_pixels_cp = hot_pixels_cp.astype(cp.float32)
        inverted_hot_pixels_cp = cp.ones_like(hot_pixels_cp) - hot_pixels_cp.copy()
    
        data_cp = cp.asarray(data,dtype=cp.float32)
        del data
        gc.collect()
        
        for z_idx in range(data.shape[0]):
            median_cp = ndimage.median_filter(data_cp[z_idx,:,:],size=3)
            data_cp[z_idx,:]=inverted_hot_pixels_cp*data_cp[z_idx,:] + hot_pixels_cp*median_cp
            
        data = cp.asnumpy(data_cp)
        del data_cp, hot_pixels_cp, inverted_hot_pixels_cp, median_cp
        gc.collect()
        cp.clear_memo()
        cp._default_memory_pool.free_all_blocks()
    else:
        hot_pixels = np.squeeze(darkfield_image)
        hot_pixels[hot_pixels<=16] = 0
        hot_pixels[hot_pixels>16] = 1
        hot_pixels = hot_pixels.astype(np.float32)
        inverted_hot_pixels = np.ones_like(hot_pixels) - hot_pixels.copy()
    
        for z_idx in range(data.shape[0]):
            median = ndimage.median_filter(data[z_idx,:,:],size=3)
            data[z_idx,:]=inverted_hot_pixels*data[z_idx,:] + hot_pixels*median

    return data.astype(np.uint16)