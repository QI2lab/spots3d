"""
Perform image pre-processing, spot candidate identification, spot fitting,
and spot filtering on a 3D image dataset with a known PSF. 

SPOTS3D: Designed for raw oblique data, such as that  acquired by oblique plane
microscopy or lattice light sheet microscopy.

Heavily relies on QI2lab localize-psf package and QI2lab helper functions.

Available routines:
- Deconvolution (GPU only)
- Difference of Gaussian filtering (CPU or GPU)
- Spot candidate identification (CPU or GPU)
- Spot fitting (CPU or GPU)
- Spot filtering (CPU or GPU)

Chained functionality for automated localization of spots in large images, 
either in memory or out-of-memory, is possible if parameters are provided.
"""

import numpy as np
import pandas as pd
import dask.array as da
import warnings
import gc
from typing import Union, Optional, Dict, Sequence
from localize_psf import localize
from spots3d import _imageprocessing, _skeweddatatools

class SPOTS3D:

    def __init__(self,
                 data: Union[np.ndarray, da.Array],
                 psf: Union[np.ndarray, da.Array],
                 metadata : Dict,
                 microscope_params: Optional[Dict],
                 scan_chunk_size : Optional[int] = 128,
                 decon_params: Optional[Dict] = None,
                 DoG_filter_params: Optional[Dict] = None,
                 find_candidates_params: Optional[Dict] = None,
                 fit_candidate_spots_params: Optional[Dict] = None,
                 spot_filter_params: Optional[Dict] = None,
                 chained: Optional[Dict] = None):
        """
        Find spots in 3D across iterative multiplexed, tiled imaging data. 
        This class is currently only works for skewed data.
        (i.e. oblique plane microscopy, dual illumination SPIM, lattice LSFM).
        Support for widefield data in work-in-progress.

        Parameters
        ----------
        data : Union[np.ndarray, da.Array]
            Data to process. Can be numpy array or dask array.
        psf : Union[np.ndarray, da.Array]
            PSF for data.
        metadata : Dict
            {'pixel_size','scan_step','wvl'}. Effective camera pixel size,
            distance between oblique plane, and emission wavelength.
        microscope_params : Dict
            {'na','ri','theta'}. Numerical aperture, refractive index, and oblique angle
            relative to coverslip.
        scan_chunk_size : Optional[int]
            Chunk size along scan direction
        decon_params : Optional[Dict]
            {'iterations','tv_tau'}. Number of iterations and TV regularization 
            parameter.
        DoG_filter_params : Optional[Dict]
            {'sigma_small_x','sigma_small_y','sigma_small_z',
             'sigma_large_x','sigma_large_y','sigma_large_z'}. 
             Size of Gaussian filters in % of PSF sigma for Difference of Gaussian filtering.
        find_candidates_params : Optional[Dict]
            {'threshold', 'min_spot_xy', 'min_spot_z'}. 
            Threshold for spot finding, xy and z distance to use for collapsing spots
        fit_candidate_spots_params: Optional[Dict]
            {'spots_to_fit','roi_z_factor','roi_y_factor','roi_x_factor'}. 
            Number of spots to fit at once. Multiplicative factors of PSF size in each 
            dimension for fitting ROI size.
        spot_filter_params : Optional[Dict]
            {'min_sigma_x','max_sigma_x','min_sigma_y','max_sigma_y','min_sigma_z',
             'max_sigma_z', 'min_sigma_ratio', 'max_sigma_ratio', 'min_amplitude','max_amplitude'}. 
             Ranges for allowed spot size and amplitude. 
        chained : Optional[Dict]
            {'deconvolve', 'dog_filter', 'find_candidates','localize','filter'}. 
            Boolean values for steps of pipeline to run chained. This 
            approach processing each image chunk fully and saves results to zarr
            before moving on. Can ingest disk-backed Dask arrays, allowing for 
            out of memory computation.
        """
        
        self._metadata = metadata
        if microscope_params is not None:
            self._microscope_params = microscope_params
            self._sigma_xy = 0.22 *\
                             self._metadata['wvl']\
                             / self._microscope_params['na']
            self._sigma_z = np.sqrt(6)\
                            / np.pi *\
                            self._microscope_params['ri'] *\
                            self._metadata['wvl']\
                            / self._microscope_params['na'] ** 2
        else:
            self._microscope_params = {'na' : 1.35,
                                       'ri' : 1.4,
                                       'theta' : 30}
            self._sigma_xy = 0.22 *\
                             self._metadata['wvl']\
                             / self._microscope_params['na']
            self._sigma_z = np.sqrt(6)\
                            / np.pi *\
                            self._microscope_params['ri'] *\
                            self._metadata['wvl']\
                            / self._microscope_params['na'] ** 2

        
        # set-up flag for cartesian vs skewed data
        if 'theta' not in self._microscope_params.keys():
            self._microscope_params['theta'] = 0
        self._is_skewed = self._microscope_params['theta'] != 0
        
        self._psf = psf
        if self._is_skewed:
            self._scan_chunk_size = scan_chunk_size
            self._chunk_size = [self._scan_chunk_size,data.shape[1],data.shape[2]]
        else:
            self._chunk_size = [data.shape[0],data.shape[1]//2,data.shape[2]//2]
        

        self._data,\
        self._overlap_depth = _imageprocessing.convert_data_to_dask(data,
                                                                   self._chunk_size,psf.shape)

        self._cupy_available,\
        self._opencl_avilable,\
        self._DECON_LIBRARY = _imageprocessing.return_gpu_status()



        

        if decon_params is not None:
            self._decon_params = decon_params
        else:
            decon_params = {'iterations' : 30,
                            'tv_tau' : .01}
            self._decon_params = decon_params

        if DoG_filter_params is not None:
            self._DoG_filter_params = DoG_filter_params
        else:
            DoG_filter_params = {'sigma_small_x_factor' : 0.707,
                                 'sigma_small_y_factor' : 0.707,
                                 'sigma_small_z_factor' : 0.707,
                                 'sigma_large_x_factor' : 1.4,
                                 'sigma_large_y_factor' : 1.4,
                                 'sigma_large_z_factor' : 1.4}                            
            self._DoG_filter_params = DoG_filter_params
        
        self._DoG_filter_params['sigma_small_x'] = self._DoG_filter_params['sigma_small_x_factor'] * self._sigma_xy
        self._DoG_filter_params['sigma_small_y'] = self._DoG_filter_params['sigma_small_y_factor'] * self._sigma_xy
        self._DoG_filter_params['sigma_small_z'] = self._DoG_filter_params['sigma_small_z_factor'] * self._sigma_z
        self._DoG_filter_params['sigma_large_x'] = self._DoG_filter_params['sigma_large_x_factor'] * self._sigma_xy
        self._DoG_filter_params['sigma_large_y'] = self._DoG_filter_params['sigma_large_y_factor'] * self._sigma_xy
        self._DoG_filter_params['sigma_large_z'] = self._DoG_filter_params['sigma_large_z_factor'] * self._sigma_z   
        
        if find_candidates_params is not None:
            self._find_candidates_params = find_candidates_params
        else:
            find_candidates_params = {'threshold': 100,
                                      'min_spot_xy_factor': 2.5,
                                      'min_spot_z_factor' : 2.5}
            self._find_candidates_params = find_candidates_params
        
        self._find_candidates_params['min_spot_xy'] = self._find_candidates_params['min_spot_xy_factor'] * self._sigma_xy
        self._find_candidates_params['min_spot_z'] = self._find_candidates_params['min_spot_z_factor'] * self._sigma_z
     
        if fit_candidate_spots_params is not None:
            self._fit_candidate_spots_params = fit_candidate_spots_params
        else:
            fit_candidate_spots_params = {'n_spots_to_fit': 5000,
                                          'roi_z_factor': 3,
                                          'roi_y_factor': 2,
                                          'roi_x_factor': 2}
            self._fit_candidate_spots_params = fit_candidate_spots_params
        
        self._fit_candidate_spots_params['roi_z'] =\
            self._fit_candidate_spots_params['roi_z_factor'] * self._sigma_z
        self._fit_candidate_spots_params['roi_y'] =\
            self._fit_candidate_spots_params['roi_y_factor'] * self._sigma_xy
        self._fit_candidate_spots_params['roi_x'] =\
            self._fit_candidate_spots_params['roi_x_factor'] * self._sigma_xy
        
        if spot_filter_params is not None:
            self._spot_filter_params = spot_filter_params
        else:
            spot_filter_params = {'sigma_min_z_factor' : 0.2,
                                  'sigma_min_xy_factor' : 0.25,               
                                  'sigma_max_z_factor' : 6,
                                  'sigma_max_xy_factor' : 8,
                                  'fit_dist_max_err_z_factor': 5,
                                  'fit_dist_max_err_xy_factor': 7,
                                  'min_spot_sep_z_factor' : 2,
                                  'min_spot_sep_xy_factor' : 1,
                                  'amp_min' : 50,
                                  'dist_boundary_z_factor' : .05,
                                  'dist_boundary_xy_factor' : .05,
                                  'min_sigma_ratio' : 1.25, 
                                  'max_sigma_ratio' : 6,
                                  }
            self._spot_filter_params = spot_filter_params

        self._spot_filter_params['sigma_min_xy'] = self._spot_filter_params['sigma_min_xy_factor'] * self._sigma_xy
        self._spot_filter_params['sigma_min_z'] =  self._spot_filter_params['sigma_min_z_factor'] * self._sigma_z
        self._spot_filter_params['sigma_max_xy'] =  self._spot_filter_params['sigma_max_xy_factor'] * self._sigma_xy
        self._spot_filter_params['sigma_max_z'] =  self._spot_filter_params['sigma_max_z_factor'] * self._sigma_z
        self._spot_filter_params['sigma_bounds'] = ((self._spot_filter_params['sigma_min_z'],self._spot_filter_params['sigma_min_xy']),
                                                    (self._spot_filter_params['sigma_max_z'],self._spot_filter_params['sigma_max_xy']))
        self._spot_filter_params['fit_dist_max_err'] = ((self._spot_filter_params['fit_dist_max_err_z_factor'] * self._sigma_z),
                                                        (self._spot_filter_params['fit_dist_max_err_xy_factor'] * self._sigma_xy))
        self._spot_filter_params['min_spot_sep'] = ((self._spot_filter_params['min_spot_sep_z_factor'] * self._sigma_z),
                                                    (self._spot_filter_params['min_spot_sep_xy_factor'] * self._sigma_xy))
        self._spot_filter_params['dist_boundary_min'] = ((self._spot_filter_params['dist_boundary_z_factor'] * self._metadata['pixel_size']),
                                                         (self._spot_filter_params['dist_boundary_xy_factor'] * self._metadata['pixel_size']))
        self._spot_filter_params['sigma_ratios_bounds'] = (self._spot_filter_params['min_sigma_ratio'],self._spot_filter_params['max_sigma_ratio'])
        
        if chained is not None:
            self._chained = chained
        else:        
            self._chained= {
                'deconvolve' : True,
                'dog_filter' : True,
                'find_candidates' : True,
                'merge_candidates' : True,
                'localize' : True,
                'save' : True
            }

        self._dog_filter_source_data = 'raw'
        self._find_candidates_source_data = 'raw'

        self._decon_data = None
        self._dog_data = None
        self._spot_candidates = None
        self._collapsed_candidates = None
        self._fitting_results = None
        self._filtered_results = None

        self._image_params = {'pixel_size' : self._metadata['pixel_size'],
                              'scan_step' : self._metadata['scan_step'],
                              'theta' : self._microscope_params['theta']}

        if self._is_skewed:
            z,y,x = _skeweddatatools.get_skewed_coords(sizes = self._data.shape,
                                                       image_params = self._image_params,
                                                       pixel_size = None,
                                                       scan_step = None,
                                                       theta = None)
        else:
            z, y, x = localize.get_coords(self._data.shape,
                                          (self._image_params['scan_step'], 
                                          self._image_params['pixel_size'], 
                                          self._image_params['pixel_size']))
        self._coords = (z,y,x)
        del z,y,x
        self._return_rois = False

    # -----------------------------------
    # property access for class variables
    # -----------------------------------
    @property
    def data(self):
        return self._data.compute()

    @data.setter
    def data(self,new_data: Union[np.ndarray, da.Array]):
        self._data,\
        self._overlap_depth = _imageprocessing.convert_data_to_dask(new_data,
                                                                   self._chunk_size,
                                                                   self._psf.shape)

    @data.deleter
    def data(self):
        del self._data

    @property
    def decon_data(self):
        try:
            return self._decon_data.compute()
        except:
            return None
    
    @decon_data.deleter
    def decon_data(self):
        try:
            del self._decon_data
            gc.collect()
        except:
            pass
    
    @property
    def dog_data(self):
        try:
            return self._dog_data.compute()
        except:
            return None
    
    @dog_data.deleter
    def dog_data(self):
        try:
            del self._dog_data
            gc.collect()
        except:
            pass

    @property
    def psf(self):
        return self._psf

    @psf.setter
    def psf(self,new_psf: Union[np.ndarray, da.Array]):
        self._psf = new_psf

    @psf.deleter
    def psf(self):
        del self._psf

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self,new_metadata: Dict):
        self._metadata.update(new_metadata)

        self._sigma_xy = 0.22 *\
                         self._metadata['wvl']\
                         / self._microscope_params['na']
        self._sigma_z = np.sqrt(6)\
                        / np.pi *\
                        self._microscope_params['ri'] *\
                        self._metadata['wvl']\
                        / self._microscope_params['na'] ** 2

    @metadata.deleter
    def metadata(self):
        del self._metadata

    @property
    def microscope_params(self):

        # only return public keys
        keys = ['na', 'ri', 'theta']
        public_microscope_params = dict((key, self._microscope_params[key]) for key in keys)

        return public_microscope_params

    @microscope_params.setter
    def microscope_params(self,new_microscope_params : Dict):
        self._microscope_params.update(new_microscope_params)
        self._sigma_xy = 0.22 *\
                         self._metadata['wvl']\
                         / self._microscope_params['na']
        self._sigma_z = np.sqrt(6)\
                        / np.pi *\
                        self._microscope_params['ri'] *\
                        self._metadata['wvl']\
                        / self._microscope_params['na'] ** 2
        self._is_skewed = self._microscope_params['theta'] != 0
        self._image_params['theta'] = self._microscope_params['theta']

    @property
    def scan_chunk_size(self):
        return self._data.chunks

    @scan_chunk_size.setter
    def scan_chunk_size(self,new_scan_chunk_size : Sequence[int]):
        
        if self._is_skewed:
            self._scan_chunk_size = new_scan_chunk_size
            self._chunk_size = [self._scan_chunk_size,
                                self._data.shape[1],
                                self._data.shape[2]]
            self._data, self._overlap_depth = _imageprocessing.convert_data_to_dask(self._data,
                                                                                    self._chunk_size,
                                                                                    self._psf.shape)
            if self._decon_data is not None:
                self._decon_data, self._overlap_depth = _imageprocessing.convert_data_to_dask(self._decon_data,
                                                                                            self._chunk_size,
                                                                                            self._psf.shape)
            if self._dog_data is not None:
                self._dog_data, self._overlap_depth = _imageprocessing.convert_data_to_dask(self._dog_data,
                                                                                            self._chunk_size,
                                                                                            self._psf.shape)
            # # check if _overlap is greater than the number of scan positions
            # if self._overlap_depth[0] > self._data.shape[0]:
            #     print(f"Initial chunk sizes for skewed data:\ndata shape:{self._data.shape} \nchunk size:{self._chunk_size} \noverlap:{self._overlap_depth}")
            #     overlap_list = list(self._overlap_depth)
            #     overlap_list[0] = self._data.shape[0]
            #     self._overlap_depth = tuple(overlap_list)
                
            #     print(f"New overlap:{self._overlap_depth}")
            # print(f"Initial chunk sizes for skewed data:\ndata shape:{self._data.shape} \nchunk size:{self._chunk_size} \noverlap:{self._overlap_depth}")
        
    @property
    def wf_chunk_size(self):
        return self._data.chunks

    @wf_chunk_size.setter
    def wf_chunk_size(self,new_wf_chunk_size : Sequence[int]):
        self._wf_chunk_size = new_wf_chunk_size
        self._data, self._overlap_depth = _imageprocessing.convert_data_to_dask(self._data,
                                                                                self._wf_chunk_size,
                                                                                self._psf.shape)
        if self._decon_data is not None:
            self._decon_data, self._overlap_depth = _imageprocessing.convert_data_to_dask(self._decon_data,
                                                                                          self._wf_chunk_size,
                                                                                          self._psf.shape)
        if self._dog_data is not None:
            self._dog_data, self._overlap_depth = _imageprocessing.convert_data_to_dask(self._dog_data,
                                                                                        self._wf_chunk_size,
                                                                                        self._psf.shape)

    @property
    def decon_params(self):
        return self._decon_params

    @decon_params.setter
    def decon_params(self,new_decon_params: Dict):
        self._decon_params.update(new_decon_params)

    @property
    def DoG_filter_params(self):

        # only return public keys
        keys = ['sigma_small_z_factor','sigma_small_y_factor','sigma_small_x_factor',
                'sigma_large_z_factor','sigma_large_y_factor','sigma_large_x_factor']
        public_DoG_filter_params = dict((key, self._DoG_filter_params[key]) for key in keys)

        return public_DoG_filter_params

    @DoG_filter_params.setter
    def DoG_filter_params(self,new_DoG_filter_params: Dict):
        self._DoG_filter_params.update(new_DoG_filter_params)

        self._DoG_filter_params['sigma_small_x'] = self._DoG_filter_params['sigma_small_x_factor'] * self._sigma_xy
        self._DoG_filter_params['sigma_small_y'] = self._DoG_filter_params['sigma_small_y_factor'] * self._sigma_xy
        self._DoG_filter_params['sigma_small_z'] = self._DoG_filter_params['sigma_small_z_factor'] * self._sigma_z
        self._DoG_filter_params['sigma_large_x'] = self._DoG_filter_params['sigma_large_x_factor'] * self._sigma_xy
        self._DoG_filter_params['sigma_large_y'] = self._DoG_filter_params['sigma_large_y_factor'] * self._sigma_xy
        self._DoG_filter_params['sigma_large_z'] = self._DoG_filter_params['sigma_large_z_factor'] * self._sigma_z

    @property
    def find_candidates_params(self):

        # only return public keys
        keys = ['threshold','min_spot_z_factor','min_spot_xy_factor']
        public_find_candidates_params = dict((key, self._find_candidates_params[key]) for key in keys)

        return public_find_candidates_params

    @find_candidates_params.setter
    def find_candidates_params(self,new_find_candidates_params: Dict):
        self._find_candidates_params.update(new_find_candidates_params)

        self._find_candidates_params['min_spot_xy'] = self._find_candidates_params['min_spot_xy_factor'] * self._sigma_xy
        self._find_candidates_params['min_spot_z'] = self._find_candidates_params['min_spot_z_factor'] * self._sigma_z

    @property
    def fit_candidate_spots_params(self):

        # only return public keys
        keys = ['n_spots_to_fit','roi_z_factor','roi_y_factor','roi_x_factor']
        public_fit_candidate_spots_params = dict((key, self._fit_candidate_spots_params[key]) for key in keys)

        return public_fit_candidate_spots_params

    @fit_candidate_spots_params.setter
    def fit_candidate_spots_params(self,new_fit_candidate_spots_params: Dict):
        self._fit_candidate_spots_params.update(new_fit_candidate_spots_params)

        self._fit_candidate_spots_params['roi_z'] =\
                self._fit_candidate_spots_params['roi_z_factor'] * self._sigma_z
        self._fit_candidate_spots_params['roi_y'] =\
            self._fit_candidate_spots_params['roi_y_factor'] * self._sigma_xy
        self._fit_candidate_spots_params['roi_x'] =\
            self._fit_candidate_spots_params['roi_x_factor'] * self._sigma_xy

    @property
    def spot_filter_params(self):

        # only return public keys
        keys = ['sigma_min_z_factor', 'sigma_min_xy_factor', 'sigma_max_z_factor', 'sigma_max_xy_factor',
                'fit_dist_max_err_z_factor', 'fit_dist_max_err_xy_factor', 'min_spot_sep_z_factor', 'min_spot_sep_xy_factor',
                'amp_min', 'dist_boundary_z_factor', 'dist_boundary_xy_factor', 'min_sigma_ratio', 'max_sigma_ratio']
        public_spot_filter_params = dict((key, self._spot_filter_params[key]) for key in keys)

        return public_spot_filter_params

    @spot_filter_params.setter
    def spot_filter_params(self,new_spot_filter_params: Dict):
        self._spot_filter_params.update(new_spot_filter_params)

        self._spot_filter_params['sigma_min_xy'] = self._spot_filter_params['sigma_min_xy_factor'] * self._sigma_xy
        self._spot_filter_params['sigma_min_z'] =  self._spot_filter_params['sigma_min_z_factor'] * self._sigma_z
        self._spot_filter_params['sigma_max_xy'] =  self._spot_filter_params['sigma_max_xy_factor'] * self._sigma_xy
        self._spot_filter_params['sigma_max_z'] =  self._spot_filter_params['sigma_max_z_factor'] * self._sigma_z
        self._spot_filter_params['sigma_bounds']= ((self._spot_filter_params['sigma_min_z'], self._spot_filter_params['sigma_min_xy']),
                                                   (self._spot_filter_params['sigma_max_z'], self._spot_filter_params['sigma_max_xy']))
        self._spot_filter_params['fit_dist_max_err'] = ((self._spot_filter_params['fit_dist_max_err_z_factor'] * self._sigma_z),
                                                        (self._spot_filter_params['fit_dist_max_err_xy_factor'] * self._sigma_xy))
        self._spot_filter_params['min_spot_sep'] = ((self._spot_filter_params['min_spot_sep_z_factor'] * self._sigma_z),
                                                    (self._spot_filter_params['min_spot_sep_xy_factor'] * self._sigma_xy))
        self._spot_filter_params['dist_boundary_min'] = ((self._spot_filter_params['dist_boundary_z_factor'] * self._metadata['pixel_size']),
                                                         (self._spot_filter_params['dist_boundary_xy_factor'] * self._metadata['pixel_size']))
    @property
    def dog_filter_source_data(self):
        return self._dog_filter_source_data

    @dog_filter_source_data.setter
    def dog_filter_source_data(self,new_dog_filter_source_data : str):
        self._dog_filter_source_data = new_dog_filter_source_data

    @property
    def find_candidates_source_data(self):
        return self._find_candidates_source_data

    @find_candidates_source_data.setter
    def find_candidates_source_data(self,new_find_candidates_source_data : str):
        self._find_candidates_source_data = new_find_candidates_source_data

    @property
    def chained(self):
        return self._chained

    @chained.setter
    def chained(self,new_chained : Dict):
        self._chained.update(new_chained)

    def run_deconvolution(self):
        if self._decon_params is not None:
            if self._is_skewed:
                self._decon_data = _imageprocessing.deconvolve(self._data,
                                                            self._psf,
                                                            self._decon_params,
                                                            self._overlap_depth)
                self._decon_data, _ = _imageprocessing.convert_data_to_dask(self._decon_data,
                                                                            self._chunk_size,
                                                                            self._psf.shape)
            else:
                from clij2fft.richardson_lucy_dask import richardson_lucy_dask
                self._decon_data = richardson_lucy_dask(self._data.compute(),
                                                        self._psf,
                                                        numiterations=int(self._decon_params['iterations']),
                                                        regularizationfactor=float(self._decon_params['tv_tau']),
                                                        mem_to_use=12).astype(np.uint16)
                self._chunk_size = [self._data.shape[0],self._data.shape[1]//2,self._data.shape[2]//2]
                self._decon_data, _ = _imageprocessing.convert_data_to_dask(self._decon_data,
                                                                            self._chunk_size,
                                                                            self._psf.shape)
        else:
            warnings.warn("No deconvolution parameters.")

    def run_DoG_filter(self):

        if self._DoG_filter_params is not None:
            if self._dog_filter_source_data == 'raw':
                self._dog_data = _imageprocessing.DoG_filter(self._data,
                                                            self._DoG_filter_params,
                                                            self._overlap_depth,
                                                            self._image_params)
                if not(self._is_skewed):
                    self._chunk_size = [self._data.shape[0],self._data.shape[1]//2,self._data.shape[2]//2]
                self._dog_data, _ = _imageprocessing.convert_data_to_dask(self._dog_data,
                                                                          self._chunk_size,
                                                                          self._psf.shape)
            elif self._dog_filter_source_data == 'decon':
                self._dog_data = _imageprocessing.DoG_filter(self._decon_data,
                                                            self._DoG_filter_params,
                                                            self._overlap_depth,
                                                            self._image_params)
                if not(self._is_skewed):
                    self._chunk_size = [self._data.shape[0],self._data.shape[1]//2,self._data.shape[2]//2]
                self._dog_data, _ = _imageprocessing.convert_data_to_dask(self._dog_data,
                                                                          self._chunk_size,
                                                                          self._psf.shape)
            else:
                warnings.warn("No source data for Difference of Gaussian filtering.")
        else:
            warnings.warn("No Difference of Gaussian filter kernel sizes.")

    def run_find_candidates(self):

        if self._find_candidates_params is not None:
            if self._find_candidates_source_data == 'raw':
                    self._spot_candidates =\
                        _imageprocessing.find_candidates(self._data,
                                                        self._find_candidates_params,
                                                        self._coords,
                                                        self._image_params)
            elif self._find_candidates_source_data == 'decon':
                if self._decon_data is not None:
                    self._spot_candidates =\
                        _imageprocessing.find_candidates(self._decon_data,
                                                        self._find_candidates_params,
                                                        self._coords,
                                                        self._image_params)
                else:
                    warnings.warn("Run deconvolution before finding spot candidates.")
            elif self._find_candidates_source_data == 'dog':
                if self._dog_data is not None:
                    self._spot_candidates =\
                        _imageprocessing.find_candidates(self._dog_data,
                                                        self._find_candidates_params,
                                                        self._coords,
                                                        self._image_params)
                else:
                    warnings.warn("Run DoG filtering before finding spot candidates.")
        else:
            warnings.warn("No spot finding candidate parameters.")

    # TO DO: add variable to select what data to fit
    def run_fit_candidates(self):
        if self._fit_candidate_spots_params is not None:
            if self._spot_candidates is not None:
                self._fitting_results =\
                    _imageprocessing.fit_candidate_spots(self._data, 
                                                         self._spot_candidates,
                                                         self._coords,
                                                         self._fit_candidate_spots_params,
                                                         self._image_params,
                                                         self._return_rois)

                self._init_params = self._fitting_results[0]
                self._fit_params = self._fitting_results[1]
                self._fit_states = self._fitting_results[2]
                self._fit_states_key = self._fitting_results[3]
                self._chi_sqrs = self._fitting_results[4] 
                self._niters = self._fitting_results[5]
                self._fit_t = self._fitting_results[6]
                if self._return_rois:
                    self._rois = self._fitting_results[7]
        else:
            warnings.warn("Generate candidate list before fitting.")

    def run_filter_spots(self, return_values=False):

        if self._spot_filter_params is not None:
            if self._fitting_results is not None:
                if return_values:
                    if self._is_skewed:
                        self._to_keep, self._conditions, self._condition_names, \
                        self._filter_values, self._filter_names =\
                            _imageprocessing.filter_localizations(self._fit_params,
                                                                  self._init_params,
                                                                  self._coords,
                                                                  self._spot_filter_params,
                                                                  return_values)
                    else:
                        self._to_keep, self._conditions, self._condition_names, self._filter_settings, \
                        self._filter_values, self._filter_names =\
                            localize.filter_localizations(self._fit_params,
                                                          self._init_params,
                                                          self._coords,
                                                          self._spot_filter_params['fit_dist_max_err'],
                                                          self._spot_filter_params['min_spot_sep'],
                                                          self._spot_filter_params['sigma_bounds'],
                                                          self._spot_filter_params['amp_min'],
                                                          self._spot_filter_params['dist_boundary_min'],
                                                          self._spot_filter_params['sigma_ratios_bounds'],
                                                          return_values)
                else:
                    if self._is_skewed:
                        self._to_keep, self._conditions, self._condition_names =\
                            _imageprocessing.filter_localizations(self._fit_params,
                                                                  self._init_params,
                                                                  self._coords,
                                                                  self._spot_filter_params)
                    else:
                        self._to_keep, self._conditions, self._condition_names, self._filter_settings =\
                            localize.filter_localizations(self._fit_params,
                                                          self._init_params,
                                                          self._coords,
                                                          self._spot_filter_params['fit_dist_max_err'],
                                                          self._spot_filter_params['min_spot_sep'],
                                                          self._spot_filter_params['sigma_bounds'],
                                                          self._spot_filter_params['amp_min'],
                                                          self._spot_filter_params['dist_boundary_min'],
                                                          self._spot_filter_params['sigma_ratios_bounds'],
                                                          return_values)

            else:
                warnings.warn("Generate fitting results before filtering.")
        else:
            warnings.warn("No result filtering parameters.")
    
    def save_results(self, dir_localize, base_name=None):

        if base_name is None:
            base_name = ''
        else:
            base_name = '_' + base_name

        # save coords z / y / x and boolean selector
        centers_fit = self._fit_params[:, 3:0:-1]
        centers_fit = np.hstack([centers_fit, self._to_keep.reshape((-1, 1))])
        df = pd.DataFrame(
            data=centers_fit,
            columns=['z', 'y', 'x', 'select'],
            )
        df['select'] = df['select'].astype(bool)
        file_name = f'localized_spots{base_name}.parquet'
        df.to_parquet(dir_localize / file_name)

        # save fitted variables
        file_name = f'fitted_variables{base_name}.parquet'
        df = pd.DataFrame(
            data=self._fit_params,
            columns=['amplitude', 'x', 'y', 'z', 'sigma_xy', 'sigma_z', 'background'],
            )
        df.to_parquet(dir_localize / file_name)

        # save candidates
        file_name = f'localization_candidates{base_name}.parquet'
        df = pd.DataFrame(
            data=self._spot_candidates,
            columns=['z', 'y', 'x', 'amplitude'],
            )
        df.to_parquet(dir_localize / file_name)

        # save filter conditions
        file_name = f'filter_conditions{base_name}.parquet'
        df = pd.DataFrame(
            data=self._conditions,
            columns=[
                "in_bounds", "center_close_to_guess_xy", 
                "center_close_to_guess_z", "xy_size_small_enough", 
                "xy_size_big_enough", "z_size_small_enough",
                "z_size_big_enough", "amp_ok",
                "sigma_ratio_small_enough", "sigma_ratio_big_enough", 
                "unique"],
            ).astype(bool)
        df.to_parquet(dir_localize / file_name)

        # save filter values
        file_name = f'filter_values{base_name}.parquet'
        df = pd.DataFrame(
            data=self._filter_values,
            columns=['amplitude', 'sigma xy', 'sigma y', 'sigmas ratio',
                        'dist to guess xy', 'dist to guess z'],
            )
        df.to_parquet(dir_localize / file_name)

    def run_chained(self,dir_localize, base_name=None):

        if self._chained is not None:
            if self._chained['deconvolve']:
                self.run_deconvolution()
                self._dog_filter_source_data == 'decon'
            else:
                self._dog_filter_source_data == 'raw'
            if self._chained['dog_filter']:
                self.run_DoG_filter()
            if self._chained['find_candidates']:
                self.run_find_candidates()
            if len(self._spot_candidates) > 0:               
                self.run_fit_candidates()
                self.run_filter_spots()
                if self._chained['save_results']:
                    self.save_results(dir_localize, base_name)
        else:
            warnings.warn("Set all parameters and chained options before running.")