import numpy as np
from localize_psf import rois
from typing import Optional
from numba import njit, prange

# cupy imports
cupy_available = True

try:
    import cupy as cp # type: ignore
except ImportError:
    cp = np
    cupy_available = False

def get_skewed_coords(sizes: list[int], 
                      image_params: Optional[dict],
                      pixel_size: Optional[float],
                      scan_step: Optional[float],
                      theta: Optional[float],) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Get laboratory coordinates (i.e. coverslip coordinates) for a lateral set 
    of oblique image planes

    Parameters
    ----------
    sizes : list
        {nimgs, ny_cam, nx_cam}. Number of images in scan, y camera pixels, 
        x camera pixels
    image_params : Optional[dict]
        {'pixel_size','scan_step','theta'}. Camera pixel size, 
        spacing between oblique planes, and oblique angle.
    pixel_size : Optional[float]
        Effective camera pixel size
    scan_step : Optional[float]
        Spacing between oblique planes
    theta : Optional[float]
        Oblique angle in degrees
    
    Returns
    -------
    z : np.ndarray
        Z coordinates, broadcastable to image shape
    y : np.ndarray
        Y coordinates, broadcastable to image shape
    x : np.ndarray
        X coordinates, broadcastable to image shape
    """

    nimgs, ny_cam, nx_cam = sizes

    if image_params is not None:
        pixel_size = image_params['pixel_size']
        scan_step = image_params['scan_step']
        theta = image_params['theta']

    x = pixel_size * np.arange(nx_cam)[None, None, :]
    y = scan_step * np.arange(nimgs)[:, None, None] + pixel_size\
        * np.cos(theta * np.pi/180) * np.arange(ny_cam)[None, :, None]
    z = pixel_size * np.sin(theta * np.pi/180) * np.arange(ny_cam)[None, :, None]

    return z,y,x


def get_nearest_pixel(centers: np.ndarray[float],
                      pixel_size: float,
                      scan_step: float,
                      theta: float) -> np.ndarray[int]:
    """
    Get nearest pixel indices for centers given in real coordinates

    NOTE: currently not guaranteed to give exact closest i1 point, but only an approximation

    Parameters
    ----------
    centers : np.ndarray
         array of size ncenters x ndims = [[c0_z, c0_y, c0_x], [c1_z, ...], ....]
    pixel_size
    scan_step
    theta

    Returns
    -------
    pixel_indices : numpy array

    """

    theta_rad = theta * np.pi / 180

    i2 = np.rint(centers[..., 2] / pixel_size)
    i1 = np.rint(centers[..., 0] / (pixel_size * np.sin(theta_rad)))
    i0 = np.rint((centers[..., 1] - pixel_size * np.cos(theta_rad) * i1) / scan_step)

    return np.stack((i0, i1, i2), axis=1).astype(int)


@njit(parallel=True)
def prepare_rois(image: np.ndarray[float],
                 coords: tuple[np.ndarray, np.ndarray, np.ndarray],
                 rois: np.ndarray[int],
                 centers: np.ndarray[float],
                 max_seps: tuple[float, float],
                 nmax_roi_size: int) -> (np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray[int]):
    """
    Cut out ROI's from an image and estimate initial parameters prior to fitting centers
    (1) cut out parallelipiped shaped ROI's
    (2) cut out portions of the parallelipiped which are far away from the center
    (3) estimate fit parameters from data

    :param image: 3D array
    :param coords: (z, y, x)
    :param rois: nroi x 2*ndim array
    :param centers: nroi x 3 array of (cz, cy, cx)
    :param max_seps: (max_z, max_xy)
    :return:
        img_roi_out_all: list of masked arrays
        (zrois, yrois, xrois): list of masked x-coordinates
        roi_sizes:
    """

    z, y, x = coords

    nrois = len(rois)

    img_rois = np.ones((nrois, nmax_roi_size)) * np.nan
    xrois = np.zeros((nrois, nmax_roi_size)) * np.nan
    yrois = np.zeros((nrois, nmax_roi_size)) * np.nan
    zrois = np.zeros((nrois, nmax_roi_size)) * np.nan
    sizes = np.zeros(len(rois))

    for rr in prange(nrois):
        roi = rois[rr]
        center = centers[rr]

        # cut initial ROIs
        img_roi = image[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
        x_roi = x[:, :, roi[4]:roi[5]]
        y_roi = y[roi[0]:roi[1], roi[2]:roi[3], :]
        z_roi = z[:, roi[2]:roi[3], :]

        # mask ROIs
        mask = get_roi_mask(center, max_seps, (z_roi, y_roi, x_roi))
        nmask = np.sum(mask)
        sizes[rr] = nmask

        # # construct masked arrays (previously done with broadcasting + logical indexing)
        n0 = roi[1] - roi[0]
        n1 = roi[3] - roi[2]
        n2 = roi[5] - roi[4]

        counter = 0
        for ii in range(n0):
            for jj in range(n1):
                for kk in range(n2):
                    if mask[ii, jj, kk]:
                        xrois[rr, counter] = x_roi[0, 0, kk]
                        yrois[rr, counter] = y_roi[ii, jj, 0]
                        zrois[rr, counter] = z_roi[0, jj, 0]
                        img_rois[rr, counter] = img_roi[ii, jj, kk]

                        counter += 1

    return img_rois, (zrois, yrois, xrois), sizes


def get_skewed_roi_size(sizes : list[float],
                        pixel_size : float, 
                        scan_step : float,
                        theta : float,
                        ensure_odd : bool =True) -> list[int]:
    """
    Get ROI size in oblique data that includes sufficient xy and z points

    Parameters
    ----------
    sizes : list[float] 
        [z-size, y-size, x-size] in same units as camera_pixel and scan_step
    pixel_size : float
        Effective camera pixel size
    scan_step : float
        Spacing between oblique planes
    theta : float
        Oblique angle in degrees
    ensure_odd : bool
        Ensure ROI is odd sized

    Returns
    -------
    [n0, n1, n2] : list[int]
        Integer size of ROI in skewed coordinates
    """

    # x-size determines n2 size
    n2 = int(np.ceil(sizes[2] / pixel_size))

    # z-size determines n1
    n1 = int(np.ceil(sizes[0] / pixel_size / np.sin(theta * np.pi/180)))

    # set so that @ top and bottom z-points, ROI includes the full y-size
    n0 = int(np.ceil((0.5 * (n1 + 1)) * pixel_size\
         * np.cos(theta * np.pi/180) + sizes[1]) / scan_step)

    if ensure_odd:
        if np.mod(n2, 2) == 0:
            n2 += 1

        if np.mod(n1, 2) == 0:
            n1 += 1

        if np.mod(n0, 2) == 0:
            n0 += 1

    return [n0, n1, n2]

def get_filter_kernel_skewed(sigmas : list[float],
                             pixel_size : float, 
                             scan_step : float, 
                             theta : float, 
                             sigma_cutoff : float = 2.) -> np.ndarray:
    """
    Get gaussian filter convolution kernel in skewed coordinates

    Parameters
    ----------
    sigmas : list[float]
        (sz, sy, sx) in the same units as camera_pixel and scan_step (typically microns)
    pixel_size : float
        Effective camera pixel size
    scan_step : float
        Spacing between oblique planes 
    theta : float
        Oblique angle in degrees
    sigma_cutoff : float
        Number of standard deviations to include in the filter. This parameter 
        determines the fitler size.
    
    Returns
    -------
    kernel : np.ndarray
        Gaussian convolution kernel
    """

    # normalize everything to camera pixel size
    sigma_x_pix = sigmas[2] / pixel_size
    sigma_y_pix = sigmas[2] / pixel_size
    sigma_z_pix = sigmas[0] / pixel_size 
    nk_x = 2 * int(np.round(sigma_x_pix * sigma_cutoff)) + 1
    nk_y = 2 * int(np.round(sigma_y_pix * sigma_cutoff)) + 1
    nk_z = 2 * int(np.round(sigma_z_pix * sigma_cutoff)) + 1

    # determine how large the oblique geometry ROI needs to be to fit the desired filter
    roi_sizes = get_skewed_roi_size([nk_z, nk_y, nk_x],  
                                    1, 
                                    scan_step / pixel_size, 
                                    theta, 
                                    ensure_odd=True)

    # get coordinates to evaluate kernel at
    zk, yk, xk = get_skewed_coords(roi_sizes,
                                   image_params = None,
                                   pixel_size = 1, 
                                   scan_step = scan_step / pixel_size, 
                                   theta = theta)
    xk = xk - np.mean(xk)
    yk = yk - np.mean(yk)
    zk = zk - np.mean(zk)

    kernel = np.exp(-xk ** 2 / 2 / sigma_x_pix ** 2 -\
                     yk ** 2 / 2 / sigma_y_pix ** 2 -\
                     zk ** 2 / 2 / sigma_z_pix ** 2)
    kernel = kernel / np.sum(kernel)

    return kernel.astype(np.float32)

def get_skewed_footprint(min_sep_allowed : list[float],
                         pixel_size : float, 
                         scan_step : float,
                         theta : float) -> np.ndarray:
    """
    Get footprint for maximum filter in skewed coordinates

    Parameters
    ----------
    min_sep_allowed : list[float]
        (dz, dy, dx). Minimum separation allowed in physical units (typically microns)
    pixel_size : float
        Effective camera pixel size
    scan_step : float
        Spacing between oblique planes 
    theta : float 
        Angle in degrees

    Returns
    -------
    footprint : np.ndarray
        Footprint of filter
    """

    footprint_roi_size = get_skewed_roi_size(min_sep_allowed, 
                                             pixel_size, 
                                             scan_step, 
                                             theta, 
                                             ensure_odd=True)
    footprint_form = np.ones(footprint_roi_size, dtype=bool)
    zf, yf, xf = get_skewed_coords(footprint_form.shape, 
                                   image_params = None,
                                   pixel_size = pixel_size, 
                                   scan_step = scan_step, 
                                   theta = theta)
    xf = xf - xf.mean()
    yf = yf - yf.mean()
    zf = zf - zf.mean()
    footprint_mask = get_roi_mask((0, 0, 0), min_sep_allowed, (zf, yf, xf))
    footprint_mask[np.isnan(footprint_mask)] = 0
    footprint_mask = footprint_mask.astype(bool)

    return footprint_form * footprint_mask

def get_skewed_roi(center : list[float], 
                   image : np.ndarray, 
                   coords : tuple[np.ndarray,np.ndarray,np.ndarray],
                   sizes : list[int]
                ) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Given a center value (not necessarily aligned to the coordinates), find the 
    closest region of interest (ROI) centered around that point.

    Parameters
    ----------
    center : float  
        [cz, cy, cx] in same units as x, y, z
    image : np.ndarray
        Image to cut out ROI from
    coords : tuple[np.ndarray,np.ndarray,np.ndarray] 
        A tuple of coordinates (z, y, x), where z, y, and x are broadcastable 
        to the same shape as imgs. These coordinates are supplied as
        produced by get_skewed_coords()
    sizes : list[int] 
        [nz, ny, nx], the size of the desired ROI in number of pixels 
        along the skewed coordinate directions. This should be calculated 
        with the help of get_skewed_roi_size().
    
    Returns
    -------
    (roi,img_roi,z_roi,y_roi,x_roi) : tuple[np.ndarray]
        ROI, data from ROI, z ROI coords, y ROI coords, x ROI coords outputs
    """

    z, y, x = coords
    shape = image.shape

    i2 = np.argmin(np.abs(x.ravel() - center[2]))
    i0, i1, _ = np.unravel_index(np.argmin((y - center[1]) ** 2 + (z - center[0]) ** 2), y.shape)
    roi = rois.get_centered_rois([i0, i1, i2], sizes, min_vals=[0, 0, 0], max_vals=np.array(shape))[0]

    img_roi = image[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    # slice x,y,z accounting for fact only broadcastable to same shape as img_roi
    x_roi = x[:, :, roi[4]:roi[5]]
    y_roi = y[roi[0]:roi[1], roi[2]:roi[3], :]
    z_roi = z[:, roi[2]:roi[3], :]
    # broadcast to make coordinate arrays same shape as img_roi
    z_roi, y_roi, x_roi = np.broadcast_arrays(z_roi, y_roi, x_roi)

    return roi, img_roi, x_roi, y_roi, z_roi

@njit
def get_roi_mask(center : list[float],
                 max_seps : list[float],
                 coords: tuple[np.ndarray,np.ndarray,np.ndarray]) -> np.ndarray:
    """
    Get mask to exclude points in the ROI that are far from the center. We do not want 
    to include regions at the edges of the trapezoidal ROI in processing.

    Parameters
    ----------
    center : list[float] 
        [cz, cy, cx] in same units as x, y, z
    max_seps : list[float] 
        (dz, dxy) in same units as x, y, z
    coords : tuple[np.ndarray,np.ndarray,np.ndarray] 
        (z, y, x) sizes must be broadcastable

    Returns
    -------
    mask : np.ndarray
         Same size as roi, 1 where point is allowed and nan otherwise
    """
    z_roi, y_roi, x_roi = coords

    # roi is parallelogram, 
    # so still want to cut out points which are too far from center
    too_far_xy = np.sqrt((x_roi - center[2]) ** 2 + (y_roi - center[1]) ** 2) > max_seps[1]
    too_far_z = np.abs(z_roi - center[0]) > max_seps[0]
    too_far = np.logical_or(too_far_xy, too_far_z)

    # mask[too_far] = False # previously was broadcasting to get mask ... but don't need to do that ...
    mask = np.logical_not(too_far)

    return mask


def get_trapezoid_zbound(cy : float, 
                         coords : tuple[np.ndarray]) -> tuple[np.ndarray]:
    """
    Find z-range of trapezoid for given center position cy

    Parameters
    ----------
    cy : float
        Center y position
    coords : tuple[np.ndarray,np.ndarray,np.ndarray] 
        (z, y, x) sizes must be broadcastable
    
    Returns
    -------
    zmax : np.ndarray
        Maximum z coordinates
    zmin : np.ndarray
        Minimum z coordinates
    """
    cy = np.array(cy)

    z, y, x = coords
    slope = (z[:, -1, 0] - z[:, 0, 0]) / (y[0, -1, 0] - y[0, 0, 0])

    # zmax
    zmax = np.zeros(cy.shape)
    cy_greater = cy > y[0, -1]
    zmax[cy_greater] = z.max()
    zmax[np.logical_not(cy_greater)] = slope * (cy[np.logical_not(cy_greater)] - y[0, 0])
    # if cy > y[0, -1]:
    #     zmax = z.max()
    # else:
    #     zmax = slope * (cy - y[0, 0])

    # zmin
    zmin = np.zeros(cy.shape)
    cy_less = cy < y[-1, 0]
    zmin[cy_less] = z.min()
    zmin[np.logical_not(cy_less)] = slope * (cy[np.logical_not(cy_less)] - y[-1, 0])

    # if cy < y[-1, 0]:
    #     zmin = z.min()
    # else:
    #     zmin = slope * (cy - y[-1, 0])

    return zmax, zmin


def get_trapezoid_ybound(cz : float,
                         coords : tuple[np.ndarray]) -> tuple[np.ndarray]:
    """
    Find y-range of trapezoid for given center position cz

    Parameters
    ----------
    cz : float
        Center z position
    coords : tuple[np.ndarray,np.ndarray,np.ndarray] 
        (z, y, x) sizes must be broadcastable
    
    Returns
    -------
    ymax : np.ndarray
        Maximum y coordinates
    ymin : np.ndarray
        Minimum y coordinates
    """

    cz = np.array(cz)

    z, y, x = coords
    slope = (z[:, -1, 0] - z[:, 0, 0]) / (y[0, -1, 0] - y[0, 0, 0])

    ymin = cz / slope
    ymax = cz / slope + y[-1, 0]

    return ymax, ymin

def get_skewed_coords_deriv(sizes : list[int,int,int],
                            pixel_size : float,
                            scan_step: float, 
                            theta : float) -> tuple[tuple[np.ndarray],
                                                    tuple[np.ndarray],
                                                    tuple[np.ndarray]]:
    """
    Calculate derivative of coordinates with respect to theta

    Parameters
    ----------
    sizes : list[int,int,int]
        number of images, y camera pixels, x camera pixels.
    pixel_size : float
        Effective camera pixel size
    scan_step : float
        Spacing between oblique planes 
    theta : float 
        Angle in degrees

    Returns
    -------
    [dxdt, dydt, dzdt], [dxds, dyds, dzds], [dxdc, dydc, dzdc]
        Derivatives with respect to theta
    """

    nimgs, ny_cam, nx_cam = sizes
    dxdt = 0 * pixel_size * np.arange(nx_cam)[None, None, :]
    dydt = 0 * scan_step * np.arange(nimgs)[:, None, None] - pixel_size * np.sin(theta * np.pi/180) * np.arange(ny_cam)[None, :, None]
    dzdt = scan_step * np.cos(theta * np.pi/180) * np.arange(ny_cam)[None, :, None]

    dxds = 0 * pixel_size * np.arange(nx_cam)[None, None, :]
    dyds = np.arange(nimgs)[:, None, None] + 0 * pixel_size * np.cos(theta * np.pi/180) * np.arange(ny_cam)[None, :, None]
    dzds = 0 * pixel_size * np.sin(theta * np.pi/180) * np.arange(ny_cam)[None, :, None]

    dxdc = np.arange(nx_cam)[None, None, :]
    dydc = 0 * scan_step * np.arange(nimgs)[:, None, None] + np.cos(theta * np.pi/180) * np.arange(ny_cam)[None, :, None]
    dzdc = np.sin(theta * np.pi/180) * np.arange(ny_cam)[None, :, None]

    return [dxdt, dydt, dzdt], [dxds, dyds, dzds], [dxdc, dydc, dzdc]


def lab2cam(x, y, z, theta):
    """
    Convert xyz coordinates to camera coordinates sytem, x', y', and stage position.
    :param x:
    :param y:
    :param z:
    :param theta:
    :return xp:
    :return yp: yp coordinate
    :return stage_pos: distance of leading edge of camera frame from the y-axis
    """
    xp = x
    stage_pos = y - z / np.tan(theta)
    yp = (y - stage_pos) / np.cos(theta)
    return xp, yp, stage_pos


def xy_lab2cam(x, y, stage_pos, theta):
    """
    Convert xy coordinates to x', y' coordinates at a certain stage position
    
    :param x: 
    :param y: 
    :param stage_pos: 
    :param theta: 
    :return: 
    """
    xp = x
    yp = (y - stage_pos) / np.cos(theta)

    return xp, yp


def nearest_pt_line(pt, slope, pt_line):
    """
    Get shortest distance between a point and a line.
    :param pt: (xo, yo), point of interest
    :param slope: slope of line
    :param pt_line: (xl, yl), point the line passes through
    :return pt: (x_near, y_near), nearest point on line
    :return d: shortest distance from point to line
    """
    xo, yo = pt
    xl, yl = pt_line
    b = yl - slope * xl

    x_int = (xo + slope * (yo - b)) / (slope ** 2 + 1)
    y_int = slope * x_int + b
    d = np.sqrt((xo - x_int) ** 2 + (yo - y_int) ** 2)

    return (x_int, y_int), d


def point_in_trapezoid(pts: np.ndarray,
                       coords: tuple[np.ndarray]) -> np.ndarray:
    """
    Test if a point is in the trapzoidal region described by x,y,z
    :param pts: np.array([[cz0, cy0, cx0], [cz1, cy1, cx1], ...[czn, cyn, cxn]])
    :param coords: (z, y, x)
    :return:
    """

    z, y, x = coords
    if pts.ndim == 1:
        pts = pts[None, :]

    # get theta
    dc = x[0, 0, 1] - x[0, 0, 0]
    dz = z[0, 1, 0] - z[0, 0, 0]
    theta = np.arcsin(dz / dc)

    # get edges
    zstart = z.min()
    ystart = y[0, 0, 0]
    yend = y[-1, 0, 0]

    # need to round near machine precision, or can get strange results when points right on boundary
    decimals = 6
    not_in_region_x = np.logical_or(np.round(pts[:, 2], decimals) < np.round(x.min(), decimals),
                                    np.round(pts[:, 2], decimals) > np.round(x.max(), decimals))
    not_in_region_z = np.logical_or(np.round(pts[:, 0], decimals) < np.round(z.min(), decimals),
                                    np.round(pts[:, 0], decimals) > np.round(z.max(), decimals))
    # tilted lines describing ends
    not_in_region_yz = np.logical_or(np.round(pts[:, 0] - zstart, decimals) > np.round((pts[:, 1] - ystart) * np.tan(theta), decimals),
                                     np.round(pts[:, 0] - zstart, decimals) < np.round((pts[:, 1] - yend) * np.tan(theta), decimals))

    in_region = np.logical_not(np.logical_or(not_in_region_yz, np.logical_or(not_in_region_x, not_in_region_z)))

    return in_region
