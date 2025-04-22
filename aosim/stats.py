#!/usr/bin/env python3
# pylint: disable=too-many-lines,line-too-long
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods,too-many-public-methods,too-many-instance-attributes,attribute-defined-outside-init
# pylint: disable=invalid-name,too-many-arguments,too-many-locals,too-many-statements,too-many-branches

from scipy.interpolate import interp1d
from p3.aoSystem.FourierUtils import otf2psf, telescopeOtf, find_contour_points, fwhm_1d

try:
    gpuEnabled = True
    import cupy as np
    import cupyx.scipy.ndimage as scnd
    import numpy as nnp
except: # pylint: disable=bare-except
    gpuEnabled = False
    import numpy as np
    import scipy.ndimage as scnd
    nnp = np

def cpuArray(v):
    if isinstance(v,nnp.ndarray) or isinstance(v, nnp.float64):
        return v
    else:
        return v.get()

def get_stats(simulation):
    psfs = np.array([img.sampling for img in simulation.results])
    psfs_centered = _center_psfs(psfs, normalize=True)

    SR = _get_strehl(psfs_centered, simulation.fao.ao.tel.pupil, simulation.wvl[0], simulation.tel_radius, simulation.psInMas, assume_centered=True, assume_normalized=True)
    FWHM = _get_FWHM(psfs_centered, simulation.psInMas, assume_centered=True)
    EE = _get_ensquared_energy_at_radius(psfs_centered, simulation.eeRadiusInMas, simulation.psInMas, assume_centered=True)

    return cpuArray(psfs), cpuArray(SR), cpuArray(FWHM), cpuArray(EE)

def get_stats_matlab(psfs, tel_diameter, tel_pupil, wavelength, pixel_scale, ee_size):
    psfs = np.array(psfs)
    psfs_centered = _center_psfs(psfs, normalize=True)

    SR = _get_strehl(psfs_centered, np.array(tel_pupil), wavelength, tel_diameter/2, pixel_scale, assume_centered=True, assume_normalized=True)
    FWHM = _get_FWHM(psfs_centered, pixel_scale, assume_centered=True)
    EE = _get_ensquared_energy_at_radius(psfs_centered, ee_size/2, pixel_scale, assume_centered=True)

    return cpuArray(SR), cpuArray(FWHM), cpuArray(EE)

def _center_psfs(psfs, normalize=False):
    psfs = np.atleast_3d(psfs)
    _, nY, nX = psfs.shape

    # Compute peak positions for all PSFs
    peak_indices = np.array([np.unravel_index(psf.argmax(), psf.shape) for psf in psfs])
    center = np.array([nY // 2, nX // 2])
    shifts = center - peak_indices

    # Shift each PSF so its peak is centered
    psfs_centered = np.empty_like(psfs)
    for i, shift_xy in enumerate(shifts):
        psfs_centered[i] = scnd.shift(psfs[i], shift=shift_xy, order=1, mode='constant')

    # Normalize PSFs
    if normalize:
        psfs_centered = np.clip(psfs_centered, 0, None)
        psfs_centered /= psfs_centered.sum(axis=(1, 2), keepdims=True)

    return psfs_centered

def _get_strehl(psfs, pupil, wavelength, tel_radius, pixel_scale, assume_centered=False, assume_normalized=False):
    _, Ny, Nx = psfs.shape

    rad2mas  = 3600 * 180 * 1000 / np.pi
    samp     = wavelength * rad2mas / (pixel_scale*2*tel_radius)
    otfDL    = telescopeOtf(pupil, samp)
    psfDL    = otf2psf(otfDL, psfInOnePix=True)
    psfDL.clip(0, None)
    psfDL   /= psfDL.sum()
    maxPSFDL = psfDL.max()

    if not assume_normalized:
        psfs /= psfs.sum(axis=(1, 2), keepdims=True)

    if assume_centered:
        cy, cx = Ny // 2, Nx // 2
        maxPSF = psfs[:, cy, cx]
    else:
        maxPSF = psfs.max(axis=(1, 2))

    SR = maxPSF/maxPSFDL

    return SR

def _get_FWHM_from_psf(psf, pixel_scale, assume_centered=False):
    Ny, Nx = psf.shape
    peak_val = nnp.max(psf)
    min_val = nnp.min(psf)

    # Check if not enough points below half height
    if min_val >= 0.5*peak_val:
        fwhmX = nnp.sqrt(2)*Nx*pixel_scale
        fwhmY = nnp.sqrt(2)*Ny*pixel_scale
    else:
        use_cutting = False

        # initial guess of PSF center
        if assume_centered:
            y_max, x_max = Ny//2, Nx//2
        else:
            y_max, x_max = nnp.unravel_index(nnp.argmax(psf), psf.shape) # pylint: disable=unbalanced-tuple-unpacking

        # cutting method used first to check if FWHM is too large for other methods
        # X and Y profiles passing through the max
        profile_x = psf[y_max, :]
        profile_y = psf[:, x_max]

        # X and Y FWHM
        fwhmXcut = fwhm_1d(profile_x)
        fwhmYcut = fwhm_1d(profile_y)

        # Check if FWHM is too large, falling back to cutting method
        if fwhmXcut >= profile_x.size-1 or fwhmYcut >= profile_y.size-1:
            use_cutting = True

        # Check if FWHM is too small, falling back to cutting method
        if fwhmXcut == 1 and fwhmYcut == 1:
            use_cutting = True

        if not use_cutting:
            # Find contour points at half maximum
            contour_points = find_contour_points(psf, peak_val/2)

            # not enough contour points found, falling back to cutting method
            if len(contour_points) < 3:  # Need at least 3 points for meaningful analysis
                use_cutting = True

        if use_cutting:
            fwhmX = fwhmXcut * pixel_scale
            fwhmY = fwhmYcut * pixel_scale
        else:
            xC = contour_points[:, 0]
            yC = contour_points[:, 1]

            # Centering the ellipse
            mx = nnp.array([xC.max(), yC.max()])
            mn = nnp.array([xC.min(), yC.min()])
            cent = (mx + mn)/2
            wx = xC - cent[0]
            wy = yC - cent[1]

            # Get the module
            wr = nnp.hypot(wx, wy)*pixel_scale

            # Getting the FWHM
            fwhmX = 2*wr.max()
            fwhmY = 2*wr.min()

    return 0.5 * (fwhmX+fwhmY)

def _get_FWHM(psfs, pixel_scale, assume_centered=False):
    psfs = cpuArray(psfs)
    nPSF = psfs.shape[0]

    FWHM = np.zeros((nPSF))
    for i in range(psfs.shape[0]):
        psf = psfs[i,:,:]
        FWHM[i] = _get_FWHM_from_psf(psf, pixel_scale, assume_centered)

    return FWHM

def _get_ensquared_energy_at_radius(psfs, ee_radius, pixel_scale, assume_centered=False):
    EE = cpuArray(_get_ensquared_energy(psfs, assume_centered))
    rr = nnp.arange(1, EE.shape[1]*2, 2) * pixel_scale * 0.5

    EEatRadius = np.empty(psfs.shape[0])
    for i, ee in enumerate(EE):
        EEatRadius[i] = interp1d(rr, ee, kind='cubic', bounds_error=False)(ee_radius)

    return EEatRadius

def _get_ensquared_energy(psfs, assume_centered=False):
    psfs = np.atleast_3d(psfs)
    nPSF, nY, nX = psfs.shape

    if not assume_centered:
        psfs = _center_psfs(psfs)

    center = np.array([nY // 2, nX // 2])
    max_radius = min(nY, nX)//2

    radii = np.arange(max_radius + 1)
    EE = np.zeros((nPSF, len(radii)))
    for r in radii:
        y0, y1 = center[0] - r, center[0] + r + 1
        x0, x1 = center[1] - r, center[1] + r + 1
        EE[:, r] = psfs[:, y0:y1, x0:x1].sum(axis=(1, 2))

    return EE
