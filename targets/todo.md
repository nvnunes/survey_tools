# Project Overview

### Phase 1
1. DONE Specify single target providing target details including line(s) of interest
1. DONE Export table with gathered information
1. DONE Compute sky information
    - DONE plot sky transmission near target wavelength(s) of interest
    - DONE plot skylines near target wavelength(s) of interest
    - DONE reject lines based on sky and instrument wavelengths
    - DONE illustrate effect of redshift uncertainty
1. DONE Find stars near the target compatible with AO system
    - DONE Select best stars for AO system [closest, brightest]
    - DONE Display list of stars near target
    - DONE Use cached Gaia data if available, otherwise query Gaia directly
1. DONE Append information from catalogs
    - DONE Specify catalog and ID for target to pull information
    - DONE Load ra/dec/z/flux_radius from catalog
    - DONE Load line fluxes
    - DONE Add support for Ha estimated flux
    - DONE Add support for NIIa/b from Ha flux
1. DONE Display catalog information for comparison
    - DONE IDs
    - DONE z_specs
    - DONE Line Fluxes
    - DONE SPP values
1. DONE Load 1D spectra from catalogs
    - DONE ZCB
    - DONE 3DHST
    - DONE VUDS
    - DONE DEIMOS
    - DONE MOSDEF
    - DONE FMOS
    - DONE C3R2
    - DONE LEGAC
    - DONE DESI
1. DONE Plot cut-outs and finder-chart for selected target
    - DONE Based on matched catalog imaging
    - DONE Additional downloaded imaging
1. DONE Plan IFU observation
    - DONE Configure IFU properties
    - DONE Compute segmentation map of source
    - DONE Plot IFU showing spaxels
1. Other:
    - Add support for loading z_unc from catalog
    - Add support for loading dispersion/FWHM from catalog
    - Add support for mapping different column names
    - Match against catalogs by RA/Dec to identify source when no id
    - Find optimal IFU position and position angle
    - Add support for plotting additional 2D spectra
    - Add support for plotting KMOS3D data cubes
    - Use segmentation map and pixel scale to estimate area for SB instead of using flux_radius (keep both way as current way better for filtering catalog to find sources)
    - Weight sky transmission across gaussian profile
    - Take z_unc into account when rejecting skylines (estimate probability of collision, cut on probability)

### Phase 2
1. Select targets from catalog
