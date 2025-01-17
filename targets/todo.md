# Project Overview

### Miscellaneous
1. Weight sky transmission across gaussian profile
1. Take z_unc into account when rejecting skylines (estimate probability of collision, cut on probability)
1. Use segmentation map and pixel scale to estimate area for SB instead of using flux_radius (keep both way as current way better for filtering catalog to find sources)

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
1. Append information from catalogs
    - DONE Specify catalog and ID for target to pull information
    - DONE Load ra/dec/z/flux_radius from catalog
    - DONE Load line fluxes
    - DONE Add support for Ha estimated flux
    - DONE Add support for NIIa/b from Ha flux
    - Add support for different column name cases!
    - Add support for z_unc
    - Add support for dispersion/FWHM
    - Add support for Field and Filter catalog parameters
    - Match against catalogs to identify source if possible when no id
1. DONE Display catalog information for comparison
    - DONE IDs
    - DONE z_specs
    - DONE Line Fluxes
    - DONE SPP values
1. Load spectra from catalogs
    - DONE ZCB
    - 3DHST
    - VUDS
    - Casey DSFG
    - DONE DEIMOS
    - MOSDEF
    - DONE FMOS
    - KMOS3D
    - C3R2
    - DONE LEGAC
    - DONE DESI
1. Plot cut-outs and finder-chart for selected target
    - Based on matched catalog imaging
    - Additional downloaded imaging
    - Web-service for COSMOS
1. Plan IFU observation
    - Configure IFU properties
    - Compute segmentation map of source (throw error if too many)
    - Find best orientation (position angle) for IFU
    - Plot IFU showing spaxels

### Phase 2
1. Select targets from catalog
