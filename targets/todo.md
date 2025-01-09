# Project Overview

### Miscellaneous
1. Weight sky transmission across gaussian profile
1. Take dz into account when rejecting skylines

### Phase 1
1. DONE Specify single target providing target details including line(s) of interest
1. DONE Export table with gathered information
1. DONE Compute sky information
    - DONE plot sky transmission near target wavelength(s) of interest
    - DONE plot skylines near target wavelength(s) of interest
    - DONE reject lines based on sky and instrument wavelengths
    - DONE illustrate effect of redshift uncertainty
1. Find stars near the target compatible with AO system
    - DONE Select best stars for AO system [closest, brightest]
    - DONE Display list of stars near target
    - Use cached Gaia data if available, otherwise query Gaia directly
1. Specify catalog and ID for target to pull information
1. Append target information from catalogs
1. Match against catalogs to identify source if possible
1. Plot cut-outs of target
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
