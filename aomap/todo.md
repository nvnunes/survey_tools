# Project Overview

DONE Phase 0:
1. DONE Create FITS map, store dummy values and plot map
1. DONE Load stars from Gaia and cache Gaia results
1. DONE Count stars in outer pix
1. DONE Make maps of star counts and stellar density
1. DONE Find promising areas for AO based on stellar density
1. DONE Optionally limit processing based on HEALpix level and pix

DONE Phase 1:
1. DONE Load dust map and compute extinction per inner pixel
1. DONE Optionally draw contours of extinction on map
1. DONE Load Zodiacal light model and study emission versus ecliptic latitude (https://cosmoglobe.github.io/zodipy/)
1. DONE Load outline of EWS and plot position of 3D-HST fields
1. DONE Optionally draw outlines of regions on map
1. DONE Plot wide spectroscopic surveys (Euclid, DESI) 
1. DONE Find patches of sky where:
    - DONE reasonably high stellar density
    - DONE reasonably low dust extinction (Av < 0.3)
    - DONE reasonably low zodiacal light emission (cut on ecliptic latitude)
1. DONE Update AO fields based on Av < 0.3

DONE Phase 2:
1. DONE Find n-star asterisms in each outer pix
1. DONE Reduce number of overlapping asterisms
1. DONE Count remaining asterisms within an FOV of each inner pix
1. DONE Make map of asterism counts per inner pix
1. DONE Compute percent area covered by inner pix with at least one asterism

Phase 3:
1. Install Tip Top and configure for reference AO system
1. Run few simulations to verify results make sense
1. Run AO systems over grid of NGS positions and magnitudes
1. Use sims to develop an improved first-order AO Quality metric
1. Use AO Quality to select best asterism at each inner pix
    - Store AO Quality in inner.fits
    - Store asterism star Gaia source-ids inner.fits
1. Make map of average AO Quality per inner pix

Phase 4:
- Develop Neural Network (NN) to compute AO performance for specific instrument based on asterism properties
- Estimate uncertainty of prediction
- Replace AO Quality with AO performance computed using NN

Phase 5:
- Setup for massive parallelism by running in the Cloud
- Optimize performance
- Produce full sky maps

Phase 6:
- Create NN for asterisms of 2 stars and 1 star
- Create NN for combinations of seeing and airmass (few steps each)
- Create NN for other instruments (maybe)
- Produce full sky maps for other combinations

Phase 7:
- Compare how different systems perform
- Use maps to recommend best approach for future high-z surveys
- Write paper describing code and making recommendations

# Phase 0 - Step 1

1. DONE configuration struct
    - DONE specify levels
    - DONE map name
    - DONE map location: e.g. "cache/maps" (share location?)
1. DONE create/open FITS files
    - DONE print if file already exists
    - DONE FITS images for outer pix with aggregate values (e.g. star count, average over inner pix values) using different file for each map (e.g. star count, asterism count after reducing, dust, AO quality)
1. DONE create/open state in struc saved to pickle file
    - DONE location: {map location}/{safe map-name}-{outer level}-{inner level}/state.pkl
    - DONE bool array for each processing step to indicate what is done (map_state.done_stars, map_state.done_dust, map_state.done_aosim, etc)
1. DONE outer loop over HEALpix
    - DONE print number of outer HEALpix complete / total
    - DONE skip if already done at outer level
    - DONE save updates to outer table
    - DONE save updates to state
    - DONE config item to limit looping (e.g. max outer pix processed at a time)
1. DONE plot maps
    - DONE plot map with celestial coordinates
    - DONE plot map with cartesian projection
    - DONE plot map with mollweide (or hammer or aitoff) projection
    - DONE plot maps with galactic coordinates
    - DONE store plot options in config
1. DONE parallelize outer loop
    - DONE read number of threads from configuration
1. DONE inner loop over HEALpix
    - DONE separate FITS files for each outer pix with table with all inner pix values (dust, aosim)
        - create/open FITS file: {folder}/{safe map-name}-{outer level}-{inner level}/{ra deg // 10 * 10}/{N or S}{dec deg // 10 * 10}/{outer pix}.fits
    - DONE store galaxy model
    - DONE store average inner pix value in outer pix
    - DONE save inner table
    - DONE config option to force reprocess at inner level
    - DONE compute aggregate values over inner pix and store in outer pix value
        - option 1: NORMAL aggregation done while computing inner values
            - computed while computing inner values (i.e. returned from processing code)
        - option 2: RELOAD for distributed processing where outer FITS is not built
            - read through files caching outer values and build outer FITS
        - option 3: RECALC for recalculating aggregate values without rebuilding inner values
            - open inner fits, recompute outer value, cache outer value, build outer FITS
    - DONE do all maps together when aggregating and looping over inner pix
1. DONE incremental saving of outer FITS
    - DONE support parallel processing by chunking work into batches and saving after every nth
    - DONE time how long each chunk takes and report
    - DONE if build, check if inner.fits is present and use it instead of rebuilding (for graceful recovery)
1. DONE add support for exclusions
    - DONE config option to skip outer pix based on galactic latitude (use masked arrays?)

# Phase 0 - Step 2

1. DONE Map of NGS Stellar Density
    - DONE Query GAIA for count of NGS-candidates in each inner pix and store it in inner.fits
    - DONE Cache query so that inner.fits can be rebuilt without having to re-query Gaia?
    - DONE Map of star count and stellar density
    - DONE Add units to plots
1. DONE Support multiple HEALpix map levels
    - DONE Config map_levels between inner_level and outer_level that also get generated all at same time for efficiency
    - DONE Option when plotting to specify map_level
1. DONE Count NGS stars
    - DONE Count NGS stars as defined in config
    - DONE Count inner pix with at least one NGS pix
    - DONE Map of NGS pix density
1. DONE Find promising areas for AO based on stellar density
    - DONE Cut on NGS pix density
    - DONE Custom map using raw outer pixel data
1. DONE Miscellaneous:
    - DONE Plot healpix boundaries and label healpix numbers at particular level
    - DONE Limit plotting to particular level and pix
    - DONE Limit building to particular level and pix
