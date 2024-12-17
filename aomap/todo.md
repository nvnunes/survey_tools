# Project Overview

Phase 1:
1. Create FITS map, store dummy values and plot map
1. Load stars from Gaia, save to inner FITS, count stars in outer pix, make map, optionally skip processing outer pix based on count (i.e. stellar density)
1. Find asterism in each outer pix, reduce number of asterisms, count asterisms left within an FOV of each inner pix, make map
1. Load dust map, optionally draw contours on map, optionally skip processing outer pix based on extinction
1. Load region covered by Euclid EWS, optionally draw outlines on map, optionally skip processing outer pix based on region
1. Load regions covered by important extra-galactic fields, optionally draw outlines on map, create table of average outer pix values over fields
1. Install Tip Top and use sims to develop improved AO Quality metric
1. Use AO Quality to select best asterism at each inner pix, record asterism stars, store AO Quality, make map

Phase 2:
- Develop Neural Network (NN) to compute AO performance for specific instrument based on asterism properties
- Estimate uncertainty of prediction
- Replace AO Quality with AO performance computed using NN

Phase 3:
- Setup for massive parallelism by running in the Cloud
- Optimize performance
- Produce full sky maps

Phase 4:
- Create NN for asterisms of 2 stars and 1 star
- Create NN for combinations of seeing and airmass (few steps each)
- Create NN for other instruments (maybe)
- Produce full sky maps for other combinations

Phase 5:
- Compare how different systems perform
- Use maps to recommend best approach for future high-z surveys
- Write paper describing code and making recommendations

# Phase 1 - Step 1

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

# Phase 1 - Step 2

1. TODO
