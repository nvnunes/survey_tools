# GIRMOS Asterism Catalog v01

This folder contains the first release of a catalog of asterisms optimized for GIRMOS AO correction within the Euclid Wide Survey footprint and between declinations of -20 to +60. All of the asterisms selected have 3 stars with R<16 lying within a 2 armin field of view. Additional cuts to eliminate stars that are too close together, fields that have too many stars and areas with galactic Av > 0.3 have been made.

## Contents

### `asterisms-GNAO-Optimal-v01.fits`
A FITS table containing the catalog of optimal asterisms. Each row corresponds to one asterism, composed of three stars selected from the Gaia DR3 catalog. The catalog includes positions, magnitudes, proper motions, and an estimated extinction value.

#### Columns:

| Column         | Type     | Description |
|----------------|----------|-------------|
| `id`           | int64    | Unique asterism identifier (level 14 HEALpix of center) |
| `ra`           | float64  | Right Ascension of the asterism center (degrees) |
| `dec`          | float64  | Declination of the asterism center (degrees) |
| `num_stars`    | int64    | Number of stars in the asterism |
| `star1_id`     | int64    | Gaia DR3 Source ID of star 1 |
| `star1_ra`     | float64  | RA of star 1 (degrees) |
| `star1_dec`    | float64  | Dec of star 1 (degrees) |
| `star1_pmra`   | float64  | Proper motion in RA of star 1 (mas/yr) |
| `star1_pmdec`  | float64  | Proper motion in Dec of star 1 (mas/yr) |
| `star1_pmepoch`| float64  | Reference epoch for star 1 |
| `star1_mag`    | float64  | R-band magnitude of star 1 |
| (same fields for `star2_*` and `star3_*`) |
| `Av`           | float32  | Estimated visual extinction (mag) at the asterism location |

Note coordinates in the Gaia DR3 catalog are given for a reference epoch of 2016. Proper motion has been added to the coordinates in the asterism catalog adjusting them to a reference epoch of 2028.

---

### `sample-targets.fits`
A sample galaxy catalog (5000 sources) randomly selected from the 3D-HST survey. This file serves as a test set for matching science targets with asterisms.

---

### `example-match-asterisms.ipynb`
A Python Jupyter notebook that demonstrates how to:
- Load the asterism and target catalogs
- Match science targets with suitable asterisms based on angular separation

---

For questions or feedback, contact: Nelson Nunes – nvnunes@yorku.ca
