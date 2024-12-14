==========================
    NOTE ON DATA FILES
==========================

This package does not include data files required by some of the tools due to
potential licensing restrictions. The necessary files can be obtained from the
locations specified below.

-------------
Mauna Kea Sky
-------------

Sources:
 - https://www.gemini.edu/observing/telescopes-and-sites/sites#IRSky
 - https://www.gemini.edu/observing/telescopes-and-sites/sites#Transmission

Files:
 - data/sky/mk_skybg_zm_10_10_ph.dat
 - data/sky/mk_skybg_zm_10_15_ph.dat
 - data/sky/mk_skybg_zm_10_20_ph.dat
 - data/sky/mk_trans_zm_10_10.dat
 - data/sky/mk_trans_zm_10_15.dat
 - data/sky/mk_trans_zm_10_20.dat

-----------------
Cerro Paranal Sky
-----------------

Sources:
 - https://www.eso.org/sci/facilities/eelt/science/drm/tech_data/background
 - https://www.eso.org/sci/facilities/eelt/science/drm/tech_data/data/atm_abs/

Files:
 - data/sky/paranal_skybg_airm1.00_wav00.4-03.0.dat
 - data/sky/paranal_skybg_airm1.15_wav00.4-03.0.dat
 - data/sky/paranal_skybg_airm1.50_wav00.4-03.0.dat
 - data/sky/paranal_skybg_airm2.00_wav00.4-03.0.dat
 - data/sky/paranal_trans_airm1.00_wav00.4-03.0.dat
 - data/sky/paranal_trans_airm1.15_wav00.4-03.0.dat
 - data/sky/paranal_trans_airm1.50_wav00.4-03.0.dat
 - data/sky/paranal_trans_airm2.00_wav00.4-03.0.dat

-----------------------
UltraVISTA Catalog v4.1
-----------------------

References: 
 - A Public Ks-Selected Catalog in the COSMOS/UltraVISTA Field: Photometry,
   Photometric Redshifts and Stellar Population Parameters. Muzzin, A.,
   Marchesini, D., Stefanon, M., et al. (2013), ApJ, 206, 8.
   doi:10.1088/0067-0049/206/1/8

Sources:
 - https://local.strw.leidenuniv.nl/galaxyevolution/ULTRAVISTA/Ultravista/K-selected.html

Files:
 - data/catalogs/UVISTA/UVISTA_final_v4.1.cat
 - data/catalogs/UVISTA/UVISTA_final_v4.1.zout
 - data/catalogs/UVISTA/UVISTA_final_BC03_v4.1.fout

----------------------
UltraVISTA Imaging DR5
----------------------

Description:
 - https://eso.org/rm/api/v1/public/releaseDescriptions/221

Sources:
 - https://archive.eso.org/scienceportal/home?data_collection=UltraVISTA&publ_date=2023-05-03
   TODO: Update to DR6 (https://archive.eso.org/scienceportal/home?data_collection=UltraVISTA&publ_date=2024-06-27)

Files:
 - data/images/UVISTA/ADP.2023-05-02T10_57_31.149.fits
 - data/images/UVISTA/ADP.2023-05-02T10_57_31.152.fits
 - data/images/UVISTA/ADP.2023-05-02T10_57_31.155.fits

----------------------
3D-HST v4.1/v4.1.5/4.2
----------------------

References:
 - 3D-HST WFC3-selected Photometric Catalogs in the Five CANDELS/3D-HST Fields:
   Photometry, Photometric Redshifts, and Stellar Masses. Skelton, R. E., Whitaker,
   K. E., Momcheva, I. G., et al. 2014, ApJS, 214, 24. 
   doi:10.1088/0067-0049/214/2/24

 - The 3D-HST Survey: Hubble Space Telescope WFC3/G141 Grism Spectra, Redshifts,
   and Emission Line Measurements for ~ 100,000 Galaxies. Momcheva, I. G., Brammer,
   G. B., van Dokkum, P. G., et al. 2016, ApJS, 225, 27.
   doi:10.3847/0067-0049/225/2/27

Sources:
 - https://archive.stsci.edu/prepds/3d-hst/

Files:
 - data/catalogs/3D-HST/aegis_3dhst.v4.1.cats/Catalog/aegis_3dhst.v4.1.cat.FITS
 - data/catalogs/3D-HST/aegis_3dhst.v4.1.cats/Fast/aegis_3dhst.v4.1.fout.FITS
 - data/catalogs/3D-HST/aegis_3dhst_v4.1.5_catalogs/aegis_3dhst.v4.1.5.zfit.linematched.fits
 - data/catalogs/3D-HST/aegis_3dhst_v4.1.5_catalogs/aegis_3dhst.v4.1.5.linefit.linematched.fits
 - data/images/3D-HST/aegis_3dhst.v4.0.F125W_orig_sci.fits
 - data/images/3D-HST/aegis_3dhst.v4.0.F140W_orig_sci.fits
 - data/images/3D-HST/aegis_3dhst.v4.0.F160W_orig_sci.fits

 - data/catalogs/3D-HST/cosmos_3dhst.v4.1.cats/Catalog/cosmos_3dhst.v4.1.cat.FITS
 - data/catalogs/3D-HST/cosmos_3dhst.v4.1.cats/Fast/cosmos_3dhst.v4.1.fout.FITS"
 - data/catalogs/3D-HST/cosmos_3dhst_v4.1.5_catalogs/cosmos_3dhst.v4.1.5.zfit.linematched.fits
 - data/catalogs/3D-HST/cosmos_3dhst_v4.1.5_catalogs/cosmos_3dhst.v4.1.5.linefit.linematched.fits
 - data/images/3D-HST/cosmos_3dhst.v4.0.F125W_orig_sci.fits
 - data/images/3D-HST/cosmos_3dhst.v4.0.F140W_orig_sci.fits
 - data/images/3D-HST/cosmos_3dhst.v4.0.F160W_orig_sci.fits

 - data/catalogs/3D-HST/goodsn_3dhst.v4.1.cats/Catalog/goodsn_3dhst.v4.1.cat.FITS"
 - data/catalogs/3D-HST/goodsn_3dhst.v4.1.cats/Fast/goodsn_3dhst.v4.1.fout.FITS"
 - data/catalogs/3D-HST/goodsn_3dhst_v4.1.5_catalogs/goodsn_3dhst.v4.1.5.zfit.linematched.fits
 - data/catalogs/3D-HST/goodsn_3dhst_v4.1.5_catalogs/goodsn_3dhst.v4.1.5.linefit.linematched.fits
 - data/images/3D-HST/goodsn_3dhst.v4.0.F125W_orig_sci.fits
 - data/images/3D-HST/goodsn_3dhst.v4.0.F140W_orig_sci.fits
 - data/images/3D-HST/goodsn_3dhst.v4.0.F160W_orig_sci.fits

 - data/catalogs/3D-HST/goodss_3dhst.v4.1.cats/Catalog/goodss_3dhst.v4.1.cat.FITS"
 - data/catalogs/3D-HST/goodss_3dhst.v4.1.cats/Fast/goodss_3dhst.v4.1.fout.FITS"
 - data/catalogs/3D-HST/goodss_3dhst_v4.1.5_catalogs/goodss_3dhst.v4.1.5.zfit.linematched.fits
 - data/catalogs/3D-HST/goodss_3dhst_v4.1.5_catalogs/goodss_3dhst.v4.1.5.linefit.linematched.fits
 - data/images/3D-HST/goodss_3dhst.v4.0.F125W_orig_sci.fits
 - data/images/3D-HST/goodss_3dhst.v4.0.F140W_orig_sci.fits
 - data/images/3D-HST/goodss_3dhst.v4.0.F160W_orig_sci.fits

 - data/catalogs/3D-HST/uds_3dhst.v4.2.cats/Catalog/uds_3dhst.v4.2.cat.FITS"
 - data/catalogs/3D-HST/uds_3dhst.v4.2.cats/Fast/uds_3dhst.v4.2.fout.FITS"
 - data/catalogs/3D-HST/uds_3dhst_v4.1.5_catalogs/uds_3dhst.v4.1.5.zfit.linematched.fits
 - data/catalogs/3D-HST/uds_3dhst_v4.1.5_catalogs/uds_3dhst.v4.1.5.linefit.linematched.fits
 - data/images/3D-HST/uds_3dhst.v4.0.F125W_orig_sci.fits
 - data/images/3D-HST/uds_3dhst.v4.0.F140W_orig_sci.fits
 - data/images/3D-HST/uds_3dhst.v4.0.F160W_orig_sci.fits

------------------
zCOSMOS Bright DR2
------------------

References:
 - zCOSMOS: A Large VLT/VIMOS Redshift Survey Covering 0 < z < 3 in the COSMOS Field.
   S. J. Lilly, O. Le Fèvre, A. Renzini, et al. 2007, ApJS, 172, 1.
   doi:10.1086/516589

Sources:
 - DR2 source unknown
   TODO: Upgrade to DR3 (https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/172/70#/browse)

Files:
 - data/catalogs/zCOSMOS-BRIGHT/zCOSMOS_v2.csv

---------
VUDS DR1
---------

References:
 - The VIMOS Ultra Deep Survey first data release: Spectra and spectroscopic
   redshifts of 698 objects up to zspec ~ 6 in CANDELS. 2017, A&A, 600, A110.
   doi.org/10.1051/0004-6361/201527963

Sources:
 - https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/600/A110#/browse
   TODO: use dat files instead of CSV

Files:
 - data/catalogs/VUDS/cesam_vuds_spectra_dr1_cosmos_catalog.csv
 - data/catalogs/VUDS/cesam_vuds_spectra_dr1_ecdfs_catalog.csv

----------
Casey DSFG
----------

References:
 - A Redshift Survey of Herschel Far-infrared Selected Starbursts and
   Implications for Obscured Star Formation. Casey, C. M., Berta, S.,
   Bethermin, M., et al. 2012, ApJ, 761, 140.
   doi:10.1088/0004-637X/761/2/140

Sources:
 - https://iopscience.iop.org/article/10.1088/0004-637X/761/2/140#apj449592t1

Files:
 - data/catalogs/Casey/apj449592t1_mrt.txt

----------
DEIMOS 10K
----------

References:
 - The DEIMOS 10K Spectroscopic Survey Catalog of the COSMOS Field. Hasinger, G.,
   Capak, P., Salvato, M., et al. 2018, ApJ, 858, 77. 
   doi:10.3847/1538-4357/aabacf

Sources:
 - https://irsa.ipac.caltech.edu/data/COSMOS/tables/deimos/

Files:
 - data/catalogs/DEIMOS/deimos_redshifts.tbl

------------
MOSDEF Final
------------

References:
 - The MOSFIRE Deep Evolution Field (MOSDEF) Survey: Rest-frame Optical Spectroscopy
   for ~1500 H-selected Galaxies at 1.37 < z < 3.8. Kriek, M., Shapley, A. E., Reddy,
   N. A., et al. 2015, ApJS, 218, 15.
   doi:10.1088/0067-0049/218/2/15

Sources:
 - https://mosdef.astro.berkeley.edu/for-scientists/data-releases/

Files:
 - mosdef_zcat.final_slitap.fits
 - linemeas_nocor.fits

--------------
FMOS-COSMOS v2
--------------

References:
 - The FMOS-COSMOS Survey of Star-forming Galaxies at z ∼ 1.6. VI. Redshift and
   Emission-line Catalog and Basic Properties of Star-forming Galaxies. Kashino,
   D., Silverman, J. D., Sanders, D., et al. 2019, ApJS, 241, 10.
   doi:10.3847/1538-4365/ab06c4

Sources:
 - https://member.ipmu.jp/fmos-cosmos/FC_catalogs.html

Files:
 - data/catalogs/FMOS/fmos-cosmos_catalog_2019.fits

------
KMOS3D
------

References:
 - The KMOS3D Survey: Data Release and Final Survey Paper. Wisnioski, E.,
   Förster Schreiber, N. M., Fossati, M., et al. 2019, ApJ, 886, 124.
   doi:10.3847/1538-4357/ab4db8

Sources:
 - https://www.mpe.mpg.de/ir/KMOS3D/data

Files:
 - data/catalogs/KMOS3D/k3d_fnlsp_table_v3.fits
 - data/catalogs/KMOS3D/k3d_fnlsp_table_hafits_v3.fits

----------------
C3R2 DR1/DR2/DR3
----------------

References:
 - The Complete Calibration of the Color-Redshift Relation (C3R2) Survey: Survey
   Overview and Data Release 1. Masters, D. C., Stern, D. K., Cohen, J. G., et al.
   2017, ApJ, 841, 111.
   doi:10.3847/1538-4357/aa6f08

 - The Complete Calibration of the Color-Redshift Relation (C3R2) Survey: Analysis
   and Data Release 2. Masters, D. C., Stern, D. K., Cohen, J. G., et al. 2019,
   ApJ, 877, 81.
   doi:10.3847/1538-4357/ab184d

 - Euclid Preparation. XIV. The Complete Calibration of the Color-Redshift Relation
   (C3R2) Survey: Data Release 3. Stanford, S. A., Masters, D., Darvish, B., et al.
   2021, ApJS, 256, 9. 
   doi:10.3847/1538-4365/ac0833

Sources:
 - https://sites.google.com/view/c3r2-survey/catalog

Files:
 - data/catalogs/C3R2/c3r2_DR1+DR2_2019april11.txt
 - data/catalogs/C3R2/C3R2-DR3-18june2021.txt

----------
LEGA-C DR3
----------

References:
 - The Large Early Galaxy Astrophysics Census (LEGA-C) Data Release 3: 3000
   High-quality Spectra of Ks-selected Galaxies at z > 0.6. van der Wel, A.,
   Bezanson, R., D'Eugenio, F., et al. 2021, ApJS, 256, 44.
   doi:10.3847/1538-4365/ac1356

Sources:
 - https://users.ugent.be/~avdrwel/research.html#legac

Files:
 - data/catalogs/LEGAC/legac_dr3_cat.fits

------------
HSC-SSP PDR3
------------

References:
 - Third Data Release of the Hyper Suprime-Cam Subaru Strategic Program. Aihara, H.,
   AlSayyad, Y., Ando, M., et al. 2022, PASJ, 74, 247.
   doi:10.1093/pasj/psab122

Sources:
 - https://hsc-release.mtk.nao.ac.jp/doc/index.php/catalog-of-spectroscopic-redshifts__pdr3/
  
Files:
 - data/catalogs/HSCSSP/COSMOS-specz-v2.8-public.fits
 - data/catalogs/HSCSSP/EGS-specz-v2.3.fits

--------
DESI EDR
--------

References:
 - A Large Sample of Extremely Metal-poor Galaxies at z < 1 Identified from the DESI
   Early Data. Zou, H., Sui, J., Saintonge, A., et al. 2024, ApJ, 961, 173.
   doi:10.3847/1538-4357/ad1409

Sources:
 - https://data.desi.lbl.gov/doc/releases/edr/vac/stellar-mass-emline/

Files:
 - data/catalogs/DESI/edr_galaxy_stellarmass_lineinfo_v1.0.fits
