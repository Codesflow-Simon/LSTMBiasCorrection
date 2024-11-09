Each file contains 20 years of daily rainfall data.

Filename: the six digit code is the catchment ID. Details and location of the catchments can be found in Charles et al. (2020).

Columns:

dayno: since each csv file contains historical and future data, “dayno” is used as the date column. That is, when the column is historical data, dayno measures the number of days after 31 December 1989, when the column is future data, dayno measures the number of days after 31 December 2059, i.e. the historical period is 1990-2009 inclusive, and the future period is 2060-2079 inclusive.

“agg” in each column name indicates that the data was aggregated pointwise to catchment outlines (same for each column).

“agg.AWAP” is observed historical rainfall from AWAP, see Potter et al. (2020, section 2.2). This is our ground truth or reference observational data that we use to bias correct modelled output to.

“CCCMA3.1”, “CSIRO.MK3.0”, “ECHAM5”, “MIROC3.2” are the host global climate models (GCMs) used as forcing for the RCM.

“NNRP_reanalysis”, and “ERAI_reanalysis” are climate reanalysis products. These are blends of past observations with modelled dynamics and can be thought of as “climate modelled observations”, but generally also need some bias correction (although not as much as GCM/RCM outputs).

The “R1”, “R2”, and “R3” suffixes in each column refers to the physics configurations from the WRF regional climate model (RCM) (see Potter et al., 2020, section 2.1 and references therein). We generally consider each GCM/physics scheme combination as a separate modelled product.

“raw” is raw RCM output (i.e. what we want to bias correct)

“bc” is previously bias-corrected rainfall as per Potter et al. (2020) and Charles et al. (2020).

“hist” and “futb” suffixes refer to historical and future time periods respectively as mentioned above.

Units are mm/day, or millimetres of rainfall falling on each square metre of the catchment area per day.