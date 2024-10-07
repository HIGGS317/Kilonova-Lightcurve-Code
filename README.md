# Kilonova-Lightcurve-Code
A code for generating Kilonova Lightcurves

This code is primaraly a very basic modification of the code [Heating Rate](https://github.com/hotokezaka/HeatingRate) code of hotokezaka for the paper [Radioactive Heating Rate of r-process Elements and Macronova Light Curve](https://iopscience.iop.org/article/10.3847/1538-4357/ab6a98)

## Note to run the code:

- Lightcurves can be generated using any of the <b>*_lightcurve.ipynb</b> notebook using the eos files

- Parameter fitting and comparison can be done with <b> Ejecta_Parameter and Ejecta_Fit_Comparison </b>

- Bolometric Lightcurves with absolute magnitude on y-axis is generated in <b> Bolometric_LightCurve.ipynb </b>

- Some common used EOSs are given in the folder
**EOS**. Change the path in the code accordingly

## Information on provided EOS

2 of the most common EOSs has been provided in the EOS folder: 1. DD2  2. Sly4

a) For DD2 coloumn structure is 
  -  Energy Density = 3rd Column
  -  Pressure = 4th Column

b) For SLY4 coloumn structure is 
  -  Energy Density = 2nd Column
  -  Pressure = 3rd Column

**<span style="color:red">Note : </span>** There are additional columns in EOSs files but these are not required by us at the current moment.

