# Pioneer Final Paper Appendix A
This is Appendix A of Yu Lu's Pioneer final paper. The project performs sulfur speciation in petroleum asphaltene using K-edge XANES spectra. Codes for data analysis can be found here.
Parameters used for background subtraction and results not selected in Table 6 are available. This README describes the function and contents of each file
in this repository. 
There are several points in the code where pandas.read or pandas.to_excel are used. Please change the address in the following codes to the address on your computer. You may adjust any parameters you want.
The notation of samples in the program is s1-petroleum asphaltene; s2-DBS; s3-DBSO; s4-DBT; s5-FeSO4; s6-DBSO2.

**1. Appendix A.pdf**

It is a pdf recording the data not prsented in the paper but necessary for data analysis. These include 1) the parameters and result for background subtraction; 2) all the speciation results using differnet methods and parameters (including those not selected in the final paper).

**2. data_prepare.py; coef_lc_reg_localmin.py; coef_lc_globalmin.py**

You can run these files if you want to perform speciation using linear combination fitting. 

If you want to perform linear combination fitting using the local minimum of chi2 method, then 1) run data_prepare.py (remember to adjust the location of output), 2) run coef_lc_reg_localmin.py (remember to adjust the location of input)

If you want to perform linear combination fitting using the global minimum of chi2 method, then 1) run data_prepare.py (remember to adjust the location of output), 2) run coef_lc_globalmin.py (remember to adjust the location of input)

Function of each file is described below.

*a) data_prepare.py*

It performs background subtraction, normalization, and energy-column unification. A file called norm.xlsx is output to the location specified. PLease change that to the location on your own computer.

*b) coef_lc_reg_localmin.py*

It performs speciation using linear combination fitting that first finds the global min of residual squared and then finds the local min of chi2(the first method described in the paper).
The file generates some fitting figures for comparison and print the fitting result.

*c) coef_lc_globalmin.py*


It performs speciation using linear combination fitting that directly finds the global min of chi2(the second method described in the paper).
The file generates some fitting figures for comparison and print the fitting result.

**3. coef_decon_in1.py**

You just need to run this file if you want to perform speciation using deconvolution method (no need to run data_prepare.py because it is contained in this file).
8 figures are produced each loop. The first one is the result of background subtraction, the second one is the result of normalizatoin, the rest are result of deconvolution fitting.
Results of deconvolution will also be printed (the fitting report object in lmfit). Because this program runs for a long time (100~3000 seconds each loop), a notification 
will be sent if you are using a windows 10 system. You can delete this part if this causes bug. 
