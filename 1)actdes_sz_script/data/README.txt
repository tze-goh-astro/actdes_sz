

-----------------------------------------
D A T A    &   R A N D O M    M A P S    | 
-----------------------------------------

---------------------------------------------------------------------------------------------------------------------------

-------------------
 C L U S T E R S   |
-------------------



D A T A :   C L U S T E R S    A C T 
------------------------------------

image_file = get_pkg_data_filename('1)data/DR5_cluster-catalog_v1.1.fits') 


NOTE: 	

	https://iopscience.iop.org/article/10.3847/1538-4365/abd023/pdf

	


R A N D O M :   G A L A X I E S   D E S   Y 3   
---------------------------------------------

image_fileM = get_pkg_data_filename('1)data/S18d_202006_DESSNR6Scaling_oversampledmock_DESY3.fits')


NOTE:

	The DES Y3 galaxies are used as the random map to the ACT clusters 


---------------------------------------------------------------------------------------------------------------------------


-------------------
 G A L A X Y       |
-------------------



# note that you can fix these files later... it is just naming them properly 


D A T A :   G A L A X I E S   D E S     
---------------------------------------------
# This is from -1)data of -1_Oth_Project_REDO

lens_maglim_z1.fits --> 020-040.npy
etx
etc
etc


R A N D O M :   G A L A X I E S   D E S     
---------------------------------------------

# This is from 3)data of -1_Oth_Project_REDO

rand_maglim_z1.fits   -> rand_maglim_020-040.fits rand_maglim_z2.fitsrand_maglim_z3.fitsrand_maglim_z4.fits


---------------------------------------------------------------------------------------------------------------------------


-------------------
 P R I O R S       |
-------------------

# This is from 5)data of -1_Oth_Project_REDO

chain_3x2pt_lcdm_SR_maglim.txt -- > bgAverage.npy + bgCovMatrix.npy



-------------------
 W E I G H T S     |
-------------------

# This is from 0)data of -1_Oth_Project_REDO

# These are the galaxy weights

'2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits'



-------------------
 P A T C H E S     |
-------------------


# These are the pathces generated from TreeCorr
# We generate them in the notebooks, and we put them herecat



