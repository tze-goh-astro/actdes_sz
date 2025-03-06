---------------------------------------------------------------------------------------------------------------------------------
																
	NAME:															
		README.txt													
																
																
	WHAT:															
		Read me to figure out how to use this script									
																	
	DATE:																
																
		Jan 27, 2025													
																
---------------------------------------------------------------------------------------------------------------------------------

--------------------------------------
Set the right virtual environment 1st |
--------------------------------------
1) Move into the right folder 
	$ cd ~/Destop/actdes_sz

2) Check that the PYTHONPATH (it is in .bashrc ) is correct 
	$ echo $PYTHONPATH
		-- > /Users/tzegoh_2/Desktop/actdes_sz/bin

3) Source the virtual environment ( in the ~/Desktop/actdes_sz/ folder)
	$ source actdes_sz_env/bin/activate
		--> Note: you must source it 1st before you can pip install XXX XXX 

4) Go to the jupyter environment 
	$ jupyter-lab


-------------------
Running the script |
-------------------
	$ cd actdes_sz
	$ cd 1)actdes_sz_script
	$ python actdes_sz.py 

It runs through the following steps
	0) Get Relevant Info
	1) Find the Area for HealPix
	2) Run TreeCorr
		Process angular correlation measurements
	4) Run Limber Approximation
		Make theoretical angular correlation
	5)Run MCMC
		Minimizes the difference between the 2 & 3
		Gets you the bias of the cluster in 4 different redshift bins

NOTE that in main(), you can block out the parts of the code in blocks indicated above.

-------------------------
After running the script | 
-------------------------
    $ jupyter-lab

        Use notebook 7C)ContourPlots to plot the final plots


------
NOTES |
------
Postscripts & annotations:
CC --> Cluster-Cluster
CG --> Cluster-Galaxy
G  --> Galaxy
REF -> reference only 

zFiles 	= Redshift Bins = zRange 
z1 	= 0.20 - 0.40  	= 020-040 
z2 	= 0.40 - 0.55	= 040-055
z3 	= 0.55 - 0.70	= 055-070
z4 	= 0.70 - 0.85	= 070-085

THEORY :
	ω_cnc, ω_cng --> the angular correlation calculated from Limber Approximation 

OBSERVATION :
	ω_corr, ω_corr --> the angular correlation calculated from Landy Szalay (TreeCorr) 
