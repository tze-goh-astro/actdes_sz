# actdes_sz

Introduction
Physics
		 	 	 		
The project very simply aims to calculate angular correlation of a model (Limber Approximation) vs. that of observation (from DES and ACT).			

From DES and ACT, it obtains location of clusters and galaxies, which it calculates the correlation between clusters vs clusters, and then clusters vs galaxies, using HealPy and TreeCorr.					

For the model, it uses the interpolation function on weights taken from the observations above to calculate the Power Spectrum from CAMB. And then from CAMB, we calculate the angular correlation using the Limber Approximation.		

With the angular correlation from model and data, it is then possible to calculate the bias (overdensity) between the two using MCMC.

