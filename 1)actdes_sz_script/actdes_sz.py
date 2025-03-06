#!/usr/bin/env python3
"""

Name: actdes_sz.py

Former name: TzeFinalProj.py

by Tze Goh

University of Cambridge
Kavli Institute of Cosmology 

This python file runs angular correlation, split broadly into 5 parts :
        0) Getting and cutting the data to relevant info
        1) Runs HealPix  ( maps sky against random pixels)
        2) Runs TreeCorr ( calculate the angular correlation )
        3) Runs Limber Approx (via CAMB ) 
        4) MCMC

        
April 12, 2022 created

Feb 10, 2025 updated 

"""

from __future__ import print_function
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(12,8)
import numpy as np
import pickle 
import astropy
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import sys, platform, os
import healpy as hp 
import treecorr
print(treecorr.__version__,'treecorr version')
from IPython.display import set_matplotlib_formats
import camb
from scipy.interpolate import interp1d
from camb import model
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from scipy import interpolate
from numpy.linalg import inv
import emcee
import corner
import os.path
import sys
from scipy.optimize import minimize
from scipy.stats import chisquare

def TestADD2(a,b):
    result = a+b
    return(result)
    
class gettingData:
    """
    This class is just for taking in the initial data
    """
    def __init__(self):
        """
        """
        
    def step0_getRelevantInfo(self,zRange,path) :
        """
            0) Gets the definition for you 
        """ 
        print(outputDir,' is the outputDirectory. ',outputData,'is the outputData.')
        if zRange  == '020-040':
            zFile = 'z1'
            
        elif zRange  == '040-055':
            zFile = 'z2' 
            
        elif zRange  == '055-070':
            zFile = 'z3'
            
        elif zRange  == '070-085':
            zFile = 'z4'
            
        print(zRange,'is the redshift range')  
        print(zFile,'is the redshift file that you want')         
        """
        Get the data 
        """
        # Get the ImageFile
        image_file,image_fileG,image_fileM,image_fileGM = self.getImageFile(path,zFile)
        # Look at the headers
        self.printMeHeaders(image_file,image_fileG,image_fileM,image_fileGM)
        # Actuallyy get the data 
        image_table,\
        image_tableG,\
        image_tableM,\
        image_tableGM = self.getData(image_file, image_fileG, image_fileM, image_fileGM)
        
        """
        The Definitions you actuallly need 
        """    
        #terms for GALAXIES 
        RA_G,DEC_G,redshift_G,\
        min_redshift_G,max_redshift_G,w_G=self.GiveMeGalaxyDefinitions(image_tableG)

        #terms for CLUSTERS, CLUSTERS-MASK and GALAXIES-MASK 
        [RA,DEC,redshift,snr,\
        rand_ra_M,rand_dec_M,rand_z_M,rand_snr_M,\
        rand_ra_GM,rand_dec_GM,rand_z_GM]=self.GiveMePartialDefinitions(image_table,\
                                                          image_tableM,\
                                                          image_tableGM)
        #These are the shapes you're workiing witth
        print("BEFORE CUTTING")
        print(RA_G.shape,'RA_G.shape')
        print(RA.shape,'RA.shape -- cluster')
        print(rand_ra_M.shape,'rand_ra_M.shape -- cluster')
        print(rand_ra_GM.shape,'rand_ra_GM.shape -- galaxy')
        print(w_G.shape,'w_G.shape\n')
        myList = [RA_G,DEC_G,redshift_G,\
                            min_redshift_G,max_redshift_G,w_G,
                            RA,DEC,redshift,snr,\
                            rand_ra_M,rand_dec_M,rand_z_M,rand_snr_M,\
                            rand_ra_GM,rand_dec_GM,rand_z_GM]
        
        SaveHer = str(outputData+'/RelevantDataB4Cut')
        with open(SaveHer, 'wb') as f: 
            pickle.dump(myList, f) 
            
        """
        CUT HER UP 
        """           
        [RA,DEC,redshift_left,\
        rand_ra_M,rand_dec_M,rand_z_M_left,\
        RA_G,DEC_G,redshift_left_G,w_G_left,\
        rand_ra_GM,rand_dec_GM,rand_z_GM_left]=self.CUT(min_redshift_G,max_redshift_G,snr,\
                                               RA,DEC,redshift,\
                                               rand_ra_M,rand_dec_M,rand_z_M,rand_snr_M,\
                                               RA_G,DEC_G,redshift_G,w_G,\
                                               rand_ra_GM,rand_dec_GM,rand_z_GM)


        min_redshift_left_G = round(min(redshift_left_G),2)
        min_redshift_left_G =str("{:.2f}".format(min_redshift_left_G))
        max_redshift_left_G = round(max(redshift_left_G),2)
        max_redshift_left_G =str("{:.2f}".format(max_redshift_left_G))
        print(min_redshift_left_G,max_redshift_left_G,'min(redshift_left_G),max(redshift_left_G)')
        
        print("\n AFTER CUTTING")
        print(RA_G.shape,'RA_G.shape')
        print(RA.shape,'RA.shape -- cluster')
        print(rand_ra_M.shape,'rand_ra_M.shape -- cluster')
        print(rand_ra_GM.shape,'rand_ra_GM.shape -- galaxy')
        print(w_G_left.shape,'w_G_left.shape\n')
        print('We are done with the cutting\n')
        myList = [RA,DEC,redshift_left,\
                    rand_ra_M,rand_dec_M,rand_z_M_left,\
                    RA_G,DEC_G,redshift_left_G,w_G_left,\
                    rand_ra_GM,rand_dec_GM,rand_z_GM_left,
                     min_redshift_left_G,max_redshift_left_G]
        
        SaveHer = str(outputData+'/RelevantDataAfterCut')
        with open(SaveHer, 'wb') as f: 
            pickle.dump(myList, f)              
        
        print('step0_getRelevantInfo(zRange,path)... Ran\n')

    def getOutputDirectory(self,zRange):
        """
        This gets Output Directory for you.
        """    
        # Where you gonna save the plots 
        path = 'OutputPlots/'
        filename = str(zRange+'/')
        outputDir=path+filename
        path = 'OutputData/'
        outputData=path+filename

        return (outputDir,outputData)

    def getImageFile(self,path,zFile):
        """
        This gets the image file
        """
        # CLUSTER
        #filename = 'DR5_cluster-catalog_v1.0b2.fits'  #this is old
        filename = 'DR5_cluster-catalog_v1.1.fits'
        image_file = get_pkg_data_filename(path+filename)
        #|------------------------------------------------------
        # GALAXY
        #filename = str(zRange+'.npy') ... this is oldData
        #image_fileG =np.load(path+filename) ... this is oldData
        filename = str('lens_maglim_'+zFile+'.fits')
        image_fileG=get_pkg_data_filename(path+filename)
        #|------------------------------------------------------
        ''' NOW Let us get the random maps '''
        #CLUSTER - random mask
        filename = 'S18d_202006_DESSNR6Scaling_oversampledmock_DESY3.fits'
        image_fileM = get_pkg_data_filename(path+filename)
        #|------------------------------------------------------
        #GALAXY - random mask 
        #filename = str('rand_maglim_'+zRange+'.fits') ... this is oldData
        filename = str('rand_maglim_'+zFile+'.fits')    
        image_fileGM = get_pkg_data_filename(path+filename) 

        #|------------------------------------------------------

        return (image_file,image_fileG,image_fileM,image_fileGM)

    def getData(self,image_file,image_fileG,image_fileM,image_fileGM):
        """
        This actually gets the data for you.
        """
        #CLUSTER 
        image_table = fits.getdata(image_file, ext=1)
        print(image_table.shape,'CLUSTERS')
        #GALAXIES
        # image_tableG = image_fileG  # ... this is for oldData
        image_tableG = fits.getdata(image_fileG, ext=1)
        print(image_tableG.shape,'GALAXIES')
        # CLUSTERS - MASK 
        image_tableM = fits.getdata(image_fileM, ext=1)
        print(image_tableM.shape,'CLUSTERS MASK')
        # GALAXIES - MASK
        image_tableGM = fits.getdata(image_fileGM, ext=1)
        print(image_tableGM.shape,'GALAXIES MASK')

        return(image_table,image_tableG,image_tableM,image_tableGM)


    def GimmeHeader(self,image_file):
        '''This gives you the header from the fits file  '''
        image_data_H = fits.getheader(image_file, ext=0)
        image_table_H = fits.getheader(image_file, ext=1)
        print(image_data_H)
        print(image_table_H)    
        print('\n')
        
        return

    def printMeHeaders(self,image_file,image_fileG,image_fileM,image_fileGM):
        """
        This prints you the headers of the files, including the weights
        """
        print('CLUSTERS')
        self.GimmeHeader(image_file)
        print('\n')
        print('CLUSTERS - MASK ')
        self.GimmeHeader(image_fileM)
        print('\n')
        print('GALAXIES - MASK')
        self.GimmeHeader(image_fileGM)
        print('\n')
        print('GALAXIES')
        self.GimmeHeader(image_fileG)
        print('\n')    
    

    def GiveMeGalaxyDefinitions(self,image_tableG):
        ''' The new one 
        This gives you the names and functions for GALAXY. 
        The way of the oldData is kept here as a REF'''
        # GALAXY 
        RA_G =image_tableG['ra'] #         RA_G =image_tableG[0,:] ... oldData
        DEC_G = image_tableG['dec']#        DEC_G = image_tableG[1,:]... oldData
        redshift_G = image_tableG['z'] #        redshift_G = image_tableG[2,:] ... oldData 
        w_G  = image_tableG['w'] #         w_G  = image_tableG[3,:] ...oldDATA
        min_redshift_G = (round(min(redshift_G),2))
        min_redshift_G =str("{:.2f}".format(min_redshift_G))
        max_redshift_G = (round(max(redshift_G),2))
        max_redshift_G =str("{:.2f}".format(max_redshift_G))
        print('GALAXY')
        print(min(RA_G),max(RA_G),'min(RA_G),max(RA_G)')
        print(min(DEC_G),max(DEC_G),'min(DEC_G),max(DEC_G)')
        print(min_redshift_G,max_redshift_G,'min(redshift_G),max(redshift_G)')
        print('\n')
        
        return(RA_G,DEC_G,redshift_G,min_redshift_G,max_redshift_G,w_G)
    
    
    def GiveMePartialDefinitions(self, image_table,image_tableM,image_tableGM):
        '''This gives you the names and functions for the rest of program '''
        # CLUSTERS
        RA =image_table['RADeg']
        DEC = image_table['decDeg']
        redshift = image_table['redshift']
        snr = image_table['fixed_SNR']
        print('CLUSTER')
        print(min(RA),max(RA),'min(RA),max(RA)')
        print(min(DEC),max(DEC),'min(DEC),max(DEC)')
        print(min(redshift),max(redshift),'min(redshift),max(redshift)')
        print(min(snr),max(snr),'min(snr),max(snr)')

        #CLUSTERS - MASK
        rand_ra_M =image_tableM['RADeg']
        rand_dec_M = image_tableM['decDeg']
        rand_z_M =image_tableM['redshift']
        rand_snr_M =image_tableM['fixed_SNR']
        print('CLUSTER--MASK')
        print(min(rand_ra_M),max(rand_ra_M),'min(rand_ra_M),max(rand_ra_M)')
        print(min(rand_dec_M),max(rand_dec_M),'min(rand_dec_M),max(rand_dec_M)')
        print(min(rand_z_M),max(rand_z_M),'min(rand_z_M),max(rand_z_M)')
        print(min(rand_snr_M),max(rand_snr_M),'min(rand_snr_M),max(rand_snr_M)')
        print(rand_ra_M.shape,'rand_ra_M.shape')

        #GALAXY - MASK 
        rand_ra_GM =image_tableGM['ra']
        rand_dec_GM = image_tableGM['dec']
        rand_z_GM = image_tableGM['z']

        print('GALAXY--MASK')
        print(min(rand_ra_GM),max(rand_ra_GM),'min(rand_ra_GM),max(rand_ra_GM)')
        print(min(rand_dec_GM),max(rand_dec_GM),'min(rand_dec_GM),max(rand_dec_GM)')
        print(min(rand_z_GM),max(rand_z_GM),'min(rand_z_GM),max(rand_z_GM)')
        print(rand_ra_GM.shape,'rand_ra_GM.shape -- THIS IS HUGE')
        print('\n')

        return(RA,DEC,redshift,snr,\
               rand_ra_M,rand_dec_M,rand_z_M,rand_snr_M,\
               rand_ra_GM,rand_dec_GM,rand_z_GM)

    def CUT(self,min_redshift_G,max_redshift_G,snr,\
           RA,DEC,redshift,\
           rand_ra_M,rand_dec_M,rand_z_M,rand_snr_M,\
           RA_G,DEC_G,redshift_G,w_G,\
           rand_ra_GM,rand_dec_GM,rand_z_GM):
        ''' This cuts the data down for you. There are  3 main cuts :
              1) redshift (clusters & cluster randoms only) within redshift range
              2) snr ( clusters & cluster randoms only)  > 5
              3) dec ( galaxies & galaxies randoms only) > -60 deg

              '''

        #CLUSTER
        RA=RA[(redshift>float(min_redshift_G))&(redshift<float(max_redshift_G))&(snr>5)]
        DEC=DEC[(redshift>float(min_redshift_G))&(redshift<float(max_redshift_G))&(snr>5)]
        redshift_left = redshift[(redshift>float(min_redshift_G))&\
                                 (redshift<float(max_redshift_G))&(snr>5)]
        print('CLUSTERS')
        print(min(RA),max(RA),'min(RA),max(RA)')
        print(min(DEC),max(DEC),'min(DEC),max(DEC)')
        print(min(redshift_left),max(redshift_left),'min(redshift_left),max(redshift_left)')
        print(RA.shape,'RA.shape --left\n')

        #Clusters--MASK
        rand_ra_M =rand_ra_M[(rand_z_M>float(min_redshift_G))&\
                             (rand_z_M<float(max_redshift_G))&(rand_snr_M>5)]
        rand_dec_M =rand_dec_M[(rand_z_M>float(min_redshift_G))&\
                               (rand_z_M<float(max_redshift_G))&(rand_snr_M>5)]
        rand_z_M_left=rand_z_M[(rand_z_M>float(min_redshift_G))&\
                               (rand_z_M<float(max_redshift_G))&(rand_snr_M>5)]
        print('CLUSTERS - MASK')
        print(min(rand_ra_M),max(rand_ra_M),'min(rand_ra_M),max(rand_ra_M)')
        print(min(rand_dec_M),max(rand_dec_M),'min(rand_dec_M),max(rand_dec_M)')
        print(min(rand_z_M_left),max(rand_z_M_left),'min(rand_z_M_left),max(rand_z_M_left)')
        print(rand_ra_M.shape,'rand_ra_M.shape -- left\n')

        #Galaxy
        print('GALAXY')
        TEMP_DEC_G = DEC_G
        RA_G=RA_G[TEMP_DEC_G>-60]
        DEC_G=DEC_G[TEMP_DEC_G>-60]
        redshift_left_G=redshift_G[TEMP_DEC_G>-60]
        w_G_left=w_G[TEMP_DEC_G>-60]
        print(min(RA_G),max(RA_G))
        print(min(DEC_G),max(DEC_G))
        print('RA_G.shape -- left \n')

        #Galaxy--MASK
        print('GALAXY--MASK')
        print(rand_ra_GM.shape,'rand_ra_GM.shape')
        print(rand_z_GM.shape,'rand_z_GM.shape')
        print(min(rand_z_GM),max(rand_z_GM),'min(rand_z_GM),max(rand_z_GM)\n')
        temp_rand_dec_GM = rand_dec_GM
        rand_ra_GM =rand_ra_GM[temp_rand_dec_GM>-60]
        rand_dec_GM =rand_dec_GM[temp_rand_dec_GM>-60]
        rand_z_GM = rand_z_GM[temp_rand_dec_GM>-60]
        # Cut the redshift too

        #rand_ra_GM =rand_ra_GM#[(rand_z_GM>min_redshift_G)&(rand_z_GM<max_redshift_G)]
        #rand_dec_GM =rand_dec_GM#[(rand_z_GM>min_redshift_G)&(rand_z_GM<max_redshift_G)]
        rand_z_GM_left=rand_z_GM#[(rand_z_GM>min_redshift_G)&(rand_z_GM<max_redshift_G)]
        print(rand_ra_GM.shape,'rand_ra_GM.shape')
        print(min(rand_ra_GM),max(rand_ra_GM))
        print(min(rand_dec_GM),max(rand_dec_GM))
        #print(rand_ra_GM.shape,'rand_ra_GM.shape -- left')

        return(RA,DEC,redshift_left,
               rand_ra_M,rand_dec_M,rand_z_M_left,
               RA_G,DEC_G,redshift_left_G, w_G_left,
               rand_ra_GM,rand_dec_GM,rand_z_GM_left)  
    
class FindArea:
    """
    This class essentially runs healpix for you.  
    """
    def __init__(self):
        """
        """
        
    def step1_runHealPix(self,zRange,outputDir,outputData,nside,npix):
        """
        This runs healpix for you: turns the points in the sky to pixels.
        """
        print(zRange,'is the redshift range')
        print(outputDir,' is the outputDirectory. ',outputData,'is the outputData.')
        
        #Load the data After Cut
        LoadHer = str(outputData+'/RelevantDataAfterCut')
        with open(LoadHer, 'rb') as f: 
            myList = pickle.load(f)
        [RA,DEC,redshift_left,rand_ra_M,rand_dec_M,rand_z_M_left,\
         RA_G,DEC_G,redshift_left_G,w_G_left,\
         rand_ra_GM,rand_dec_GM,rand_z_GM_left,
         min_redshift_left_G,max_redshift_left_G] = myList

        print('\n BEFORE HEALPY')
        print(nside,'nside')
        print(npix , 'npix ' )
        print(rand_ra_GM.shape , 'rand_ra_GM.shape ' )
        print(min(rand_ra_GM),max(rand_ra_GM),'min(rand_ra_GM),max(rand_ra_GM)')
        print(rand_dec_GM.shape , 'rand_dec_GM.shape ' )
        print(min(rand_dec_GM),max(rand_dec_GM),'min(rand_dec_GM),max(rand_dec_GM)')
      
        print(rand_ra_M.shape , 'rand_ra_M.shape ' )
        print(min(rand_ra_M),max(rand_ra_M),'min(rand_ra_M),max(rand_ra_M)')
        print( rand_dec_M.shape, 'rand_dec_M.shape ' )
        print(min(rand_dec_M),max(rand_dec_M),'min(rand_dec_M),max(rand_dec_M)')
      
        print(RA.shape , 'RA.shape ' )
        print(min(RA),max(RA),'min(RA),max(RA))')
        print(DEC.shape , 'DEC.shape ' )
        print(min(DEC),max(DEC),'min(DEC),max(DEC)')

        
        # Turn points in sky to pixels
        [rand_ra_restricted_CM,\
        rand_dec_restricted_CM,\
        data_ra_restricted_C,\
        data_dec_restricted_C,\
        data_pix_indices_C,\
        rand_pix_indices_CM] = self.GiveMeHealPyClusters(nside,npix,rand_ra_GM,rand_dec_GM,\
                                                   rand_ra_M,rand_dec_M,RA,DEC )

        print('\n AFTER HEALPY')
        print(rand_ra_restricted_CM.shape,'rand_ra_restricted_CM.shape')
        print(min(rand_ra_restricted_CM),max(rand_ra_restricted_CM),
              'min(rand_ra_restricted_CM),max(rand_ra_restricted_CM)')
        print(rand_dec_restricted_CM.shape,'rand_dec_restricted_CM.shape')
        print(min(rand_dec_restricted_CM),max(rand_dec_restricted_CM),
              'min(rand_dec_restricted_CM),max(rand_dec_restricted_CM)')

        print(data_ra_restricted_C.shape,'data_ra_restricted_C.shape')
        print(min(data_ra_restricted_C),max(data_ra_restricted_C),
              'min(data_ra_restricted_C),max(data_ra_restricted_C)')

        print(data_dec_restricted_C.shape,'data_dec_restricted_C.shape')
        print(min(data_dec_restricted_C),max(data_dec_restricted_C),
              'min(data_dec_restricted_C),max(data_dec_restricted_C)')

        
        myList = [RA,DEC,redshift_left,rand_ra_M,rand_dec_M,rand_z_M_left,\
         RA_G,DEC_G,redshift_left_G,w_G_left,\
         rand_ra_GM,rand_dec_GM,rand_z_GM_left,
         min_redshift_left_G,max_redshift_left_G,
         rand_ra_restricted_CM,\
         rand_dec_restricted_CM,\
         data_ra_restricted_C,\
         data_dec_restricted_C,\
         data_pix_indices_C,\
         rand_pix_indices_CM]
        
        SaveHer = str(outputData+'/RelevantDataAfterCut+HealPix')
        with open(SaveHer, 'wb') as f: 
            pickle.dump(myList, f) 
        
        print("Healpix ran !")
        return

    
    def GiveMeHealPyClusters(self,nside,npix,rand_ra_GM,rand_dec_GM,rand_ra_M,rand_dec_M,RA,DEC ):
        ''' This should pixelize the Clusters -- random & data -- and intersect it 
        with the galaxy randoms !!!!!  The return result should be  '''
        ###########################################################################
        fix='C'

        # Now make the mask with the random galaxy points ...
        maskM = np.zeros(npix)
        print(maskM.shape,'maskM.shape')
        rand_phi_GM,rand_theta_GM = self.convert_radec_to_thetaphi(rand_ra_GM,rand_dec_GM)
        print(rand_phi_GM.shape,'rand_phi_GM.shape')
        rand_pix_indices_GM_X = hp.ang2pix(nside, rand_theta_GM, rand_phi_GM)
        print(rand_pix_indices_GM_X.shape,'rand_pix_indices_GM_X.shape')
        maskM[rand_pix_indices_GM_X ]=1
        print(np.where(maskM[rand_pix_indices_GM_X] == 1)[0].shape,
              'np.where(maskM[rand_pix_indices_GM_X] == 1)[0].shape')

        # ... and intersect it with the cluster random points  
        rand_ra_CM=rand_ra_M  # deg
        rand_dec_CM=rand_dec_M # deg
        print(rand_dec_CM.shape,str('rand_dec_'+fix+'M.shape'))
        rand_phi_CM,rand_theta_CM = self.convert_radec_to_thetaphi(rand_ra_CM,rand_dec_CM)
        print(rand_phi_CM.shape,'rand_phi_CM.shape')
        rand_pix_indices_CM = hp.ang2pix(nside, rand_theta_CM, rand_phi_CM)
        print(rand_pix_indices_CM.shape,str('rand_pix_indices_'+fix+'M.shape'))
        # This is where the intersection takes place
        cluster_rand_inmask_CM = np.where(maskM[rand_pix_indices_CM] == 1)[0]
        print(cluster_rand_inmask_CM.shape,'cluster_rand_inmask.shape')
        rand_ra_restricted_CM,\
        rand_dec_restricted_CM = rand_ra_CM[cluster_rand_inmask_CM],\
                                            rand_dec_CM[cluster_rand_inmask_CM]


        # ... and intersect it with the cluster data points  
        data_ra_C = RA #deg
        data_dec_C = DEC #deg
        print(data_ra_C.shape,'data_ra.shape')
        data_phi_C,data_theta_C = self.convert_radec_to_thetaphi(data_ra_C,data_dec_C)
        data_pix_indices_C = hp.ang2pix(nside, data_theta_C, data_phi_C)
        print(data_pix_indices_C.shape,str('rand_pix_indices_'+fix+'.shape'))
        # This is where the intersection takes place
        cluster_inmask_C = np.where(maskM[data_pix_indices_C] == 1)[0]
        print(cluster_inmask_C.shape,'cluster_rand_inmask.shape')
        data_ra_restricted_C,data_dec_restricted_C = RA[cluster_inmask_C],DEC[cluster_inmask_C]
        
        return(rand_ra_restricted_CM,rand_dec_restricted_CM,\
               data_ra_restricted_C,data_dec_restricted_C,\
              data_pix_indices_C,rand_pix_indices_CM) 
    
    def convert_radec_to_thetaphi(self,ra,dec):
        '''Converts from degrees to radians, and then to the angles phi and theta
        for healpy coords. Returns in units of radians.'''

        phi = ra*np.pi/180
        theta = (np.pi/2) - dec*np.pi/180.

        return phi,theta
    
    def GiveMeJacknifePatch(self,Area,GArea,nJack,AreaOnePix):
        '''This print you the Jacknife Patch for both clusters and galaxies  '''

        C_Area = Area.shape[0] * AreaOnePix # the survey area in steriadians 
        G_Area = GArea.shape[0] * AreaOnePix # thee survey area in steriadians
        C_Area_Deg2 = C_Area * (180/np.pi)**2 # the survey area in square deg 
        G_Area_Deg2 = G_Area * (180/np.pi)**2 # thee survey area in square deg
        C_jacknife_Patch2 = C_Area_Deg2/nJack
        G_jacknife_Patch2 = G_Area_Deg2/nJack
        C_jacknife_Patch = C_jacknife_Patch2**0.5
        G_jacknife_Patch = G_jacknife_Patch2**0.5

        print(C_Area,'steradians : Survey area clusters')
        print(G_Area,'steradians : Survey area galaxies')
        print(C_Area_Deg2,'sq  deg : Survey area clusters')
        print(G_Area_Deg2,'sq  deg : Survey area galaxies')
        print(C_jacknife_Patch2,'sq  deg : Jacknife Patch clusters')
        print(G_jacknife_Patch2,'sq  deg : Jacknife Patch galaxies')          
        print(C_jacknife_Patch,'deg : Jacknife Patch clusters')
        print(G_jacknife_Patch,'deg : Jacknife Patch galaxies')  
            
    def GiveMeArea(self,WhatIsIt,nside,npix,rand_ra_M,rand_dec_M):
        ''' OK, We want the area to calculate the jacknife  '''
        #Healpy values on how pixalted you want the map to be 
        print(WhatIsIt)
        #create empty mask of length npix
        mask = np.zeros(npix)
        #Is it clusters or galaxies? 
        if WhatIsIt=='Clusters':
            fix='C'
        elif WhatIsIt=='Galaxies':
            fix='G'

        '''Note that we have 2 data sets :
        rand -- this is the randomly distributed one -- this is the mask 
        data -- this is the real one '''

        #convert every pixel that contains a random point into a value of 1 --> this becomes mask
        # USE THIS AS THE MASK -- RANDOM MASK 
        rand_ra_CM=rand_ra_M  # deg
        rand_dec_CM=rand_dec_M # deg
        print(rand_dec_CM.shape,str('rand_dec_'+fix+'M.shape'))
        rand_phi,rand_theta = self.convert_radec_to_thetaphi(rand_ra_CM,rand_dec_CM)
        print(rand_phi.shape,'rand_phi.shape')
        rand_pix_indices_CM = hp.ang2pix(nside, rand_theta, rand_phi)
        print(rand_pix_indices_CM.shape,str('rand_pix_indices_'+fix+'M.shape'))
        mask[rand_pix_indices_CM] = 1  #  WHERE THERE IS THE SURVEY AREA OF THE MASK
        print(mask[rand_pix_indices_CM].shape,str('mask[rand_pix_indices_'+fix+'M.shape'))
        cluster_rand_inmask = np.where(mask[rand_pix_indices_CM] == 1)[0]
        print(cluster_rand_inmask.shape,'cluster_rand_inmask.shape')

        # This is just for calculating the survey area !!!!!!!
        unique_rand_pix_indices_CM = np.unique(rand_pix_indices_CM)
        Area = mask[unique_rand_pix_indices_CM]
        print(Area.shape,'Area.shape')
        print('\n')
        return Area
    
    
    def ShowMeTheMapAndMask(self,nside,WhatIsIt, data_ra_restricted_C, 
                            data_dec_restricted_C,rand_ra_CM,rand_dec_CM,outputData,
                            outputDir,Name,min_redshift_left_G,max_redshift_left_G ):
        '''This will show you the individual maps along  with what  it looks like masked.  '''

        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white' 
        #Is it clusters or galaxies? 
        if WhatIsIt=='Clusters':
            pointsColor = 'blue'
            maskColor = 'orange'
            size = 1
            alp = 1 
        elif WhatIsIt=='Galaxies':
            pointsColor = 'pink'
            maskColor = 'cyan' 
            size = 0.1
            alp = 0.1

        whereIsIt = np.where(rand_ra_CM > 180 )
        print(rand_ra_CM.shape)
        rand_ra_CM_modified  = rand_ra_CM
        rand_ra_CM_modified[whereIsIt]= rand_ra_CM[whereIsIt]-360
        print(rand_ra_CM_modified.shape)

        whereIsIt = np.where(data_ra_restricted_C > 180 )
        print(rand_ra_CM.shape)
        data_ra_restricted_C_modified  = data_ra_restricted_C
        data_ra_restricted_C_modified[whereIsIt]= data_ra_restricted_C[whereIsIt]-360
        print(data_ra_restricted_C_modified.shape)

        f, (ax1) = plt.subplots(1,  figsize=(15,10))
        plt.style.use('Solarize_Light2')
        ax1.scatter(rand_ra_CM_modified, rand_dec_CM, 
                    color=maskColor, s=0.1,alpha=0.25,label='random masking points')
        ax1.scatter(data_ra_restricted_C_modified, data_dec_restricted_C, 
                    color=pointsColor, s= size,alpha=alp,label='data points')


        plt.plot()
        ax1.set_xlabel('RA (degrees)')
        ax1.set_ylabel('Dec (degrees)')
        ax1.legend(facecolor="white",loc='lower center')
        #lgnd1=ax1.legend(facecolor="white",loc='lower center')
        #for handle in lgnd1.legendHandles:
        #    handle.set_sizes([100])

        titles = str('nside is '+str(nside)+
                     '\n '+WhatIsIt+' with redshift:'+str(min_redshift_left_G) +
                     '_to_'+str(max_redshift_left_G))
        plt.title(titles)

        saveName=str(outputData+Name+' betweeen z = '+str(min_redshift_left_G)+
                     '_to_'+str(max_redshift_left_G)+'_data.npy' )
        ToSave = np.array([data_ra_restricted_C, data_dec_restricted_C])  
        print(ToSave.shape,'ToSave.shape')
        print(saveName,'saveName')
        np.save(saveName,ToSave) 

        saveName=str(outputData+Name+' betweeen z = '+str(min_redshift_left_G)+
                     '_to_'+str(max_redshift_left_G)+'_rand.npy' )  
        ToSave = np.array([                   rand_ra_CM,rand_dec_CM])
        print(ToSave.shape,'ToSave.shape')
        print(saveName,'saveName')
        np.save(saveName,ToSave)    


        saveName=str(outputDir+Name+' betweeen z = '+str(min_redshift_left_G)+
                     '_to_'+str(max_redshift_left_G)+'.pdf')
        plt.savefig(saveName)
        
        
    def GiveMeMollView(self,npix,rand_pix_indices_GM,data_pix_indices_G,outputDir,
                       Name,min_redshift_G,max_redshift_G):
        '''  This just gives you the mollview of the  random points and data  points '''

        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white' 

        map_rand = np.zeros(npix)
        map_rand[rand_pix_indices_GM] = 1

        map_data = np.zeros(npix)
        map_data[data_pix_indices_G] = 1

        hp.mollview(map_rand)
        saveName=str(outputDir+Name+'_rand_'+str(min_redshift_G)+'_to_'+
                str(max_redshift_G)+'.pdf')
        plt.savefig(saveName)

        hp.mollview(map_data)
        saveName=str(outputDir+Name+'_data_'+str(min_redshift_G)+'_to_'+
                str(max_redshift_G)+'.pdf')
        plt.savefig(saveName)
    
    
    
class Tree:
    """
    This class essentially runs TreeCorr for you.  
    """
    def __init__(self):
        """
        """  
        
    def step2_runTreeCorr(self,zRange,outputDir,outputData,nside,npix,nJack,NoOfBins,
                         minSepGal,maxSepGal,minSepCluster,
                          maxSepCluster,minSepCross,maxSepCross):
        """
        This runs healpix for you: turns the points in the sky to pixels.
        """
        catG,catGM, catC,catCM = 0,0,0,0 # clear some space
        print(zRange,'is the redshift range')
        print(outputDir,' is the outputDirectory. ',outputData,'is the outputData.')
        
        #Load the data After Cut and HealPix
        LoadHer = str(outputData+'/RelevantDataAfterCut+HealPix')
        with open(LoadHer, 'rb') as f: 
            myList = pickle.load(f)                     
        [RA,DEC,redshift_left,rand_ra_M,rand_dec_M,rand_z_M_left,\
         RA_G,DEC_G,redshift_left_G,w_G_left,\
         rand_ra_GM,rand_dec_GM,rand_z_GM_left,
         min_redshift_left_G,max_redshift_left_G,
         rand_ra_restricted_CM,\
         rand_dec_restricted_CM,\
         data_ra_restricted_C,\
         data_dec_restricted_C,\
         data_pix_indices_C,\
         rand_pix_indices_CM] = myList 
        
        # GALAXIES
        #---------------------------
        # NOTE: YOU DO NOT NEED TO RUN THIS AGAIN
        # JUST LOAD THE catGM and you will be fine
        
        Name='catGM_'
        loadName = str(path+Name+zRange+'_TreeCorr_GalaxyRandom.npy')# this is the correct one
        #loadName = str('oldData/'+Name+zRange+'_TreeCorr_GalaxyRandom.npy') # catGM from old code
        print(loadName,'loadName') 


        if os.path.isfile(loadName) == True:
            print('We already have the catGM: random catalogue for Galaxies, and we will just load it.')
            catGM_patch_centers = np.load(loadName)
        
            # The random map for masking does not need weight, & use the jacknife estimation for patches
            catGM = treecorr.Catalog(ra=rand_ra_GM,dec=rand_dec_GM,
                                     patch_centers=catGM_patch_centers,
                                     ra_units='deg', dec_units='deg')        
            
            # The data map does need weight, and use the random map's patches for patches
            catG = treecorr.Catalog(ra=RA_G,dec=DEC_G, w=w_G_left,
                                    patch_centers=catGM_patch_centers,
                                  ra_units='deg', dec_units='deg')
        else:
            print('We do NOT have the catGM: random catalogue for Galaxies, and we will have to generate it.')
            print('This will take a long while ')

            Njack = nJack # this is amount of jacknife patches
            NoOfBins = NoOfBins
            MinSep = minSepGal #arcmin
            MaxSep = maxSepGal #arcmin
            catG,catGM=GiveMeTreeCorr(Njack,
                                 RA_G,
                                 DEC_G,
                                 rand_ra_GM,
                                 rand_dec_GM,
                                            w_G_left,
                                 NoOfBins,MinSep,MaxSep)
 

        print('\n AFTER TREECORR -- GALAXY')
        print(catG,'catG')
        print(catGM,'catGM')
 
        if os.path.isfile(loadName) == False: # if you don't have it, save it 
            # Only need to save catGM
    
            Name='catGM_'
            saveName = str(path+Name+zRange+'_TreeCorr_GalaxyRandom.npy')
            ToSave = np.array(catGM.patch_centers)  
            print(saveName,'saveName')
            np.save(saveName,ToSave) 
        
        # CLUSTERS -- will use catGM
        #---------------------------
        
        w_C = np.zeros(data_ra_restricted_C.shape[0])+1 # weights for clusters
        NoOfBins = NoOfBins
        MinSep = minSepCluster #arcmin
        MaxSep = maxSepCluster #arcmin
        ddC,rrC,drC,catC,catCM=self.GiveMeTreeCorr_w_Patches(catGM,data_ra_restricted_C,data_dec_restricted_C,
                       rand_ra_restricted_CM,rand_dec_restricted_CM,
                       w_C,NoOfBins,MinSep,MaxSep,
                       zRange,outputDir,outputData)
        
        print('\n AFTER TREECORR -- CLUSTER')
        print(ddC,'ddC')      # You need this
        print(catC,'catC')    # You need this
        print(catCM,'catCM')  # You need this
 
        Name='catC_'
        # Just the patch
        saveName = str(outputData+Name+zRange+'_TreeCorr_Cluster.npy')
        ToSave = np.array(catC.patch_centers)  
        print(saveName,'saveName')
        np.save(saveName,ToSave) 
        
        Name='catCM_'
        saveName = str(outputData+Name+zRange+'_TreeCorr_ClusterRandom.npy')
        ToSave = np.array(catCM.patch_centers)  
        print(saveName,'saveName')
        np.save(saveName,ToSave) 



        # GALAXIES AND CLUSTERS CORRELATE  -- will use catGM, catG, catCM, catC & ddC
        # ---------------------------------------------------------------------------

        #the cluster map, the cluster random map, and the patchcenters 
        Njack = nJack # this is amount of jacknife patches
        NoOfBins = NoOfBins
        MinSep = minSepCross #arcmin
        MaxSep = maxSepCross #arcmin
        # Naming convention after Baxter
        g_cat,gr_cat = catG,catGM 
        c_cat, cr_cat = catC,catCM
        catG,catGM, catC,catCM = 0,0,0,0 # clear some space
        dd,dr,rd,rr = self.GiveMeCrossCorrelationFINAL_G(ddC, 
                                      c_cat,cr_cat,
                                     NoOfBins,MinSep,MaxSep,
                                        g_cat,gr_cat,
                                     zRange,outputDir,outputData)

        print('\n AFTER TREECORR -- GALAXY-CLUSTER')
    
        print("\n TreeCorr ran !")
        return

    def GiveMeTreeCorr(self,Njack,
                             RA_G,DEC_G,
                             rand_ra_GM, rand_dec_GM,
                              w_G_left,
                             NoOfBins,MinSep,MaxSep,
                       zRange,outputDir,outputData):



        ''' THIS IS FOR GALAXIES ONLY: 
        Note that you should only do this ONCE, because catGM generated is actually random, 
        and because this process takes about 2 hrs! 
        This will give you the correlation and the cross-correlation.
        '''

        print('\n We are in TreeCorr for galaxies ')
        print(RA_G.shape,'RA_G.shape')
        print(DEC_G.shape,'DEC_G.shape')
        print(w_G_left.shape,'w_G_left.shape')
        print('These shapes must match!')

        print(rand_ra_GM.shape,'rand_ra_GM.shape')
        print(rand_ra_GM,'rand_ra_GM')
        print(rand_dec_GM.shape,'rand_dec_GM.shape')    
        print(rand_dec_GM,'rand_dec_GM')
        print(Njack,'Njack')     

        # The random map for masking does not need weight, & use the jacknife estimation for patches
        catGM = treecorr.Catalog(ra=rand_ra_GM,
                                 dec=rand_dec_GM,
                                 npatch=Njack,
                                 ra_units='deg', dec_units='deg')

        
        
        # The data map does need weight, and use the random map's patches for patches
        catG = treecorr.Catalog(ra=RA_G,
                                dec=DEC_G,
                                w=w_G_left,
                                patch_centers=catGM.patch_centers,
                              ra_units='deg', dec_units='deg')
        
    
     
        return(catG,catGM)
        
        
    def GiveMeTreeCorr_REF(self,Njack,data_ra_restricted_G,data_dec_restricted_G,
                   rand_ra_restricted_GM,rand_dec_restricted_GM,
                   w_G_left,NoOfBins,MinSep,MaxSep,
                   min_redshift_left_G,max_redshift_left_G,outputDir):
        ''' REF ONLY 
        This is for Galaxies ONLY.
        This will give you the correlation and the cross-correlation.
        Please note that you can only use this once. '''

        print('\n We are in TreeCorr for galaxies ... THIS MAKE TAKE A WHILE  ')
        print(data_ra_restricted_G.shape,'data_ra_restricted_G.shape')
        print(data_dec_restricted_G.shape,'data_dec_restricted_G.shape')
        print(w_G_left.shape,'w_G_left.shape')
        print('These shapes must match!')
        
        # The random map for masking does not need weight, & use the jacknife estimation for patches
        catGM = treecorr.Catalog(ra=rand_ra_restricted_GM, # w_col=w_G,
                                 dec=rand_dec_restricted_GM, 
                                 npatch=Njack,# use this for jacknife
                                ra_units='deg', dec_units='deg')

        # The data map does need weight, and use the random map's patches for patches
        catG = treecorr.Catalog(ra=data_ra_restricted_G,
                                dec=data_dec_restricted_G,w=w_G_left,
                                patch_centers=catGM.patch_centers,# use this for jacknife
                              ra_units='deg', dec_units='deg')
        
        ddG = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, 
                                     nbins=NoOfBins, bin_slop = 0.0, 
                                     var_method='jackknife',
                                     sep_units='arcmin')

        print(ddG,'ddG')
        print('bin_size = %.6f'%ddG.bin_size)
        ddG.process(catG)
        rrG = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins=NoOfBins,
                                     var_method='jackknife', 
                                     bin_slop = 0.0, sep_units='arcmin')
        print(rrG,'rrG')
        rrG.process(catGM)
        print('bin_size = %.6f'%rrG.bin_size)
        # Now use : the correlation function is the Landy-Szalay formula (dd-2dr+rr)/rr.
        drG = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins=NoOfBins,
                                     var_method='jackknife',
                                     bin_slop = 0.0,  sep_units='arcmin')
        drG.process(catG, catGM)
        
        
        # THIS IS THE CORRELATION FUNCTION -- GALAXIES -- DATA NOT SAVED BUT PLOT SAVED
        # WE DO NOT DO GALAXIES -- THIS IS JUST KEPT AS A REF.
        xiG, varxi = ddG.calculateXi(rr=rrG, dr=drG)
        sigG = np.sqrt(varxi)
        ddG_cov = ddG.cov  # Can access covariance now.
        rG = np.exp(ddG.logr)
        WhatIsIt='Galaxies'
        yAxis='ω'
        self.PlotMeAngularCorr(WhatIsIt,rG, xiG,sigG,
                          min_redshift_left_G,max_redshift_left_G,outputDir,yAxis)
        yAxis='ωθ'
        self.PlotMeAngularCorr(WhatIsIt,rG,rG*xiG,rG*sigG,
                          min_redshift_left_G,max_redshift_left_G,outputDir,yAxis)
     
        return(ddG,rrG,drG,catG,catGM)

        
    def GiveMeTreeCorr_w_Patches_REF(self,catGM,data_ra_restricted_C,data_dec_restricted_C,
                       rand_ra_restricted_CM,rand_dec_restricted_CM,
                       w_C,NoOfBins,MinSep,MaxSep,
                                 min_redshift_left_G,max_redshift_left_G,outputDir,outputData):
        '''
        REF ONLY
        This is for Clusters ONLY .
        This will give you the correlation and the cross-correlation.
        Includes the patch-centers, as you are lining this up with another cross-corelation.
        Please note that you can only use this once.'''
        print('\n We are in TreeCorr for cluster')
        print(data_ra_restricted_C.shape,'data_ra_restricted_C.shape')
        print(data_dec_restricted_C.shape,'data_dec_restricted_C.shape')
        print(w_C.shape,'w_C.shape')
        print('These shapes must match!')

        
        
        # The random map of clusters does not need weight ... 
        # ... but use the patches of the random map of galaxies !!!! 
        catCM = treecorr.Catalog(ra=rand_ra_restricted_CM, 
                                 dec=rand_dec_restricted_CM, 
                                 patch_centers=catGM.patch_centers,# use this for jacknife
                                ra_units='deg', dec_units='deg')

        # The random map of clusters does not need weight ... 
        # ... but use the patches of the random map of galaxies !!!!    
        catC = treecorr.Catalog(ra=data_ra_restricted_C,
                                dec=data_dec_restricted_C,w=w_C,
                                patch_centers=catGM.patch_centers,# use this for jacknife
                              ra_units='deg', dec_units='deg')
        
        ddC = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, 
                                     nbins=NoOfBins, bin_slop = 0.0, 
                                     var_method='jackknife',
                                     sep_units='arcmin')
        print(ddC,'ddC')
        print('bin_size = %.6f'%ddC.bin_size)
        ddC.process(catC)
        rrC = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins=NoOfBins,
                                     var_method='jackknife', 
                                     bin_slop = 0.0, sep_units='arcmin')
        rrC.process(catCM)
        print('bin_size = %.6f'%rrC.bin_size)
        # Now use : the correlation function is the Landy-Szalay formula (dd-2dr+rr)/rr.
        drC = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins=NoOfBins,
                                     var_method='jackknife',
                                     bin_slop = 0.0,  sep_units='arcmin')
        drC.process(catC, catCM)
        
        # THIS IS THE CORRELATION FUNCTION -- CLUSTER --
        xiC, varxi = ddC.calculateXi(rr = rrC, dr = drC)
        sigC = np.sqrt(varxi)
        ddC_cov = ddC.estimate_cov('jackknife')  # Can access covariance now.
        rC = np.exp(ddC.logr)
        WhatIsIt='Clusters'
        yAxis='ω'
        self.PlotMeAngularCorr(WhatIsIt,rC,
                                 xiC,sigC,min_redshift_left_G,max_redshift_left_G,outputDir,yAxis)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_theta_corr.npy"), rC)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_ω_corr.npy"), xiC)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_ω_corr_uncertainty_corr.npy"), sigC)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_ω_jacknife_covariant_matrix.npy"), ddC_cov)
        # Now  we also look into ωθ
        yAxis='ωθ'
        self.PlotMeAngularCorr(WhatIsIt,rC,rC*xiC,rC*sigC,min_redshift_left_G,
                                   max_redshift_left_G,outputDir,yAxis)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_thetaθ_corr.npy"), rC)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_ωθ_corr.npy"), rC*xiC)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_ωθ_corr_uncertainty_corr.npy"), rC*sigC)
        
        return(ddC,rrC,drC,catC,catCM)


    def GiveMeTreeCorr_w_Patches(self,catGM,data_ra_restricted_C,data_dec_restricted_C,
                       rand_ra_restricted_CM,rand_dec_restricted_CM,
                       w_C,NoOfBins,MinSep,MaxSep,
                                 zRange,outputDir,outputData):
        '''
        THIS IS FOR CLUSTERS ONLY
        This will give you the correlation and the cross-correlation.
        This one includes the patch-centers, as you are lining this up with another cross-corelation. '''
    
        print('\n We are in TreeCorr for cluster')
        print(data_ra_restricted_C.shape,'data_ra_restricted_C.shape')
        print(data_dec_restricted_C.shape,'data_dec_restricted_C.shape')
        print(w_C.shape,'w_C.shape')
        print('These shapes must match!')

        try:
            catGM_patch_centers = catGM.patch_centers # If catGM is actually the entire catalogue file 
        except:
            catGM_patch_centers = catGM  # If catGM is the catGM.patch_centers .npy file
        
        # The random map of clusters does not need weight ... 
        # ... but use the patches of the random map of galaxies !!!! 
        catCM = treecorr.Catalog(ra=rand_ra_restricted_CM,               # w_col=w_G,
                                 dec=rand_dec_restricted_CM, 
                                 patch_centers=catGM_patch_centers,       # use this for jacknife
                                ra_units='deg', dec_units='deg')
        
        # The random map of clusters does not need weight ... 
        # ... but use the patches of the random map of galaxies !!!!      
        catC = treecorr.Catalog(ra=data_ra_restricted_C,                  #,w_col=w_G,
                                dec=data_dec_restricted_C,w=w_C,
                                patch_centers=catGM_patch_centers,       #catGM.patch_centers,# use this for jacknife
                              ra_units='deg', dec_units='deg')
        
    
    
        ddC = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins=NoOfBins, bin_slop = 0.0, 
                                     var_method='jackknife',
                                     sep_units='arcmin')
        
        print(ddC,'ddC')
        print('bin_size = %.6f'%ddC.bin_size)
        ddC.process(catC,catC)
        
    
        rrC = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins=NoOfBins,
                                     var_method='jackknife', 
                                     bin_slop = 0.0, sep_units='arcmin')
        rrC.process(catCM,catCM)
        print(rrC,'rrC')
        print('bin_size = %.6f'%rrC.bin_size)
        
        # Now use : the correlation function is the Landy-Szalay formula (dd-2dr+rr)/rr.
        drC = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins=NoOfBins,
                                     var_method='jackknife',
                                     bin_slop = 0.0,  sep_units='arcmin')
        drC.process(catC, catCM)

                
        # THIS IS THE CORRELATION FUNCTION -- CLUSTER --
        xiC, varxi = ddC.calculateXi(rr = rrC, dr = drC)
        sigC = np.sqrt(varxi)
        ddC_cov = ddC.estimate_cov('jackknife')  # Can access covariance now.
        rC = np.exp(ddC.logr)
        WhatIsIt='Clusters'
        yAxis='ω'
        self.PlotMeAngularCorr(WhatIsIt,rC,
                                 xiC,sigC,zRange,outputDir,yAxis)
        np.save(str(outputData+zRange+"_theta_corr.npy"), rC)
        np.save(str(outputData+zRange+"_ω_corr.npy"), xiC)
        np.save(str(outputData+zRange+"_ω_corr_uncertainty_corr.npy"), sigC)
        np.save(str(outputData+zRange+"_ω_jacknife_covariant_matrix.npy"), ddC_cov)
        
        # Now  we also look into ωθ
        yAxis='ωθ'
        self.PlotMeAngularCorr(WhatIsIt,rC,rC*xiC,rC*sigC,zRange,outputDir,yAxis)
        np.save(str(outputData+zRange+"_thetaθ_corr.npy"), rC)
        np.save(str(outputData+zRange+"_ωθ_corr.npy"), rC*xiC)
        np.save(str(outputData+zRange+"_ωθ_corr_uncertainty_corr.npy"), rC*sigC)
        
        return(ddC,rrC,drC,catC,catCM)

        
    def GiveMeCrossCorrelationFINAL_G_REF(self,Njack,RA_G,DEC_G,rand_ra_GM,rand_dec_GM,
                                     w_G_left,ddC,
                                     c_cat, cr_cat,
                                     #data_ra_restricted_C,data_dec_restricted_C,
                                     #rand_ra_restricted_CM,rand_dec_restricted_CM,
                                     NoOfBins,MinSep,MaxSep,g_cat,gr_cat,
                                     min_redshift_left_G,max_redshift_left_G,outputDir,outputData):
        ''' REF ONLY
        It uses the galaxies as the  patch centers 
        Now use : the correlation function is the Landy-Szalay formula :
        (dGdC - dGrC - dCrR + rRrC)/rRrC.'''
        print("Running TreeCorr on cross corelate clusters and galaxies...")
        
        # The cluster data uses the random galaxies maps as patches    
        #c_cat = treecorr.Catalog(ra = data_ra_restricted_C, dec = data_dec_restricted_C, 
        #                         patch_centers=gr_cat.patch_centers,
        #                         ra_units='degrees', dec_units='degrees')
        
        # The cluster random alo uses the random galaxies maps as patches
        #cr_cat = treecorr.Catalog(ra = rand_ra_restricted_CM, dec = rand_dec_restricted_CM,
        #                          patch_centers=gr_cat.patch_centers,
        #                          ra_units='degrees', dec_units='degrees')

        dd = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins = NoOfBins, 
                                    var_method='jackknife',sep_units='arcmin', bin_slop = 0.0)
        dr = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins = NoOfBins, 
                                    var_method='jackknife',sep_units='arcmin', bin_slop = 0.0)
        rd = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins = NoOfBins, 
                                    var_method='jackknife',sep_units='arcmin', bin_slop = 0.0)
        rr = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins = NoOfBins, 
                                    var_method='jackknife',sep_units='arcmin', bin_slop = 0.0)

        # Make the cross correlation
        dd.process(c_cat, g_cat) # cluster data <--> galaxy data
        dr.process(c_cat, gr_cat)# cluster data <--> galaxy random
        rd.process(cr_cat, g_cat)# cluster data <--> galaxy random
        rr.process(cr_cat, gr_cat)# cluster random <--> galaxy random
        
        # THIS IS THE CORRELATION FUNCTION: SAVE HER 
        xiGC, varxi = dd.calculateXi(rr = rr, dr = dr, rd = rd)
        sigCGC = np.sqrt(varxi)
        dGdC_cov = dd.estimate_cov('jackknife')  # Can access covariance now.
        rGC = np.exp(dd.logr)

        # ddC : cluster data (with random galaxy patches)
        # dd : cluster data (with random galaxy patches) <--> galaxy data (with random galaxy patches)
        cc_cg_cov = treecorr.estimate_multi_cov([ddC,dd],'jackknife')

        
        WhatIsIt='Clusters+Galaxies'
        yAxis='ω'
        self.PlotMeAngularCorr(WhatIsIt,rGC, xiGC,sigCGC,min_redshift_left_G,
                            max_redshift_left_G,outputDir,yAxis)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_theta_corrGC.npy"), rGC)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_ω_corrGC.npy"), xiGC)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_ω_corr_uncertainty_corrGC.npy"), sigCGC)
        
        # DO NOT USE THE FOLLOWING ANYMORE
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_ω_jacknife_covariant_matrixGC.npy"), dGdC_cov)
        
        # This is the FULL Covariance matrix: This is being used for the MCMC 
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_FULL_covariant_matrixGC.npy"), cc_cg_cov)

        
        WhatIsIt='Clusters+Galaxies'
        yAxis='ωθ'
        self.PlotMeAngularCorr(WhatIsIt,rGC,rGC*xiGC,rGC*sigCGC,
                            min_redshift_left_G,max_redshift_left_G,outputDir,yAxis)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_thetaθ_corrGC.npy"), rGC)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_ωθ_corrGC.npy"), rGC*xiGC)
        np.save(str(outputData+str(min_redshift_left_G)+'_to_'+
                    str(max_redshift_left_G)+"_ωθ_corr_uncertainty_corrGC.npy"), rGC*sigCGC)
            
        return(dd,dr,rd,rr)     
        
    def GiveMeCrossCorrelationFINAL_G(self,ddC, 
                                      c_cat,cr_cat,
                                     NoOfBins,MinSep,MaxSep,g_cat,gr_cat,
                                     zRange,outputDir,outputData):
        ''' 
        It uses the galaxies as the  patch centers 
        Now use : the correlation function is the Landy-Szalay formula :
        (dGdC - dGrC - dCrR + rRrC)/rRrC.'''
        print('\n BEFORE TREECORR -- GALAXY-CLUSTER')
        print('We are now going to cross-correlate the clusters <--> galaxies. This will take a few seconds ... ')
        
        dd = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins = NoOfBins, 
                                    var_method='jackknife',sep_units='arcmin', bin_slop = 0.0)
        dr = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins = NoOfBins, 
                                    var_method='jackknife',sep_units='arcmin', bin_slop = 0.0)
        rd = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins = NoOfBins, 
                                    var_method='jackknife',sep_units='arcmin', bin_slop = 0.0)
        rr = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins = NoOfBins, 
                                    var_method='jackknife',sep_units='arcmin', bin_slop = 0.0)
        
        
        # Make the cross correlation
        dd.process(c_cat, g_cat) # cluster data <--> galaxy data
        dr.process(c_cat, gr_cat)# cluster data <--> galaxy random
        rd.process(cr_cat, g_cat)# cluster data <--> galaxy random
        rr.process(cr_cat, gr_cat)# cluster random <--> galaxy random

        # THIS IS THE CORRELATION FUNCTION: CLUSTER <--> GALAXY
        xiGC, varxi = dd.calculateXi(rr = rr, dr = dr, rd = rd)
        sigCGC = np.sqrt(varxi)
        dGdC_cov = dd.estimate_cov('jackknife')  # Can access covariance now.
        rGC = np.exp(dd.logr)
        
        # THIS IS IT ! THIS IS THE COVARIANCE MATRIX THAT DOES INTO MCMC 
        # ddC : cluster data (with random galaxy patches)
        # dd : cluster data (with random galaxy patches) <--> galaxy data (with random galaxy patches)
        cc_cg_cov = treecorr.estimate_multi_cov([ddC,dd],'jackknife')
        print(cc_cg_cov, 'This is the important cluster-cluster & cluster-galaxy covariance matrix')

        
        WhatIsIt='Clusters+Galaxies'
        yAxis='ω'
        self.PlotMeAngularCorr(WhatIsIt,rGC, xiGC,sigCGC,zRange,outputDir,yAxis)
        np.save(str(outputData+zRange+"_theta_corrGC.npy"), rGC)
        np.save(str(outputData+zRange+"_ω_corrGC.npy"), xiGC)
        np.save(str(outputData+zRange+"_ω_corr_uncertainty_corrGC.npy"), sigCGC)
        
        # DO NOT USE THE FOLLOWING ANYMORE: Just save it for REF 
        np.save(str(outputData+zRange+"_ω_jacknife_covariant_matrixGC.npy"), dGdC_cov)
        
        # This is the FULL Covariance matrix: This is being used for the MCMC 
        np.save(str(outputData+zRange+"_FULL_covariant_matrixGC.npy"), cc_cg_cov)

        
        WhatIsIt='Clusters+Galaxies'
        yAxis='ωθ'
        self.PlotMeAngularCorr(WhatIsIt,rGC,rGC*xiGC,rGC*sigCGC,zRange,outputDir,yAxis)
        np.save(str(outputData+zRange+"_thetaθ_corrGC.npy"), rGC)
        np.save(str(outputData+zRange+"_ωθ_corrGC.npy"), rGC*xiGC)
        np.save(str(outputData+zRange+"_ωθ_corr_uncertainty_corrGC.npy"), rGC*sigCGC)
         
        return(dd,dr,rd,rr)     
    
    def PlotMeAngularCorr(self,WhatIsIt,rC, xiC,sigC,zRange,outputDir,yAxis): 
        #Is it clusters or galaxies? 
        if WhatIsIt=='Clusters':
            sideColor = 'blue'
            mainColor = 'orange'

        elif WhatIsIt=='Galaxies':
            sideColor = 'pink'
            mainColor = 'cyan' 

        elif WhatIsIt=='Clusters+Galaxies':
            sideColor = 'limegreen'
            mainColor = 'red' 

        #PLOT IT LIKE IT'S HOT 
        f, (ax1) = plt.subplots(1,  figsize=(15,10))
        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white' 
        plt.plot(rC, xiC, color=mainColor,lw=3,label=r'$w(\theta)$')
        print(rC,'rC')
        print(xiC,'xiC')
        plt.plot(rC, -xiC, color=mainColor, ls='--',lw=3,label=r'$-w(\theta)$')
        plt.errorbar(rC[xiC>0], xiC[xiC>0], yerr=sigC[xiC>0], color=sideColor, 
                     lw=1, ls='',label='error')
        plt.errorbar(rC[xiC<0], -xiC[xiC<0], yerr=sigC[xiC<0], color=sideColor, 
                     lw=1, ls='',label='-error')
        leg = plt.errorbar(-rC, xiC, yerr=sigC, color=mainColor)

        plt.xscale('log')
        #plt.yscale('log', nonposy='clip')
        plt.yscale('log', nonpositive='clip')
        plt.xlabel('arcmins')
        plt.ylabel(r'$w(\theta)$')
        TITLE=str('AngularCorrelation--'+yAxis+'_'+WhatIsIt+'_with z_'+zRange)
        plt.title(TITLE)
        plt.legend([leg], [r'$w(\theta)$'], loc='lower left')
        plt.xlim([2.5,250])
        plt.axvline(66,linewidth=2, color='g',ls=":")
        plt.axvline(18,linewidth=2, color='g',ls=":")
        plt.legend(facecolor="white",loc='lower center')

        saveName=str(outputDir+TITLE+'.pdf')
        plt.savefig(saveName)
        
        
class Limber :
    """
    This class prepares the Limber Approximation for you to run CAMB, and also runs the 
    Angular Power Spectrum 
    """
    def __init__(self):
        """
        """ 
  
    def step3_runCAMB(self,path,zRange,outputDir,outputData,nbinsCluster,nbinsGalaxy,
                     AS,OMBH2,OMCH2,lspace,photoShiftAndStretch,cosmos):
        """
        Runs the Limbe approximation for you via CAMB
        """
        print(zRange,'is the redshift range')
        print(outputDir,' is the outputDirectory. ',outputData,'is the outputData.')

        
        """
        Get the data
        """     
        #Load the data After Cut and HealPix
        LoadHer = str(outputData+'/RelevantDataAfterCut+HealPix')
        with open(LoadHer, 'rb') as f: 
            myList = pickle.load(f)                     
        [RA,DEC,redshift_left,rand_ra_M,rand_dec_M,rand_z_M_left,\
         RA_G,DEC_G,redshift_left_G,w_G_left,\
         rand_ra_GM,rand_dec_GM,rand_z_GM_left,
         min_redshift_left_G,max_redshift_left_G,
         rand_ra_restricted_CM,\
         rand_dec_restricted_CM,\
         data_ra_restricted_C,\
         data_dec_restricted_C,\
         data_pix_indices_C,\
         rand_pix_indices_CM] = myList   
        
        # The weights for bins ( for limber Approximation ) 
        filename='2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits'
        weights = get_pkg_data_filename(path+filename)
        image_weights = fits.getdata(weights, ext=7)
        print(image_weights.shape,'GALAXIES WEIGHTS')
        nW,zW=self.GimmeWeights(image_weights,zRange)
        
        """
        Prepare the Bins for Limber 
        """         
        # Makes and saves the weighted bins for you
        nC,binsC,patchesC,\
        nG,binsG,patchesG = self.prepareBins(nbinsCluster,nbinsGalaxy,\
                                             RA,redshift_left,data_ra_restricted_C,\
                                                RA_G,redshift_left_G,rand_ra_GM,\
                                                      zRange,\
                                                      outputDir)
        print(nC,'nC')
        print(binsC,'binsC')
        print(nG,'nG')
        print(binsG,'binsG')
        
        """
        Prepare the Weights for Limber 
        """
        Z11,W11,Z1,dzBins1,W1,zStart,zEnd = self.WeightingsProbability(nC,binsC,nW,zW)
        print(Z1.shape,'Z1.shape: These are the old ones ')
        print(W1.shape,'W1.shape: These are the old ones')
        print(Z11.shape,'Z11.shape: These are the old ones ... these do not really change' )
        print(W11.shape,'W11.shape: These are the old ones ... these do not really change')
    
        """
        Set CAMB up.  
        """
        #Get ready for the Limber Integral for CAMB 
        print("Getting ready for the Limber Integral ")
        # Let us get camb ready
        set_matplotlib_formats('retina')
        #Assume installed from github using 
        # "git clone --recursive https://github.com/cmbant/CAMB.git"
        camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
        print(sys.path.insert(0,camb_path))
        print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))
        print(lspace,'lspace')
        print(lspace.shape,'lspace.shape')
        
        """
        Interpolation: fixing the ends 
        """
        f1,f11,znew,W1,Z1 = self.prepareInterpolation(Z11,W11,Z1,W1,
                                                         zStart,zEnd,
                                                         min_redshift_left_G,
                                                         max_redshift_left_G,
                                                          zRange,
                                                         outputDir,outputData,cosmos)
        print(znew.shape,'znew.shape')
        print(f1(znew).shape,'f1(znew).shape')
        print(f11(znew).shape,'f11(znew).shape') 
        print(Z1.shape,'Z1.shape: These are the new ones ')
        print(W1.shape,'W1.shape: These are the new ones')
        print(Z11.shape,'Z11.shape: These are the new ones ... these do not really change' )
        print(W11.shape,'W11.shape: These are the new ones ... these do not really change')

        """
        Finally, make the interpolation
        """       
        znew,zBins,normW1,normW11 = self.makeTheInterpolation(Z11,W11,Z1,W1,zStart,
                                                                     zEnd,znew,f1,f11,)
        print(znew,'znew')
        print(zBins,'zBins')
        print(normW11,'normW11')
        print(normW11.shape,'normW11.shape')
        print(znew.shape)
        print(zBins.shape)
        
        # You need the photoshift and stretch to save this correctly
        Δz,σz,Wg_new = check().checkPhotoShiftAndStretch(zRange,nside,
                                          outputData,photoShiftAndStretch)
        print(Δz,σz,'Δz,σz')
        print(Wg_new,'Wg_new: The new weights for the galaxies in the Limber Integral')
        print(normW11,'normW11: These are the old weights DO NOT USE !  ')

        '''
        THIS IS WHEN Limber Approximation HAPPENS.
        '''
        CL_cng,CL_cnc = self.makeMeLimber(lspace,znew,zBins,normW1,Wg_new,
                                                AS,OMBH2,OMCH2,zRange,cosmos)
         # It's done ! Now here's the plot
        self.ShowMeCLvsL(lspace,CL_cng,CL_cnc,cosmos,outputDir,zRange)

        print("CAMB ran ! Now,let us run the Angular Power Spectrum")
        ################################################################################
        ''' OBSERVATION : angular correlation ,which we already have. !!!!'''
        #GALAXIES -- CLUSTER
        rGC = np.load(str(outputData+zRange+"_theta_corrGC.npy"))
        xiGC = np.load(str(outputData+zRange+"_ω_corrGC.npy"))
        sigCGC = np.load(str(outputData+zRange+"_ω_corr_uncertainty_corrGC.npy"))
        
        print(rGC,'rGC',xiGC,'xiGC',sigCGC,'sigCGC')
        
        #CLUSTER ONLY 
        rC = np.load(str(outputData+zRange+"_theta_corr.npy"))
        xiC=np.load(str(outputData+zRange+"_ω_corr.npy"))
        sigC = np.load(str(outputData+zRange+"_ω_corr_uncertainty_corr.npy"))
        print(rC,'rC',xiC,'xiC',sigC,'sigC')
        
        """ Just get it into right words. """
        #The radial distance
        θ = rC
        theta_corr=rC        
        #The angular correlation--cluster
        ω_corr = xiC
        #The uncertainty correlation - cluster
        sigC  =  sigC
        #The angle -- cluster & galaxy
        theta_corrGC =  rGC
        #The angular correlation  -- cluster & galaxy
        ω_corrGC = xiGC
        #The uncertainty correlation -- cluster & galaxy
        sigCGC = sigCGC 
        #The radial distance '''
        print(θ.shape,'θ.shape')

        ''' 
        THEORY :  now it  is  time to  get the  angular correlation  !!
        Finally, these are the two main things you need : these are the
        angular correelatioon functionn
        '''
        print('\n These are the cosmology that we use')
        print(cosmos,cosmos)
        print(AS,OMBH2,OMCH2,'AS,OMBH2,OMCH2\n')
        ω_cnc = self.Angular(lspace,CL_cnc,θ)
        print(ω_cnc,'ω_cnc') 
        print(ω_cnc.shape,'ω_cnc.shape') 
        
        ω_cng = self.Angular(lspace,CL_cng,theta_corrGC)
        print(ω_cng,'ω_cng') 
        print(ω_cng.shape,'ω_cng.shape') 
        

        """
        Finally, let us save the angular correlation.
        """
        # First let us save cnc
        corrName = '_ω_cnc'
        saveData = ω_cnc
        self.SavingAngularCorrelation(corrName,saveData,Δz,σz, AS,OMBH2,OMCH2,zRange,outputData)
        
        # Then let us save cng
        corrName = '_ω_cng'
        saveData = ω_cng
        self.SavingAngularCorrelation(corrName,saveData,Δz,σz, AS,OMBH2,OMCH2,zRange,outputData)
        """
        Last thing is just to plot the angular correlation
        """        

        yAxis='ω'
        self.ShowMeAngular(θ,ω_cng,ω_cnc,theta_corr,ω_corr,sigC,
                         theta_corrGC,ω_corrGC,sigCGC,
                         cosmos,outputDir,zRange,yAxis)
    
        ''' Now, we just look at wθ'''
        #The angle  - CLUSTER
        thetaθ_corr =rC
        #The angular correlation- CLUSTER 
        ωθ_corr   = rC*xiC
        #The uncertainty correlation- CLUSTER
        sigCθ = rC*sigC
        #The angle - CLUSTER  GALAXY
        thetaθ_corrGC = rGC
        #The angular correlation - CLUSTER  GALAXY 
        ωθ_corrGC =  rGC*xiGC
        #The uncertainty correlation  - CLUSTER  GALAXY 
        sigCθGC  = rGC*sigCGC
        yAxis='ωθ'
        self.ShowMeAngular(θ,ω_cng*thetaθ_corrGC,ω_cnc*θ,thetaθ_corr,ωθ_corr,sigCθ,
                 thetaθ_corrGC,ωθ_corrGC,sigCθGC,
                 cosmos,outputDir,zRange,yAxis)
        print('\n')
        print(Δz,σz,'Δz,σz')
        print(AS,OMBH2,OMCH2,'AS,OMBH2,OMCH2')
        print("Angular Power Spectrum also ran ! ")        
        return 
    
    def ShowMeAngular(self,θ,ω_cng,ω_cnc,theta_corr,ω_corr,sigC,
                     theta_corrGC,ω_corrGC,sigCGC,
                     cosmos,outputDir,zRange,yAxis):

        ''' The following us the angular correlation from TreeCorr'''
        f, (ax1) = plt.subplots(1,  figsize=(15,10))
        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white' 
        theta_corr=theta_corr
        theta_corrGC = theta_corrGC
        # This is the one from TreeCorr
        plt.plot(theta_corr,ω_corr,label="TreeCorrC measurement: + ",c='lightgray',lw=6,alpha=0.5)
        plt.plot(theta_corr,-ω_corr,label="TreeCorrC measurement: -",c='lightgray',
                 ls="--",lw=1,alpha=0.5)
        plt.errorbar(theta_corr,ω_corr, yerr=sigC,c='lightgray', lw=6,alpha=0.5)
        plt.errorbar(theta_corr,-ω_corr,yerr=sigC,c='lightgray', lw=6,alpha=0.5)
        plt.plot(theta_corrGC,ω_corrGC,
                 label="TreeCorrGC measurement: + ",c='lightgray',lw=3,alpha=0.5)
        plt.plot(theta_corrGC,-ω_corrGC,label="TreeCorrGC  measurement: -",c='lightgray',
                 ls="--",lw=1,alpha=0.5)
        plt.errorbar(theta_corrGC,ω_corrGC, yerr=sigCGC,c='lightgray', lw=3,alpha=0.5)
        plt.errorbar(theta_corrGC,-ω_corrGC,yerr=sigCGC,c='lightgray', lw=3,alpha=0.5)
        #  My DATA
        plt.plot(theta_corrGC,ω_cng,label= str(cosmos+'_model: clusters + gal'),
                    c='cyan',lw=6)
        plt.plot(θ,ω_cnc,label=str(cosmos+'_model: clusters + clusters'),
                    c='orange',lw=6)
        
        plt.xscale('log')
        plt.yscale('log')
        TITLE  = str('Angular_Correlation:'+str(cosmos)+'_'+yAxis
                     +'_with_z_'+zRange)
        plt.title(TITLE)
        plt.xlabel('arcmins')
        plt.ylabel(r'$\omega$ (degrees')
        plt.xlim([2.5,250])
        plt.axvline(18,linewidth=2, color='g',ls=":")
        plt.axvline(66,linewidth=2, color='g',ls=":")
        legend = plt.legend(facecolor="white",loc='upper left')
        legend.get_frame().set_alpha(0.5)
        #legend.get_frame().set_facecolor((1.0, 1.0, 0.1, 0.1))

        saveName=str(outputDir+'THEORY_'+str(cosmos)+'_'+yAxis+'_with_z_'+zRange+'.pdf')
        plt.savefig(saveName)
        #plt.show()
    
    def SavingAngularCorrelation(self,corrName,saveData,Δz,σz, AS,OMBH2,OMCH2,zRange,outputData):
        ''' 
           The calculations are done, and here, we are just saving it to the correct names
        '''

        if photoShiftAndStretch == 'SHIFTnSTRETCH':
            #Shift and stretch of photometric redshift 
            if Δz !=0 and σz == 1: # only stretch
                title = str(outputData+'THEORY_'+str(cosmos)+'_photoStretch_'+zRange+corrName+'.npy')
            
            elif Δz !=0 and σz != 1: #  stretch & shift       
                title = str(outputData+'THEORY_'+str(cosmos)+'_photoShiftAndStretch_'+zRange+corrName+'.npy')
            
        else:
            # This is the plank data with no additional modification 
            if AS == 2.1005829e-9 and OMBH2==0.022383 and OMCH2==0.12011: #PLANCK2022
                title = str(outputData+'THEORY_'+str(cosmos)+'_'+zRange+corrName+'.npy')  # This is the plank data with not additional modification 
    
            #REF elif AS == 2.1005829e-9 and OMBH2==0.022383 and OMCH2==0.12011:#PLANCK2022 # This is old 
            #    title = str(outputData+str(min_redshift_left_G)+
            #                     '_to_'+str(max_redshift_left_G)+corrName+
            #                '_PLANCK2022_mod'+'.npy')
            
            elif AS == 2.1005829e-9*1.05:  # 5% INCREASE as 
                title = str(outputData+'THEORY_'+str(cosmos)+'_AS_5percent_'+zRange+corrName+'.npy')
    
            elif OMBH2 == 0.022383*1.05 and OMCH2 ==0.12011*1.05:  # 5% INCREASE Omeega maater
                title = str(outputData+'THEORY_'+str(cosmos)+'_OmegaM_5percent'+zRange+corrName+'.npy')
      
            print(title,'title') 
            np.save(title, saveData)

   
    def Angular(self,lspace0,CL0,θ):
        ''' 
        INPUT:
            You already have the CL & the Ls
        OUTPUT:
             GETTING ω (θ) !!
        VIA:
            We are largely doing the Legendre Polynomials
        '''

        ω = np.zeros(1)
        try:
            CL0 = CL0.value # get rid of the units
        except:
            CL0 = CL0

        '''The angle in the correlation function: 
        We will literally compare everthing to this
        '''
        print(lspace0.shape,'lspace0.shape')
        print(CL0.shape,'CL0.shape')
        #LOW
        low = int(np.ceil(lspace0)[0])
        #print(low,'low')
        # Add these numbers to the front to complete the interpolation
        AddTheseNumsXlo = np.arange(0,low,1)
        AddTheseNumsYlo = np.zeros(low)
        #print(AddTheseNumsXlo,'AddTheseNumsXlo')
        #print(AddTheseNumsYlo,'AddTheseNumsYlo')
        #LOW
        hi = int(np.ceil(lspace0)[-1])
        #print(hi,'hi')
        # Add these numbers to the BACK to complete the interpolation
        try:
            AddTheseNumsXhi = np.arange(hi,30001,1)
            AddTheseNumsYhi = np.zeros(30001-hi)
        except:
            AddTheseNumsXhi = np.arange(30000,hi,1)
            AddTheseNumsYhi = np.zeros(hi-30000)

        # This is just interpolation 
        lspace0 = np.insert(lspace0,0,AddTheseNumsXlo)
        CL0 = np.insert(CL0,0,AddTheseNumsYlo)
        lspace0 = np.append(lspace0,AddTheseNumsXhi)
        CL0 = np.append(CL0,AddTheseNumsYhi)
        f = interpolate.interp1d(lspace0,CL0)
        lspace_new = np.arange(1,30001,1)
        CL_new = f(lspace_new)
        multipole = CL_new.shape[0] 
        ωList=[]
        ###############################################################
        for j in θ:
        ##############################################################
            theta = (j/60)*np.pi/180 #[arcmin-->degree-->radians]
            X = np.cos(theta)
            # You have to get the range of coefficient's first 
            CoefficientRange = np.array([0])
            for i in range(3000):
                l = lspace_new[i]
                # SUMMING IT 
                Coefficient = CL_new[i]* ((2*l +1)/(4*np.pi) )
                CoefficientRange = np.append(CoefficientRange,Coefficient)


            # get the range of the x, which is the same acorrss all lsc
            Xrange = np.zeros(multipole) + X
            #print(Xrange,'Xrange')
            CoefficientRange = np.delete(CoefficientRange,(0),axis=0)
            # You legendre polynomials
            Legendre = np.polynomial.legendre.legval(x=Xrange,c=CoefficientRange) 
            # Your sum
            SUM = np.sum(Legendre[0])
            #print(SUM,'SUM')
            ωList.append(SUM)
            #print('\n')

        ω=np.asarray(ωList)
        print(ω,'ω') 
        print(ω.shape,'ω.shape') 
        
        return(ω)
    
    
    def makeTheInterpolation(self,Z11,W11,Z1,W1,zStart,zEnd,znew,f1,f11,):
        """
        Making the interpolatioin finally !!!
        """
        # Finally do the initerpolation on the fake data you creatteed 
        W1 = f1(znew)
        W11 =f11(znew)
        print(W11,'W11=f11(znew) W11 ... now this has changed')
        
        zBins = np.diff(znew)
        zBins = np.insert(zBins,0,zBins[0])
        # Makee sure that the interpolated area is still normalized
        #CLUSTER
        area = np.sum(W1*zBins)
        print(area,'Is this area normalized ? -- it should be !!!')
        normW1 = W1/area
        #GALAZY
        area = np.sum(W11*zBins)
        print(area,'Is this area normalized ? -- it should NOT be !!') 
        normW11 = W11/area
        
        #After Interpolaltion -- for checking
        myList = [Z11,W11,Z1,W1,zStart,zEnd,f11,f1,zBins,znew] 
        SaveHer = str(outputData+'/InterPolated_Weights')
        with open(SaveHer, 'wb') as f: 
            pickle.dump(myList, f) 
            
        return(znew,zBins,normW1,normW11)
    
    
    def makeMeLimber(self,lspace,znew,zBins,normW1,Wg_new,AS,OMBH2,OMCH2,
                    zRange,cosmos):
        """
        This makes the interpolation for you , for both cluster cluster and galaxies-cluster
        """    
    
        "CLUSTERS-CLUSTERS"
        CL_cnc = self.LIMBER_APPROX(lspace,znew,zBins,normW1,normW1,cosmos,AS,OMBH2,OMCH2)
        CL_nonValue = CL_cnc.value
        title = str(outputData+'THEORY_'+str(cosmos)+'_'+zRange+'_CL_cnc'+'.npy')
        np.save(title, CL_nonValue)

        
        "GALAXY-CLUSTERS"
        #  Just making sure we dont lose points 
        CL_cng = self.LIMBER_APPROX(lspace,znew,zBins,normW1,Wg_new,cosmos,AS,OMBH2,OMCH2)
        CL_nonValue = CL_cng.value
        title = str(outputData+zRange+'_CL_cng'+'.npy')
        title = str(outputData+'THEORY_'+str(cosmos)+'_'+zRange+'_CL_cng'+'.npy')
        np.save(title, CL_nonValue)
        
        return CL_cng,CL_cnc
    
    def prepareInterpolation(self,Z11,W11,Z1,W1,zStart,zEnd,min_redshift_left_G,\
                             max_redshift_left_G,zRange,outputDir,outputData,cosmos):
        """
        This prepares the ends for interpolation, sets up thhe interpolating function and
        normalilzes thee weight for you. 
        It also saves the weights for you:
                InterPolated_Weights.npy,  which you could use to check the shift and stretch
        
        """
        print('This is how far off the points are , hence you need to interpolate them ')
        print(Z1.shape)
        print(Z1,'Z1 from above')
        print(W1,'W1 from above')

        # We make new redshift and weights 
        Z1 = np.concatenate(([0,float(min_redshift_left_G)], Z1, [float(max_redshift_left_G)]))
        W1 = np.concatenate(([W1[0],W1[0]], W1, [W1[-1]]))
        print(Z1,'Z1: These are the new ones ')
        print(W1,'W1: These are the new ones ')
        
        # Interpolation through a new redshift range  
        numZ  = 100 # Numberr of points to interpolate , keep it in here for now
        znew = np.linspace(float(min_redshift_left_G),float(max_redshift_left_G),
                           num=numZ, endpoint=True)
        print(min_redshift_left_G,max_redshift_left_G,'min_redshift_left_G,max_redshift_left_G,')
        print(znew,'znew -- the x-axis that we will interpolate over')


        # This is the function that will make the interpolation later 
        # NOTE THAT znew  MUST BE WITHIN Z11[1:] or Z1[:]
        # 1: Clusters, 11: Galaxies
        f11 = interp1d(Z11[1:],W11, kind='linear',fill_value=(float(min_redshift_left_G),\
                                                              float(max_redshift_left_G)),
                       bounds_error=False)
        f1 = interp1d(Z1[1:],W1, kind='linear',fill_value=(float(min_redshift_left_G),\
                                                              float(max_redshift_left_G)),
             bounds_error=False)
        
        # This plots it out as a check
        self.ShowMeInterpolatedRedshiftWeighted(Z1,Z11,f11,f1,znew,W1,W11,
                               min_redshift_left_G,max_redshift_left_G,zRange,outputDir,cosmos)
        print('# 1: Clusters, 11: Galaxies')
        print(znew.shape,'znew.shape')
        print(f1(znew).shape,'f1(znew).shape')
        print(f11(znew).shape,'f11(znew).shape')
        print(Z11,'Z11[1:] ... new ones')
        print(Z1[1:],'Z1[1:] ... new ones')
        
        print('A histogram of the interpolated clusters & galaxies vs redshift is made!')
        return(f1,f11,znew,W1,Z1)

            
            
    def LIMBER_APPROX(self,lspace,Z1,dzBins1,W1,W11,cosmos,AS,OMBH2,OMCH2):
        '''
        LIMBER APPROXIMATION
        INPUT : lspace : You define the multipole in the step above.
                Z, dzBins, W from previous work on TreeCorr
                Saving Name : Just name it 
                cosmos : you decide the cosmological param
                            "Planck22" .. fixed at this valaue 
         '''
        print("Running the limber appoximation...")
        # Set thee parameteres for CAMBB
        pars = camb.CAMBparams()
        ####################Planck22####################
        if cosmos == "Planck22": # This is Planck 2018
            pars.set_cosmology(H0=67.45, ombh2=OMBH2, omch2=OMCH2)
            pars.InitPower.set_params(ns=0.965,As=AS)
        ################################################
        pars.NonLinear = model.NonLinear_none
        PK = camb.get_matter_power_interpolator(pars,nonlinear=True, 
                                                hubble_units=False,k_hunit=False)
        CL=np.zeros(1) #CL = CL*u.km/u.Mpc/u.Mpc/u.Mpc/u.s # rigt units
        Integral=0
        zarray=Z1#[1:]
        for l in lspace:
            '''Cosmological parameters'''
            # uses the z value in the Normalised_dc_bins.npy
            h = cosmo.h
            χ = cosmo.comoving_distance(zarray)
            H = cosmo.H(zarray) 
            da = cosmo.angular_diameter_distance(zarray)
            da2 = χ*χ
            W2 = W1*W11
            b2 = 1 # assume the bias is just 1
            
            # The POWER SPECTRUM that you want for you k and your z 
            P = np.zeros(1)*u.Mpc*u.Mpc*u.Mpc
            for i in range(zarray.shape[0]):
                k=(l+1/2)/(χ)   #*h) #k=k#*u.Mpc # get rid of the units 
                # Calculating the Power Spectrum 
                P_i = np.array([ PK.P( zarray[i],k[i].value)])*u.Mpc*u.Mpc*u.Mpc  #*h**3
                P =np.append(P,P_i) 
            P=np.delete(P,(0),axis=0)
            c= 299792*u.km/u.s
            # THIS IS THE INTERGRATION VIA SIMPSON'S RULEs
            Area = dzBins1*W2*(H/da2)*P*b2    /(c)
            Integral = np.sum(Area)
            CL = np.append(CL,Integral)
        CL = np.delete(CL,(0),axis=0)
        #print(CL,'CL')

        return(CL)
    
    
    def ShowMeCLvsL(self,lspace,CL_cng,CL_cnc,cosmos,outputDir,zRange):
        '''  This will show you the results of the LIMBER  approximation''' 
        f, (ax) = plt.subplots(1,  figsize=(12,12))
        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white' 
        plt.plot(lspace,CL_cng,label= str(cosmos +' clusters + galaxy'),
                 c='cyan',lw=10)
        plt.plot(lspace,CL_cnc,label=str(cosmos +' clusters + cluster'),
                 c='orange',lw=10)
        plt.style.context('Solarize_Light0')
        plt.xlabel('L')
        plt.ylabel('Cl')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-9,1e-4)
        title = str(zRange+
                    '--CLvsL '+str(cosmos))
        plt.title(title)
        plt.legend(facecolor="white",loc='lower center')
        saveName=str(outputDir+'THEORY_'+str(cosmos)+'_CLvsL_with_z_'+zRange+'.pdf')
        plt.savefig(saveName)

        #plt.show()
    
    def ShowMeInterpolatedRedshiftWeighted(self,Z1,Z11,f11,f1,znew,W1,W11,
                                   min_redshift_G,max_redshift_G,zRange,outputDir,cosmos):
        ''' It is time to plot out the Interpolated redshift  '''

        f, (ax) = plt.subplots(1,  figsize=(15,10))
        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white' 
        
        

        #NOW WE WILL INTERPOLLATE
        ax.plot(znew, f1(znew), '--',color='red',label='masked clusters: interpolated')
        ax.plot(znew, f11(znew), '--',color='cyan',label='masked galaxies: interpolated')

        ax.scatter(znew, f1(znew),color='red')
        ax.scatter(znew, f11(znew),color='cyan')

        # The  hard limit to the redshift 
        plt.axvline(float(min_redshift_G),c='purple', ls=":")
        plt.axvline(float(max_redshift_G),c='purple', ls=":")

        plt.xlabel('redshift')
        plt.ylabel('NORMALIZED # of galaxies and clusters -- INTERPOLATED')
        #plt.xlim(min_redshift_G,max_redshift_G)
        #lgnd1=ax.legend(facecolor="white",loc='lower center')
        ax.legend(facecolor="white",loc='lower center')

        title = str(str(min_redshift_G)+'_to_'+str(max_redshift_G)+
                    'Cl vs redshift: Histogram of redshift vs Clusters & Galaxies-- INTERPOLATED ')
        plt.title(title)
        saveName=str(outputDir+'THEORY_'+cosmos+'_Interpolated_Histogram_with_z_'+zRange+'.pdf')
        plt.savefig(saveName)  
    
    def GimmeWeights(self,image_weights,zRange):
        '''This will give you the weights that you would later need for the limber appr.'''
        zW =image_weights['Z_MID']

        if zRange  =='020-040':
            nW = image_weights['BIN1']
        elif zRange  =='040-055':
            nW = image_weights['BIN2']
        elif zRange  =='055-070':
            nW = image_weights['BIN3']
        elif zRange  =='070-085':
            nW = image_weights['BIN4']
        print('\n')
        
        return(nW,zW)
    
    def prepareBins(self,nbinsCluster,nbinsGalaxy,RA,redshift_left,data_ra_restricted_C,
                                                RA_G,redshift_left_G,rand_ra_GM,
                                                      zRange,
                                                      outputDir):
        """
        This prepares the bins for you, for the limber integral to come 
        """
        
        #CLUSTERS -- after you have trimmed it
        dataRA =RA
        dataZ = redshift_left
        dataRAZ = np.vstack((dataRA,dataZ))
        whereIsIt = np.in1d(dataRAZ[0,:],data_ra_restricted_C[:] )
        dataRAZ_left = dataRAZ[:,whereIsIt]
        print(dataRAZ_left.shape,'dataRAZ_left.shape')

        #GALAXIES-- after you have trimmed it
        dataZ_G = redshift_left_G
        dataRAZ_G = np.vstack((RA_G,redshift_left_G))
        whereIsIt = np.in1d(dataRAZ_G[0,:],rand_ra_GM[:] )
        dataRAZ_G_left = dataRAZ_G[:,whereIsIt]
        print(dataRAZ_G_left.shape,'dataRAZ_G_left.shape')

        
        nC,binsC,patchesC,nG,binsG,patchesG = self.ShowMeRedshifts(nbinsCluster,nbinsGalaxy,
                                                        dataRAZ_G_left,dataZ_G,
                                                      dataRAZ_left,dataZ,
                                                      zRange,
                                                      outputDir)
        
        np.save(str(outputData+zRange+"_Normalised_dc.npy"), nC)
        np.save(str(outputData+zRange+"_Normalised_dc_bins.npy"), binsC)
        np.save(str(outputData+zRange+"_Normalised_patches.npy"), patchesC)

        np.save(str(outputData+zRange+"_Normalised_dc_G.npy"), nG)
        np.save(str(outputData+zRange+"_Normalised_dc_bins_G.npy"), binsG)
        np.save(str(outputData+zRange+"_Normalised_patches_G.npy"), patchesG)
        
        return(nC,binsC,patchesC,nG,binsG,patchesG)
    
    def ShowMeRedshifts(self,nbinsCluster,nbinsGalaxy,dataRAZ_G_left,dataZ_G,dataRAZ_left,dataZ,
                        zRange,outputDir):
        ''' It is time to plot out the Histogram of redshift.
        Most importantly, this gets the bins, which is what you really want. 
        '''
        f, (ax1) = plt.subplots(1,  figsize=(15,10))

        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white' 

        nG,binsG,patchesG = plt.hist(dataRAZ_G_left[1,:],histtype="step",linewidth=8,alpha=0.65,
                                  bins = nbinsGalaxy,color='pink',
                                  label = 'masked Galaxies in this redshift bin',density=True)
        n0,bins0,patches0 = plt.hist(dataZ_G,histtype="step",linewidth=5,alpha=0.65,
                                  bins = nbinsGalaxy, color='cyan',ls=":",
                                  label = 'Galaxies in this redshift bin',density=True)


        nC,binsC,patchesC = plt.hist(dataRAZ_left[1,:],histtype="step",linewidth=5,alpha=0.85,
                                  bins = nbinsCluster,color='orange', ls=":",
                                  label = 'masked Clusters in this redshift bin',density=True)
        n0,bins0,patches0 = plt.hist(dataZ,histtype="step",linewidth=5,alpha=0.25,
                                  bins = nbinsCluster,color='blue',ls = "--",
                                  label = 'Clusters in this redshift bin',density=True)
        plt.xlabel('redshift')
        plt.ylabel('NORMALIZED # of galaxies and clusters')
        plt.legend(facecolor="white",loc='lower center')
        title = str(zRange+
                    ' Cl vs redshift : Histogram of redshift vs Clusters & Galaxies-- NORMALIZED ')
        plt.title(title)
        saveName=str(outputDir+'Histogram_with_z_'+zRange+'.pdf')
        plt.savefig(saveName)
        print('A histogram of the clusters & galaxies vs redshift (not interpolated yet) is made!')
        return(nC,binsC,patchesC,nG,binsG,patchesG)
    
    def WeightingsProbability(self,nC,binsC,nW,zW):
        '''This gives you the weightings that you need for the limber approximation.  '''
        #CLUSTER
        Z1 = binsC
        W1 = nC
        #GALAXIES
        Z11= zW
        W11 = nW[:-1]

        '''Make me the correct bining !!!'''
        print(W11,'W11')
        print(Z1,'Z1 : These are the redshifts')
        print(Z1.shape,'Z1.shape')
        print(Z11,'Z11 : There are the weighted redshifts')
        print(Z11.shape,'Z11.shape')
        dzBins1 = Z1[1:]-Z1[0:-1]
        dzBins11 = Z11[1:]-Z11[0:-1]
        print(dzBins1.shape,'dzBins1.shape')
        print(dzBins11.shape,'dzBins11.shape')
        zStart = Z1[0]
        zEnd = Z1[-1]
        print(zStart,'zStart')
        print(zEnd,'zEnd')
        
        ''' Make sure that is normalized !!'''
        area = np.sum(W1*dzBins1)
        print(area,'Is this area normalized ? -- it should be !!!')
        area = np.sum(W11*dzBins11)
        print(area,'Is this area normalized ? -- it should be !!!')

        return(Z11,W11,Z1,dzBins1,W1,zStart,zEnd)
    
class MCMC:
    """
    This class runs MCMC for you. 
    """        
    def __init__(self):
        """
        """      
    def step4_runMCMC(self,Initial_bc,Initial_bg,turn_off_deltacdeltag,cosmos):
        """
        This runs the MCMC for you, including chi2
        """
        ''' We force the following number of data poionts '''

        outputData = 'OutputData/Corner/'
        print(outputData,'outputData')
        OutputDirALL = 'OutputPlots/ALL/'
        print(OutputDirALL,'OutputDirALL')
        OutputDirCorner = 'OutputPlots/Corner/'
        print(OutputDirCorner,'OutputDirCorner')


        '''
        GET THE COVARIANCE  MATRIX TO USE AS GALAXY PRIORS !!!!!!
        '''
        bgAverage = np.load('data/bgAverage.npy')
        bgCovMatrix= np.load('data/bgCovMatrix.npy')


        # Set the following, and do not touch it !!! 
        CCshape = 3
        CGshape = 7
        TotSHAPE = CCshape+CGshape
        indices_to_keep = np.array([4,5,6,             #cluster-cluster
                                    7,8,9,10,11,12,13])#clulster-galaxy 
        
        dcdc_selection = np.array([0,1,2,10,11,12,20,21,22,30,31,32])
        
        nwalkers = 64  # This is how many walkers and steps you may need for the mcmc
        steps = 10000
        discardChain=2500
        thinning=15 
        # This is the directory where stuff are 
        Directory =[ "020-040", "040-055","055-070","070-085" ]
        HalfYerr = int(CCshape*len(Directory))#halfway point between those for CC and those for GC
        Redshifts= Directory #["0.20_to_0.40_", "0.40_to_0.55_","0.55_to_0.70_","0.70_to_0.85_"]
 

        # This is manually turned off or on in the header : Save your flat samples ... this the bias ! 
        if turn_off_deltacdeltag == False:
            saveName = str(outputData+'flat_samples_TurnOff_dcdg=False_fullCov.npy')
        elif turn_off_deltacdeltag == True:
            saveName = str(outputData+'flat_samples_TurnOff_dcdg=True_fullCov.npy')
        
        #if turn_off_deltacdeltag == False:
        #    # Have dcdg full covariance, and log_probability_FullCov_LP has +lp unhidden
        #    saveName = str(OutputDir+
        #       'flat_samples_w_bg_turn_off_deltacdeltag=False_fullCov_PLANCK2022_mod.npy')
        #else:            
        #    # You do not have dcdg full covariance, and log_probability_FullCov_LP has +lp hidden
        #    saveName = str(OutputDir+'flat_samples_n_bg_turn_off_deltacdeltag=True_fullCov_PLANCK2022_mod.npy')
        
        print(saveName,'saveName')
        
        # Let's get the pretty colors
        cmap = plt.get_cmap('spring')
        # Let's break the pretty colors down into individual componenets
        colorHot = cmap(np.linspace(0, 1.0, 10))
        # Let's get the pretty colors
        cmap = plt.get_cmap('winter')
        # Let's break the pretty colors down into individual componenets
        colorCool = cmap(np.linspace(0, 1.0, 10))
        

        """
        D: ω_corr + ω_corr_GC -- Everything we did with  TreeCorr
        M: ω_cnc + ω_cng -- Everything that we did  with the Limber Approximation
        θ: The  x-axis  -- the  angular size in [degrees]
        """
        D,M,θ, θ_GC =self.GiveMeDataAndModel_FullCov_COSMO(TotSHAPE,CCshape,Directory,
                                  Redshifts,indices_to_keep,cosmos)
        """
        This is the covariance matrix from our calculaations of TreeCorr 
        """

        
        C = self.GiveMeCovariance4MCMC(TotSHAPE,Directory,indices_to_keep)
        #This is the variance
        Var = np.diag(C)
        #This is the Yerr
        Yerr = np.sqrt(Var)

        print('Check everything before the MCMC!')
        print(bgAverage,'bgAverage')
        print(bgCovMatrix,'bgCovMatrix')
        print(C,'C')
        print(Var,'Var')
        print(Yerr,'Yerr')
        print(CCshape,'CCshape')
        print(LoScale,'LoScale')
        print(HiScale,'HiScale')
        print(Directory,'(Directory')

              
        # START THE MCMC !!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        0) Theta : this is the parameters that you be walking, split into redshifts  
        '''
        # Just set something for now, AND then afte running once, set it close to answer
        NoOfRedshiftBins=len(Redshifts)      
        #  THIS  IS  YOUR  WALKING  PARAMETER 
        theta = self.GiveMeWalkingParameters(NoOfRedshiftBins,
                                          Initial_bc,Initial_bg)
        print(theta,'theta is : bc0,bc1,bc2,bc3,bc4,bg0,bg1,bg2,bg3,bg4')
        
        '''
        1 : The Loglikelihood. Just check it out:You literally return the gaussian in this case
        '''
        lnlikelihood = self.lnlikelihood_FullCov_LP(theta,D,M,C,bgAverage,bgCovMatrix,
                                              CCshape,LoScale,HiScale,Directory,
                                              turn_off_deltacdeltag,dcdc_selection)
        print(lnlikelihood,'lnlikelihood')

        
        '''
        2) Always get your log - prior .You have to set the prior first. In this case, 
        the prior is on the bg -- the bias on the galaxy -- because DES has this.
        You  literally return the  log of the  gaussian.
        '''
        # Eventualy, you will run it with p, where p is a whole list of guessed values.
        log_priors  = self.log_prior(theta,bgAverage,bgCovMatrix)
        print(log_priors,'log priors testing')
        
        '''
        3)You set up the log probability 
        '''
        log_prob =  self.log_probability_FullCov_LP(theta,D,M,C,bgAverage,bgCovMatrix,
                    CCshape,LoScale,HiScale,Directory,turn_off_deltacdeltag,dcdc_selection)
        print(log_prob,'log_prob  testing')
     
        # That is it. You are done with the set up, you run that MCMC like you mean it !
        ########################  M  C  M  C       S T A R T S ########################  
        '''
        4) Initializing a starting position.
        '''
        print(np.set_printoptions(suppress=True,threshold=np.inf))
        # This is just the amount of parameters you are exploring 
        ndim = theta.shape[0]
        # You would just have to guess the number to start walking in 
        # THIS SHOULD BE ROUGHLY THE SAME AS THETA !!!!!
        p0 = np.random.rand(nwalkers, ndim) 
        p0[:,0:len(Directory)] = p0[:,0:len(Directory)] +len(Directory)
        
        '''
        5) You run that mcmc like you mean it !! 
        '''
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_FullCov_LP, 
                                        args=(D, M, C,bgAverage,bgCovMatrix, CCshape,
                                        LoScale,HiScale,Directory,
                                        turn_off_deltacdeltag,dcdc_selection))
        # You check that the log_probability is not too low a function 
        isItTooLow= self.log_probability_FullCov_LP(p0[0],D,M,C,bgAverage,bgCovMatrix,
                    CCshape,LoScale,HiScale,Directory,turn_off_deltacdeltag,dcdc_selection)
        print(isItTooLow, ' is the log_probability is too low a function?')
        
        ''' 
        6) This is it ! You run this and the MCMC RUNS 
        '''
        print('MCMC is running')
        sampler.run_mcmc(p0, steps, progress=True);
        
        '''
        7) The burn in
        '''
        self.burnItIn(Directory,sampler,ndim,steps,OutputDirCorner)
        '''
        8) The theta-chain
        '''
        flat_samples,labels = self.mixAfterBurnIn(Directory,discardChain,thinning,
                                           saveName,sampler,ndim,steps,OutputDirCorner)
        """
        9)Make that corner plot 
        """
        self.makeCorner(flat_samples,labels,steps,OutputDirCorner,turn_off_deltacdeltag)
        
        """
        10)Make plot of angular correlattion vs angles across redshift bins  
        """       

        
        # The angualr correlation 
        self.GiveMeAngularPlotsAcrossRedshiftsPlusBiasedModels2x4CUT_FullCov(D,M,C,θ,θ_GC,
                                                                    colorHot,colorCool,
                                                                    CCshape,CGshape,LoScale,
                                                                    HiScale,Directory,
                                                                    Yerr,HalfYerr,Redshifts,
                                                                    flat_samples,turn_off_deltacdeltag,
                                                                        OutputDirALL)
        # 1_SIGMA version of the angualr correlation 
        self.GiveMeAngularPlotsAcrossRedshiftsPlusBiasedModels2x4CUT_1sigma_FullCov(D,M,C,
                                                                        θ,θ_GC,
                                                                        colorHot,colorCool,
                                                                        CCshape,CGshape,LoScale,
                                                                        HiScale,Directory,
                                                                        Yerr,HalfYerr,Redshifts,
                                                                        flat_samples,turn_off_deltacdeltag,
                                                                        OutputDirALL)
        # Save the bias 
        if turn_off_deltacdeltag == False:
            saveName = str(outputData+'Bias_TurnOff_dcdg=False.npy')
        elif turn_off_deltacdeltag == True:
            saveName = str(outputData+'Bias_TurnOff_dcdg=True.npy')
        np.save(saveName,flat_samples)
        
        # The following is not affected by turning on or off <dc-dg>
        # D: ω_corr + ω_corr_GC                  --> Everything we did with  TreeCorr (OBSERVATION)
        # M: ω_cnc + ω_cng                       --> Everything that we did  with the Limber Approximation (THEORY)
        # C: allRedshift_FULL_covariant_matrixGC --> The full covariance matrix 
        
        saveName = str(outputData+'CovarianceMatrix.npy')
        np.save(saveName,C)
        saveName = str(outputData+'Data.npy')
        np.save(saveName,D)
        saveName = str(outputData+'Model.npy')
        np.save(saveName,M)
        
        #np.save('OutputData/Bias_PLANCK2022_mod.npy',flat_samples)
        #np.save('OutputData/CovarianceMatrix_PLANCK2022_mod.npy',C)
        #np.save('OutputData/Data_PLANCK2022_mod.npy',D)
        #np.save('OutputData/Model_PLANCK2022_mod.npy',M)
    
        print('We had just used turn_off_deltacdeltag = ',turn_off_deltacdeltag)
        print('MCMC ran!')

    def GiveMeAngularPlotsAcrossRedshiftsPlusBiasedModels2x4CUT_1sigma_FullCov(self,D,M,C,θ,θ_GC,
                                                                        colorHot,colorCool,
                                                                        CCshape,CGshape,LoScale,
                                                                        HiScale,Directory,
                                                                        Yerr,HalfYerr,Redshifts,
                                                                        flat_samples,turn_off_deltacdeltag,
                                                                        OutputDirALL):
        '''This one will plot it out for you across all redshifts in 2x4 format  '''
        GalCutShape = HiScale - LoScale # An old code ... this thing is 7
        average = np.mean(flat_samples,axis=0)
        RedShiftsize = len(Redshifts)# this is 4
        halfsize = len(Redshifts) # this is 4
        fullsize = halfsize*2 # this is 8
        Full = CCshape+CGshape # this is about 10


        #PLOT
        f = plt.figure(figsize=(40,20))
        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white'     
        columns = len(Directory)
        rows = 2

        #################################################
        #DATA + optimised model : CLUSTER + CLUSTER 
        for i in range(RedShiftsize):
            ax=f.add_subplot(rows, columns, i+1)
            # The Full-data-Points


            # The data points 
            LABEL = str(Redshifts[i]+' data: δcδc')
            # Calilbrating the seperation points for the input vector
            start=(Full*i)
            end=(Full*i) + CCshape
            plt.plot(θ,D[start:end],c=colorCool[8],lw=10,label = LABEL,alpha=0.35)
            plt.errorbar(θ,D[start:end],c=colorCool[8], alpha=0.35,capsize=20,elinewidth=3,
                         yerr=Yerr[start:end],lw=5)

            #  The biased model
            SIGMA = np.zeros(CCshape)
            inds = np.random.randint(len(flat_samples), size=100)
            ω_cc = M[start:end]
            #ω_cg = M[CCshape:]....

            for ind in inds:
                sample = flat_samples[ind]
                bc = sample[0:RedShiftsize]
                #bg = sample[RedShiftsize:]
                optimized_model = bc[i]*bc[i]*ω_cc
                SIGMA=np.vstack((SIGMA,optimized_model))
            SIGMA = np.delete(SIGMA, (0), axis=0)
            average = np.mean(SIGMA,axis=0)
            std = np.std(SIGMA,axis=0)
            lo_2sigma = average - 1*std
            hi_2sigma = average + 1*std

            # Remember that all you care is how the bias affects the limber model.
            plt.plot(θ,lo_2sigma, c='gold',lw=2, alpha=1)
            plt.plot(θ,hi_2sigma, c='gold',lw=2, alpha=1)
            plt.fill_between(θ, lo_2sigma,hi_2sigma, color='gold',alpha=0.5)  
            TITLE = str('Redshift: '+str(Redshifts[i])) 
            plt.tick_params(labelsize=20, width=3, length=10)
            plt.title(TITLE,fontsize=20)
            #plt.xscale('log')
            plt.yscale('log')

            plt.xlabel(r'$\theta$ (arcmin)',fontsize=20)
            plt.ylabel(r'$\omega$ (degrees)',fontsize=20)
            plt.ylim(10**-3,10**1.5)
            plt.xlim(15,80)
            plt.yticks(fontsize=20)
            plt.grid(True)
            firstColumn = str('CLUSTER-CLUSTER \n')+str(r'$\omega$ (degrees)')
            #if ax.is_first_col():
            if ax.get_subplotspec().is_first_col():
                plt.ylabel(firstColumn,fontsize=20)
        #plt.legend(fontsize=40)   


        ###################################################################
        #OPTIMISED MODEL + data : GALAXY + CLUSTER
        for i in range(RedShiftsize):
            # The data points 
            f.add_subplot(rows, columns, i+halfsize+1)
            LABEL2 = str(Redshifts[i]+' model: δgδc')
            # Calilbrating the seperation points for the input vector
            start=(Full*i) + CCshape
            end=(Full*(i+1))

            plt.plot(θ_GC,D[start:end],c=colorCool[2],lw=15,label =LABEL2,alpha=0.35,ls=":")
            # For the covariance matrix for cluster galaxy
            plt.errorbar(θ_GC,D[start:end],c=colorCool[2], alpha=0.35,ls=":",
                         yerr=Yerr[start:end],lw=2) 

            #  The biased model
            inds = np.random.randint(len(flat_samples), size=100)
            #ω_cc = M[0:CCshape].....
            ω_cg = M[start:end]
            SIGMA = np.zeros(CGshape)
            for ind in inds:
                sample = flat_samples[ind]
                bc = sample[0:RedShiftsize]
                bg = sample[RedShiftsize:]
                optimized_model = bc[i]*bg[i]*ω_cg
                SIGMA=np.vstack((SIGMA,optimized_model))

            # Remember that all you care is how the bias affects the limber model.
            SIGMA = np.delete(SIGMA, (0), axis=0)
            average = np.mean(SIGMA,axis=0)
            std = np.std(SIGMA,axis=0)
            lo_2sigma = average - 1*std
            hi_2sigma = average + 1*std

            plt.plot(θ_GC,lo_2sigma, c='gold',lw=2, alpha=1)
            plt.plot(θ_GC,hi_2sigma, c='gold',lw=2, alpha=1)
            plt.fill_between(θ_GC, lo_2sigma,hi_2sigma, color='gold',alpha=0.5)

            TITLE = str('Redshift: '+str(Redshifts[i]))
            plt.tick_params(labelsize=20, width=3, length=10)
            plt.title(TITLE,fontsize=20)
            #plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\theta$ (arcmin)',fontsize=20)
            if i == 0:
                secondColumn = str('CLUSTER-GALAXY \n')+str(r'$\omega$ (degrees)')
                plt.ylabel(secondColumn,fontsize=20)
            else:
                plt.ylabel(r'$\omega$ (degrees)',fontsize=20)
            plt.ylim(10**-3,10**1.5)
            plt.xlim(15,80)
            plt.yticks(fontsize=20)
            #plt.xticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fontsize=20)
            #ax.set_xticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            plt.grid(True)
 
        f.suptitle('Angular Correlation vs angle across redshift bins at 1SIGMA ',fontsize=40)
        if turn_off_deltacdeltag == False:
            saveName = str(OutputDirALL+'AngularCorr_Allredshift_1SIGMA_dcdg=False_fullCov.png')
        elif turn_off_deltacdeltag == True:
            saveName = str(OutputDirALL+'AngularCorr_Allredshift_1SIGMA_dcdg=True_fullCov.png')
        plt.savefig(saveName) 
        
        
    def GiveMeAngularPlotsAcrossRedshiftsPlusBiasedModels2x4CUT_FullCov(self,D,M,C,θ,θ_GC,
                                                                    colorHot,colorCool,
                                                                    CCshape,CGshape,LoScale,
                                                                    HiScale,Directory,
                                                                    Yerr,HalfYerr,Redshifts,
                                                                    flat_samples,turn_off_deltacdeltag,
                                                                        OutputDirALL):
        '''This one will plot it out for you across all redshifts in 2x4 format  '''
        GalCutShape = HiScale - LoScale # An old code ... this thing is 7
        average = np.mean(flat_samples,axis=0)
        RedShiftsize = len(Redshifts)# this is 4
        halfsize = len(Redshifts) # this is 4
        fullsize = halfsize*2 # this is 8
        Full = CCshape+CGshape # this is about 10

        #PLOT
        f = plt.figure(figsize=(40,20))
        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white'     
        columns = len(Directory)
        rows = 2

        #################################################
        #DATA + optimised model : CLUSTER + CLUSTER 
        for i in range(RedShiftsize):
            ax=f.add_subplot(rows, columns, i+1)
            # The data points 
            LABEL = str(Redshifts[i]+' data: δcδc')
            # Calilbrating the seperation points for the input vector
            start=(Full*i)
            end=(Full*i) + CCshape
            plt.plot(θ,D[start:end],c=colorCool[i*2],lw=10,label = LABEL,alpha=0.35)
            plt.errorbar(θ,D[start:end],c=colorCool[i*2], alpha=0.35,capsize=20,elinewidth=3,
                         yerr=Yerr[start:end],lw=5)

            #  The biased model
            inds = np.random.randint(len(flat_samples), size=100)
            ω_cc = M[start:end]
            #ω_cg = M[CCshape:]....

            for ind in inds:
                sample = flat_samples[ind]
                bc = sample[0:RedShiftsize]
                #bg = sample[RedShiftsize:]
                optimized_model = bc[i]*bc[i]*ω_cc

                # Remember that all you care is how the bias affects the limber model.
                plt.plot(θ,optimized_model, c='gold',lw=5, alpha=0.1)

            TITLE = str('Cluster-Cluster \n Redshift: '+str(Redshifts[i])) 
            plt.tick_params(labelsize=20, width=3, length=10)
            plt.title(TITLE,fontsize=20)
            #plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\theta$ (arcmin)',fontsize=20)
            plt.ylabel(r'$\omega$ (degrees)',fontsize=20)
            plt.ylim(10**-3,10**1.5)
            plt.xlim(15,80)
            plt.yticks(fontsize=20)
            plt.grid(True)

        ###################################################################
        #OPTIMISED MODEL + data : GALAXY + CLUSTER
        for i in range(RedShiftsize):
            # The data points 
            f.add_subplot(rows, columns, i+halfsize+1)
            LABEL2 = str(Redshifts[i]+' model: δgδc')
            # Calilbrating the seperation points for the input vector
            start=(Full*i) + CCshape
            end=(Full*(i+1))

            plt.plot(θ_GC,D[start:end],c=colorCool[i*2],lw=15,label =LABEL2,alpha=0.35,ls=":")
            # For the covariance matrix for cluster galaxy
            plt.errorbar(θ_GC,D[start:end],c=colorCool[i*2], alpha=0.35,ls=":",
                         yerr=Yerr[start:end],lw=2) 

            #  The biased model
            inds = np.random.randint(len(flat_samples), size=100)
            #ω_cc = M[0:CCshape].....
            ω_cg = M[start:end]

            for ind in inds:
                sample = flat_samples[ind]
                bc = sample[0:RedShiftsize]
                bg = sample[RedShiftsize:]
                optimized_model = bc[i]*bg[i]*ω_cg
                # Remember that all you care is how the bias affects the limber model.
                plt.plot(θ_GC,optimized_model, c='gold',lw=5, alpha=0.1)

            TITLE = str('Cluster-Galaxy \n Redshift: '+str(Redshifts[i]))
            plt.tick_params(labelsize=20, width=3, length=10)
            plt.title(TITLE,fontsize=20)
            #plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(r'$\theta$ (arcmin)',fontsize=20)
            plt.ylabel(r'$\omega$ (degrees)',fontsize=20)
            plt.ylim(10**-3,10**1.5)
            plt.xlim(15,80)
            plt.yticks(fontsize=20)
            #plt.xticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fontsize=20)
            #ax.set_xticks([0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            plt.grid(True)


        f.suptitle('Angular Correlation vs angle across redshift bins ',fontsize=40)
        if turn_off_deltacdeltag == False:
            saveName = str(OutputDirALL+'AngularCorr_Allredshift_dcdg=False_fullCov.png')
        elif turn_off_deltacdeltag == True:
            saveName = str(OutputDirALL+'AngularCorr_Allredshift_dcdg=True_fullCov.png')
    
        plt.savefig(saveName)    
        
    def makeCorner(self,flat_samples,labels,steps,OutputDirCorner,turn_off_deltacdeltag):
        """ This will make the corner plot for you """ 
        fig = corner.corner(
        flat_samples, labels=labels, truths=['bc0','bc1','bc2','bc3',
                                             'bg0','bg1','bg2','bg3',],
                           show_titles=True, 
                           title_fmt=".2f",
                           title_kwargs={"fontsize": 20,
                                          "color":'black'},
                           label_kwargs={"fontsize": 24,
                                         "fontweight":'bold',
                                          "color":'grey'},
                           hist_kwargs={"color":"blue",
                                         "alpha":0.08,
                                         "histtype":"stepfilled",},
                           contour_kwargs = {"colors":"black",
                                         "alpha":0.3,},
                           data_kwargs = {"color":"grey",
                                         "alpha":0.2,
                                          "mec":"grey",
                                           "ms" :3 },
                         quantiles=[0.16, 0.5, 0.84],
                           color = "orange",
                           plot_contours = True,
                           plot_density = True,
                           verbose = True,);

        TITLE = str("Contour Plot: \n \n bias_cluster \n vs \n bias_galaxy \n " +
                    str(steps)+' steps')
        fig.gca().annotate(TITLE,
                              xy=(0.82, 0.95), xycoords="figure fraction",
                              xytext=(-20, -10), textcoords="offset points",
                              ha="center", va="top",
                              fontsize = 40, fontweight = 'bold',
                              color = "grey", alpha = 0.58, 
                             )     

        # This is manually turned off or on in the header
        if turn_off_deltacdeltag == False:
            saveName = str(OutputDirCorner+'CornerPlot_TurnOff_dcdg=False_fullCov.pdf')
        elif turn_off_deltacdeltag == True:
            saveName = str(OutputDirCorner+'CornerPlot_TurnOff_dcdg=True_fullCov.pdf')

        plt.savefig(saveName)
        #plt.savefig(str(OutputDir+'ContourPlots_BiasCluster_VS_BiasGalaxy_Steps:'+str(strps)+'.pdf'))
        #plt.show()

    def mixAfterBurnIn(self,Directory,discardChain,thinning,saveName,sampler,ndim,steps,OutputDirCorner):
        """ This will mix the theta chain for you after burn in """
        # Discard is just the first few steps to throw away 
        flat_samples = sampler.get_chain(discard=discardChain, thin=thinning, flat=True)
        # Plot that theta-chain
        fig, axes = plt.subplots(len(Directory)*2, figsize=(10, 30), sharex=True)
        labels = ['bc0','bc1','bc2','bc3','bg0','bg1','bg2','bg3']
        colors = ['orange','orange','orange','orange',
                  'cyan','cyan','cyan','cyan',]

        for i in range(ndim):
            ax = axes[i]
            ax.plot(np.arange(0,len(flat_samples[:,i])), flat_samples[:,i],
                 c=colors[i],alpha=0.9,linewidth = 0.15)
            ax.set_xlim(0, len(flat_samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        TITLE =str('Mixing_after_burn-in_thinned_by_half:_Step=_'+str(steps))
        plt.suptitle(TITLE,fontsize=20)
        plt.savefig(str(OutputDirCorner+TITLE+'.pdf'))
        np.save(saveName,flat_samples)   
        #plt.show()
        return(flat_samples,labels)
        
    def burnItIn(self,Directory,sampler,ndim,steps,OutputDirCorner):
        """ This plots out the burn in for you"""
        fig, axes = plt.subplots(len(Directory)*2, figsize=(10, 30), sharex=True)
        samples = sampler.get_chain()
        labels = ['bc0','bc1','bc2','bc3','bg0','bg1','bg2','bg3',]
        for i in range(ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number");
        title= str('Mixing_with_burn_in:_Steps=_' + str(steps))
        plt.suptitle(title,fontsize=25)
        plt.savefig(str(OutputDirCorner+title+'.pdf'))
        #plt.show()
        
        
    def limitsToBcBg_LP(self,theta):    
        """
        This just limits what bc and bg is
        """
        # ensures that bc is not more than 10 
        for i in range(theta.shape[0]):
            if theta[i]>10:
                return -np.inf
        #  Only half of the walking  parameters -- ie bg -- will be known by DES
        half  = int(theta.shape[0]/2)
        #ensures that bg is not more than 5
        for j in range(half):
             if theta[j+half]>5:
                return -np.inf    
        return (half)

    def log_probability_FullCov_LP(self,theta,D,M,C,bgAverage,bgCovMatrix,
                    CCshape,LoScale,HiScale,Directory,turn_off_deltacdeltag,dcdc_selection):
        '''3)You set up the log probability, eveentually running the log likelihood '''
        lp = self.log_prior(theta,bgAverage,bgCovMatrix)
        #Ensure that 'bc' and 'bg' ie  the walking parameters of  theta is always positive     
        if not np.isfinite(lp):
            return -np.inf  
        # Limits to what bc and bg can be
        self.limitsToBcBg_LP(theta)
        # Get the log Posterior
        if turn_off_deltacdeltag == False:
            logPost =  self.lnlikelihood_FullCov_LP(theta,D,M,C,bgAverage,bgCovMatrix,CCshape,
                                        LoScale,HiScale,Directory,
                                        turn_off_deltacdeltag,dcdc_selection) + lp   
        else:
            logPost =  self.lnlikelihood_FullCov_LP(theta,D,M,C,bgAverage,bgCovMatrix,CCshape,
                                        LoScale,HiScale,Directory,
                                        turn_off_deltacdeltag,dcdc_selection) 
            
        return logPost
        
    def log_prior(self,theta,bgAverage,bgCovMatrix):
        '''This gives you the prior on bg : in this case only, it is a Gaussian function 
        2) Always get your log - prior . You have to set the prior first. 
        In this case, the prior is on the bg  -- the bias on the galaxy -- because DES has this.
        You  literally return the  log of the  gaussian.'''
        # The bias -- this is the only thing you are walking 
        #ensures that all bc or bg is not negative 
        for i in range(theta.shape[0]):     
            if theta[i]<=0:
                return -np.inf
        # Limits to what bc and bg can be
        half = self.limitsToBcBg_LP(theta)
        if half == -np.inf:
                return -np.inf
        
        # Priors
        b_mean   = bgAverage 
        bg_input =  theta[half:]
        uncertainty = bgCovMatrix
        log_priors = self.BgLnlikelihood(bg_input,b_mean,uncertainty)
        return log_priors
        
    def BgLnlikelihood(self,bg_input,b_mean,uncertainty):
        ''' 
        This gets you the log likelikhood for the galaxy priors
        b_mean is from the mag-lim file
        Uncertainty is taken from mag-lim file
        '''
        Cinv = inv(uncertainty)
        A = bg_input - b_mean
        RIGHT = np.dot(Cinv,A)
        AT = A.T
        lnlikelihood = -0.5*(np.dot(AT,RIGHT))

        return lnlikelihood

    def GiveMeCovariance4MCMC(self,TotSHAPE,Directory,indices_to_keep):
        """This one gets the covariance matrix that you calculated during the TreeCorr """ 
        #Get the covariance matrix for clusters and clusters vs galaxies
        dataZ0 = np.load('OutputData/020-040/020-040_FULL_covariant_matrixGC.npy')
        dataZ1 = np.load('OutputData/040-055/040-055_FULL_covariant_matrixGC.npy')
        dataZ2 = np.load('OutputData/055-070/055-070_FULL_covariant_matrixGC.npy')
        dataZ3 = np.load('OutputData/070-085/070-085_FULL_covariant_matrixGC.npy')
        
        '''Let us create the Covariance Matrix'''
        # Here we sellected the indices we want to keep.

        C = np.zeros(TotSHAPE*len(Directory)*TotSHAPE*len(Directory))
        C = C.reshape(TotSHAPE*len(Directory),TotSHAPE*len(Directory))
        print(C.shape,'C.shape')

        # Here is where we did the selection 
        new_dataZ0=dataZ0[:,indices_to_keep][indices_to_keep]
        new_dataZ1=dataZ1[:,indices_to_keep][indices_to_keep]
        new_dataZ2=dataZ2[:,indices_to_keep][indices_to_keep]
        new_dataZ3=dataZ3[:,indices_to_keep][indices_to_keep]
        print(new_dataZ0.shape)
        print(new_dataZ1.shape)
        print(new_dataZ2.shape)
        print(new_dataZ3.shape)

        # Let us have a look at the matrix shape
        C[0:10,0:10]=new_dataZ0
        C[10:20,10:20]=new_dataZ1
        C[20:30,20:30]=new_dataZ2
        C[30:40,30:40]=new_dataZ3
        
        return C
        
        
        
    def lnlikelihood_FullCov_LP(self,theta,D,M,C,bgAverage,bgCovMatrix,CCshape,
                             LoScale,HiScale,Directory,turn_off_deltacdeltag,dcdc_selection):
        ''' 1 : The Loglikelihood.This gets you the log likelikhood
        b is the bias 
        data is ξ(θ)
        model is the angular corelation ω(θ), 
        that your used CAMB +limber on
        Uncertainty is your jacknife covariance matrix
            2 : We also use this to compute chi2, because chi2 == -2 * lnlikelihoodCUT
        '''
        GalCutShape = HiScale-LoScale # An old code ... this thing is 7 ie CGshape
        Full = CCshape + GalCutShape # An old code... this thing is 10, ie TotShape
        bgThetaStarts = int(theta.shape[0]/2)
        # This part is entirely where the modification happens ie the update to the MCMC
        modified = np.zeros(M.shape[0])
        # The models will now become modfied
        # We are going to iterate through 1 per redshift bin
        for i in range(len(Directory)):
            bc = theta[i]
            start=(Full*i)
            end=(Full*i) + CCshape
            #print(start,end,"start,end")
            changes=abs(bc*bc*M[start:end])
            #print(changes,'changes')
            modified[start:end]=changes
        for j in range(len(Directory)):
            bg=theta[bgThetaStarts +j]
            bc=theta[j]
            start=(Full*j) + CCshape
            end=(Full*(j+1))
            #print(start,end,"start,end")
            changes = abs(bc*bg*M[start:end])
            #print(changes,'changes')
            modified[start:end]=changes

        # The uncertainty 
        Cinv = inv(C)
        #  The modified now works as the model
        M  = modified
        # We include this to switch off the contribution from <delta_c delta_g>
        d_minus_m = D - M
        if turn_off_deltacdeltag == True:
            #only the indices corresponding to <delta_c delta_c>
            selection = dcdc_selection
        else:
            selection = np.arange(len(d_minus_m))     
        reduced_d_minus_m = d_minus_m[selection]
        reduced_Cov = (C[selection,:])[:,selection]
        reduced_Cinv = np.linalg.inv(reduced_Cov)
        #print(M,'M - modified')
        #print(reduced_d_minus_m,'reduced_d_minus_m')
        #print(reduced_Cov,'reduced_Cov')
        #print(reduced_Cinv,'reduced_Cinv')
        #print(dcdc_selection,'dcdc_selection')
        log_likelihood = -0.5*np.dot(reduced_d_minus_m, np.dot(reduced_Cinv, reduced_d_minus_m))

        return log_likelihood
        
    def GiveMeWalkingParameters(self,NoOfRedshiftBins,Initial_bc,Initial_bg):
        '''0) Theta : this is the parameters that you be walking, split into redshifts 
        There are only 2  real initial values, split into redshift bins'''
        # Just set something for now, AND then afte running once, set it close to answer
        theta = np.zeros(NoOfRedshiftBins*2)
        theta[:NoOfRedshiftBins] =  Initial_bc
        theta[NoOfRedshiftBins:NoOfRedshiftBins*2] =  Initial_bg 
        return(theta)       
        
    def GiveMeDataAndModel_FullCov_COSMO(self,TotSHAPE,CCshape,Directory,Redshifts,indices_to_keep,
                                                                                       cosmos ):
        ''' 
        This will load and  give you back the Data Vector, the Model Vector and the 
        Covariance Matrix : This will give you only the selected points  to be used
        for the MCMC.
        '''
        ####################################################################### 
        # The DATA vector
        D = np.zeros(TotSHAPE*len(Directory))
        print(D.shape,'D.shape')
        # The MODEL vector 
        M = np.zeros(TotSHAPE*len(Directory))
        print(M.shape,'M.shape')
        ####################################################################### 
        #CLUSTERS + CLUSTERS-GALAXY
        # Let  us do  it redshift-bin by redshift-bin
        for i in range(len(Directory)):
            
            #Cluster
            ω_cnc = np.load(str('OutputData/'+Directory[i]+'/THEORY_'+str(cosmos)+'_' + Redshifts[i]+'_ω_cnc.npy'))
            ω_corr = np.load(str('OutputData/'+Directory[i]+'/'+Redshifts[i]+'_ω_corr.npy'))
            print(ω_cnc.shape,'ω_cnc.shape')
            print(ω_corr.shape,'ω_corr.shape')
            
            #Cluster-Galaxy 
            ω_cng = np.load(str('OutputData/'+Directory[i]+'/THEORY_'+str(cosmos)+'_' + Redshifts[i]+'_ω_cng.npy'))
            ω_corr_GC = np.load(str('OutputData/'+Directory[i]+'/'+Redshifts[i]+'_ω_corrGC.npy'))
            print(ω_cng.shape,'ω_cng.shape')
            print(ω_corr_GC.shape,'ω_corr_GC.shape')

            # The MODEL is the Limber Approximation 
            MODEL =np.append(ω_cnc,ω_cng,axis=0) 
            print(MODEL.shape,'MODEL.shape')
            
            # The DATA is the TreeCorr angular correlation 
            DATA=np.append(ω_corr,ω_corr_GC,axis=0)
            print(DATA.shape,'DATA.shape')

            # We keep only within certain arcmins 
            new_DATA=DATA[indices_to_keep] 
            new_MODEL=MODEL[indices_to_keep]
            print(new_DATA.shape,'new_DATA.shape')
            print(new_MODEL.shape,'new_MODEL.shape')

            initial = TotSHAPE*i
            final = TotSHAPE*(i+1)
            print(initial,'initial')
            print(final,'final')
            D[initial:final] = new_DATA
            M[initial:final] = new_MODEL
            
        # The radial distance is always the same. Just pick the 1st redshift bin 
        theta = np.load(str('OutputData/'+Directory[0]+'/'+Redshifts[0]+'_theta_corr.npy'))
        theta_GC = np.load(str('OutputData/'+Directory[0]+'/'+Redshifts[0]+'_theta_corrGC.npy'))
        THETA=np.append(theta,theta_GC,axis=0) 
        new_THETA=THETA[indices_to_keep] 
        θ = new_THETA[0:CCshape]
        θ_GC= new_THETA[CCshape:]
        print('\n')
        print(D,'D')
        print(M,'M')
        print(θ,'θ')
        print(θ_GC,'θ_GC')
        
        return D,M,θ, θ_GC 

    def log_probabilityMIN_FullCov(self,theta,D,M,C,bgAverage,bgCovMatrix,
                        CCshape,LoScale,HiScale,Directory,turn_off_deltacdeltag, dcdc_selection):
        '''3)You set up the log probability : this one is just for the scipy minimizer.
        So the difference between this and the above should be just a minuss sign
        This is for the chi2 
        '''
        lp = self.log_prior(theta,bgAverage,bgCovMatrix)

        
        logPost =  self.lnlikelihood_FullCov_LP(theta,D,M,C,bgAverage,bgCovMatrix,CCshape,
                                        LoScale,HiScale,Directory,
                                        turn_off_deltacdeltag, dcdc_selection) + lp   
        #print(-logPost,'-logPost')
        return (-logPost)

class check:
    """
    This class performs the checks for you. 
    """        
    def __init__(self):
        """
        """      

    def checkCorrelationGalaxies_ThisTakesHours(self,zRange):
        """
        This will check for you if you have these plots of the galaxy correlation in your bins.

        BE CAREFUL : THIS TAKES HOURS

        and on top of that, you do not actually need these data !! 

        """
        MaxSep=maxSepGal
        MinSep=minSepGal
        
        WhatIsIt='Galaxies'
        yAxis='ω'       
        TITLE=str('AngularCorrelation--'+yAxis+'_'+WhatIsIt+'_with_z_'+zRange)
        print(TITLE,'TITLE')
        loadName=str(outputDir+TITLE+'.pdf')
        
        yAxis='ωθ'
        TITLE2=str('AngularCorrelation--'+yAxis+'_'+WhatIsIt+'_with_z_'+zRange)        
        print(TITLE2,'TITLE2')
        loadName=str(outputDir+TITLE2+'.pdf')
        print(oops)
        
        if os.path.isfile(loadName) == True: # You already have it!
            print('already have it!')
            return
            
        elif os.path.isfile(loadName) == True: # You already have it!
            print('already have it!')
            return
        
        else:
            # Load the patch centers and remake the catalogue
            Name='catGM_'
            loadName = str('data/'+Name+zRange+'_TreeCorr_GalaxyRandom.npy')
            print(loadName,'loadName') 
            catGM_patch_centers = np.load(loadName)
            
        
            # The random map for masking does not need weight, & use the jacknife estimation for patches
            catGM = treecorr.Catalog(ra=rand_ra_GM,dec=rand_dec_GM,
                                     patch_centers=catGM_patch_centers,
                                     ra_units='deg', dec_units='deg')        
            
            # The data map does need weight, and use the random map's patches for patches
            catG = treecorr.Catalog(ra=RA_G,dec=DEC_G, w=w_G_left,
                                    patch_centers=catGM_patch_centers,
                                  ra_units='deg', dec_units='deg')
                
            
            
            ddG = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins=NoOfBins, bin_slop = 0.0, 
                                         var_method='jackknife',
                                         sep_units='arcmin')
            
         
            print('bin_size = %.6f'%ddG.bin_size)
            print(ddG,'ddG')
            ddG.process(catG)

            # This is the step that will take hours to do 
            rrG = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins=NoOfBins,
                                         var_method='jackknife', 
                                         bin_slop = 0.0, sep_units='arcmin')
            rrG.process(catGM)
            print(rrG,'rrG')
            print('bin_size = %.6f'%rrG.bin_size)

            # This also takes forever
            # Now use : the correlation function is the Landy-Szalay formula (dd-2dr+rr)/rr.
            drG = treecorr.NNCorrelation(min_sep=MinSep, max_sep=MaxSep, nbins=NoOfBins,
                                         var_method='jackknife',
                                         bin_slop = 0.0,  sep_units='arcmin')
            drG.process(catG, catGM)
                
            
            # THIS IS THE CORRELATION FUNCTION -- GALAXIES -- DATA NOT SAVED BUT PLOT SAVED
            # WE DO NOT DO GALAXIES -- THIS IS JUST KEPT AS A REF.
            xiG, varxi = ddG.calculateXi(rr=rrG, dr=drG)
            sigG = np.sqrt(varxi)
            ddG_cov = ddG.cov  # Can access covariance now.
            rG = np.exp(ddG.logr)
                       
            yAxis='ω' 
            Tree().PlotMeAngularCorr(WhatIsIt,rG, xiG,sigG,
                              zRange,outputDir,yAxis)
            
            yAxis='ωθ'
            Tree().PlotMeAngularCorr(WhatIsIt,rG,rG*xiG,rG*sigG,zRange,outputDir,yAxis)

    def checkSNR(self,path):
        """
        This checks the signal to noise OF THE CLUSTERS ACROSS ALL REDSHIFTS
        """
        # BEFORE CUTTIING
        zRange = 'ALL'
        print(zRange,'is the redshift range')
        get = gettingData()
        # Make  the output directory 
        outputDir,outputData = get.getOutputDirectory(zRange)  
        print(outputDir,' is the outputDirectory. ',outputData,'is the outputData.')
        
        # The data
        filename = 'DR5_cluster-catalog_v1.0b2.fits'
        image_file = get_pkg_data_filename(path+filename)
        image_table = fits.getdata(image_file, ext=1)
        redshift = image_table['redshift']
        snr = image_table['fixed_SNR']
        fixed_y_c = image_table['fixed_y_c']
        print("BEFORE MAKING ANY CUT")
        print(min(redshift),max(redshift),'min(redshift),max(redshift)')
        print(len(redshift),'len(redshift)')
        print(min(snr),max(snr),'min(snr),max(snr)')  
        print(min(fixed_y_c),max(fixed_y_c),'min(fixed_y_c),max(fixed_y_c)')

        #plots
        plt.figure()
        plt.hist(snr,bins=1000 )
        Name = str('Histogram_SNR')
        plt.title(Name)
        plt.xlim(2,6)
        saveName=str(outputDir+Name+'.pdf')
        plt.savefig(saveName)
        #plt.show()

        plt.figure()
        plt.hist(fixed_y_c,bins=50 )
        Name = str('Histogram_Y')
        plt.title(Name)
        saveName=str(outputDir+Name+'.pdf')
        plt.savefig(saveName)
        #plt.show()
        
        
    def checkB4Cut(self,outputDir,outputData,zRange):
        # BEFORE CUTTIING
        print(zRange,'is the redshift range')
        print(outputDir,' is the outputDirectory. ',outputData,'is the outputData.')
        
        #Load the data
        LoadHer = str(outputData+'/RelevantDataB4Cut')
        with open(LoadHer, 'rb') as f: 
            myList = pickle.load(f)
        [RA_G,DEC_G,redshift_G,min_redshift_G,max_redshift_G,w_G,\
         RA,DEC,redshift,snr,rand_ra_M,rand_dec_M,rand_z_M,rand_snr_M,\
         rand_ra_GM,rand_dec_GM,rand_z_GM] = myList
        print(len(redshift),'number of clusters BEFORE CUT ')
        print(len(redshift_G),'number of galaxies BEFORE CUT ')
       
        # OK make them histograms 
        Name='Histogram_of_redshift_vs_Clusters_&_Galaxies--NORMALIZED'
        self.PlotRedshiftVsClustersAndGal(redshift,redshift_G,outputDir,Name,
                               min_redshift_G,max_redshift_G)
        
        print("checking B4 cut.... checked !\n")
        
    def checkAftCut(self,outputDir,outputData,zRange):
        # After Cutting
        print(zRange,'is the redshift range')
        print(outputDir,' is the outputDirectory. ',outputData,'is the outputData.')
        
        #Load the data After Cut
        LoadHer = str(outputData+'/RelevantDataAfterCut')
        with open(LoadHer, 'rb') as f: 
            myList = pickle.load(f)
        [RA,DEC,redshift_left,rand_ra_M,rand_dec_M,rand_z_M_left,\
         RA_G,DEC_G,redshift_left_G,w_G_left,\
         rand_ra_GM,rand_dec_GM,rand_z_GM_left,
         min_redshift_left_G,max_redshift_left_G] = myList                                         
        print(len(redshift_left),'number of clusters AFTER CUT ')
        print(len(redshift_left_G),'number of galaxies AFTER CUT ')
       
        # OK make them histograms 
        Name=' Histogram of redshift vs Clusters & Galaxies-- NORMALIZED_w_snr+redshift_cut'
        self.PlotRedshiftVsClustersAndGal(redshift_left,redshift_left_G,outputDir,
                               Name,min_redshift_left_G,max_redshift_left_G)
        
        print("checking After cut.... checked !\n")
        
        
    def checkAftCutAndHealPix(self,outputDir,outputData,zRange):
        # After Cutting and healpix ie the DES footprint
        print(zRange,'is the redshift range')
        print(outputDir,' is the outputDirectory. ',outputData,'is the outputData.')

        #Load the data After Cut
        LoadHer = str(outputData+'/RelevantDataAfterCut+HealPix')
        with open(LoadHer, 'rb') as f: 
            myList = pickle.load(f)
        [RA,DEC,redshift_left,rand_ra_M,rand_dec_M,rand_z_M_left,\
                 RA_G,DEC_G,redshift_left_G,w_G_left,\
                 rand_ra_GM,rand_dec_GM,rand_z_GM_left,
                 min_redshift_left_G,max_redshift_left_G,
                 rand_ra_restricted_CM,\
                 rand_dec_restricted_CM,\
                 data_ra_restricted_C,\
                 data_dec_restricted_C,\
                 data_pix_indices_C,\
                 rand_pix_indices_CM] = myList                                         
        print(len(data_ra_restricted_C),'number of clusters AFTER CUT +HealPix')
  

        # OK make them histograms 
        Name=' Histogram of redshift vs Clusters & Galaxies:NORMALIZED_w_snr+redshift_cut+HealPix'
        self.PlotRedshiftVsClustersAndGal(redshift_left,redshift_left_G,outputDir,
                               Name,min_redshift_left_G,max_redshift_left_G)

        print("checking After cut And HealPix.... checked !\n")
        
    def checkAftCutAndHealPixAndSurveyMapALL(self,outputDir,outputData):
        # After Cutting and healpix ie the DES footprint and making the survey map 
        # This is like a double check ... just in case 
        print('ALL is the redshift range')

        #Load the data After Cut        
        NameC=str('Cluster+Mask')

        path = str('OutputData/020-040/')
        filename = str(NameC+' betweeen z = 0.20_to_0.40_data.npy')
        data_ra_restricted_C1, data_dec_restricted_C1 = np.load(path+filename) 
        

        path = str('OutputData/040-055/')
        filename = str(NameC+' betweeen z = 0.40_to_0.55_data.npy')
        data_ra_restricted_C2, data_dec_restricted_C2    =   np.load(path+filename) 

        path = str('OutputData/055-070/')  
        filename = str(NameC+' betweeen z = 0.55_to_0.70_data.npy')
        data_ra_restricted_C3, data_dec_restricted_C3    =   np.load(path+filename)       

        path = str('OutputData/070-085/')    
        filename = str(NameC+' betweeen z = 0.70_to_0.85_data.npy')
        data_ra_restricted_C4, data_dec_restricted_C4=   np.load(path+filename)
        
        print(data_ra_restricted_C1.shape)
        print(data_ra_restricted_C2.shape)
        print(data_ra_restricted_C3.shape)
        print(data_ra_restricted_C4.shape)
        print(data_ra_restricted_C1.shape[0]+data_ra_restricted_C2.shape[0]+\
        data_ra_restricted_C3.shape[0]+data_ra_restricted_C4.shape[0],\
        'The total nuumber of clusters used')

        print("checking After cut And HealPix and Survey Map.... checked !\n")        
        
    def PlotRedshiftVsClustersAndGal(self,redshift,redshift_G,outputDir,\
                                     Name,min_redshift_G,max_redshift_G):
        ''' Plot out the redshifts distribution of the gaalaxies and clusters '''
        f, (ax1) = plt.subplots(1,  figsize=(15,10))
        plt.style.use('classic')
        plt.rcParams['figure.facecolor'] = 'white' 
        nbins=50
        nG_Original,binsG_Original,patchesG_Original = plt.hist(redshift_G,histtype="step",
                                                                linewidth=5,alpha=0.65,
                                  bins = nbins,color='red', ls=":",
                                  label = 'Orignal # Galaxies',density=True)


        nC_Original,binsC_Original,patchesC_Original = plt.hist(redshift,histtype="step",
                                                                linewidth=5,alpha=0.25,
                                  bins = nbins,color='green', ls=":",
                                  label = 'Orignal # Clusters',density=True)

        plt.axvline(float(min_redshift_G),c='purple', ls=":")
        plt.axvline(float(max_redshift_G),c='purple', ls=":")

        ax1.fill_between([float(min_redshift_G),float(max_redshift_G)], 0,8,alpha=0.5,
                         color='yellow',label="Restricted redshift")
        plt.xlabel('redshift')
        plt.ylabel('NORMALIZED # of galaxies and clusters')
        plt.legend()
        title = str(Name+'\n'+outputDir)
        saveName=str(os.getcwd()+'/'+outputDir+Name+'_with_redshift_'+str(min_redshift_G) +
                     '_to_'+str(max_redshift_G)+'.pdf')

        plt.ylim(0,8)
        plt.xlim(0,2)
        plt.title(title)
        plt.savefig(saveName)
        #plt.show()
        
    def checkJacknifePatch(self,zRange,nside,npix,nJack,outputData):
        """
        This checks the jacknife patch you are using; along the way, it also checks
        the area used 
        """
        #Load the data After Cut
        print(zRange,'is the redshift range')
        print(outputData,'is the outputData.')
        
        #Get that data 
        LoadHer = str(outputData+'/RelevantDataAfterCut')
        with open(LoadHer, 'rb') as f: 
            myList = pickle.load(f)
        [RA,DEC,redshift_left,rand_ra_M,rand_dec_M,rand_z_M_left,\
         RA_G,DEC_G,redshift_left_G,w_G_left,\
         rand_ra_GM,rand_dec_GM,rand_z_GM_left,
         min_redshift_left_G,max_redshift_left_G] = myList  
        
        # Area of 1 pixel [steradians]
        AreaOnePix = 4*np.pi / npix 
        
        # Area of the Clusters
        WhatIsIt='Clusters'
        Area = FindArea().GiveMeArea(WhatIsIt,nside,npix,rand_ra_M,rand_dec_M)
        print(Area,'Area')
        
        # Area of the Galaxies
        WhatIsIt='Galaxies'
        GArea = FindArea().GiveMeArea(WhatIsIt,nside,npix,rand_ra_GM,rand_dec_GM)
        print(GArea,'GArea')
        
        # Whatt we are dealing with 
        print(npix,'the total #  of pixels in the sky.')
        print(Area.shape[0],'pix --Survey Area of Clusters')
        print(GArea.shape[0],'pix --Survey Area of galaxies')
        
        
        FindArea().GiveMeJacknifePatch(Area,GArea,nJack,AreaOnePix)
        
    def checkTheSurveyMapAndMask(self,zRange,nside,outputData,outputDir):
        """
        It will make the survey map and mask for you. This takes a long time to run.
        """
         #Load the data After Cut
        print(zRange,'is the redshift range')
        print(outputDir,' is the outputDirectory. ',outputData,'is the outputData.')
        
        #Get that data 
        LoadHer = str(outputData+'/RelevantDataAfterCut+HealPix')
        with open(LoadHer, 'rb') as f: 
            myList = pickle.load(f)
            
        [RA,DEC,redshift_left,rand_ra_M,rand_dec_M,rand_z_M_left,\
         RA_G,DEC_G,redshift_left_G,w_G_left,\
         rand_ra_GM,rand_dec_GM,rand_z_GM_left,
         min_redshift_left_G,max_redshift_left_G,
         rand_ra_restricted_CM,\
         rand_dec_restricted_CM,\
         data_ra_restricted_C,\
         data_dec_restricted_C,\
         data_pix_indices_C,\
         rand_pix_indices_CM] = myList 
        
        # Make them maps
        Name=str('Cluster+Mask')
        WhatIsIt='Clusters'
        FindArea().ShowMeTheMapAndMask(nside,WhatIsIt,data_ra_restricted_C, data_dec_restricted_C,\
                      rand_ra_restricted_CM,rand_dec_restricted_CM,outputData,\
                   outputDir,Name,min_redshift_left_G,max_redshift_left_G )
        
        Name=str('Galaxies+Mask')
        WhatIsIt='Galaxies'
        FindArea().ShowMeTheMapAndMask(nside,WhatIsIt,data_ra_restricted_G, data_dec_restricted_G,\
                      rand_ra_restricted_GM,rand_dec_restricted_GM,outputData,\
                   outputDir,Name,min_redshift_left_G,max_redshift_left_G )
        
        print("Maps of clusters and galaxies ...done.")
    
    def checkMolllweideMaps(self,zRange,nside,outputData,outputDir):
        """
        Plots the Mollweide maps for you 
        """
        
        #Load the data After Cut
        print(zRange,'is the redshift range')
        print(outputDir,' is the outputDirectory. ',outputData,'is the outputData.')
        
        #Get that data 
        LoadHer = str(outputData+'/RelevantDataAfterCut+HealPix')
        with open(LoadHer, 'rb') as f: 
            myList = pickle.load(f)
            
        [RA,DEC,redshift_left,rand_ra_M,rand_dec_M,rand_z_M_left,\
         RA_G,DEC_G,redshift_left_G,w_G_left,\
         rand_ra_GM,rand_dec_GM,rand_z_GM_left,
         min_redshift_left_G,max_redshift_left_G,
         rand_ra_restricted_CM,\
         rand_dec_restricted_CM,\
         data_ra_restricted_C,\
         data_dec_restricted_C,\
         data_pix_indices_C,\
         rand_pix_indices_CM] = myList 

        #clusters
        Name='ClustersMollview'
        FindArea().GiveMeMollView(npix,rand_pix_indices_CM,data_pix_indices_C,
                         outputDir,Name,min_redshift_left_G,max_redshift_left_G)

        print("Mollweide Maps of clusters ...done.") 

    def checkPhotoShiftAndStretch(self,zRange,nside,outputData,photoShiftAndStretch):
        """
        This checks the Photomeric Shift And Stretch for you
        """
        
        #Load the data After Cut
        print(zRange,'is the redshift range')
        print(outputData,'is the outputData.')
        
        #Get that data 
        LoadHer = str(outputData+'/InterPolated_Weights')
        with open(LoadHer, 'rb') as f: 
            myList = pickle.load(f)

        Z11,W11,Z1,W1,Start,zEnd,f11,f1,zBins,znew = myList
            
       # Defining the shift and stretch in the photometric redshift # MIGHT HAVE TO CHECK THIS UP LATER
        LensZ  = np.array([[-0.009, 0.007],[-0.035, 0.011],[-0.005, 0.006],[-0.007, 0.006]])
        print(LensZ,'LensZ')
        StretchZ = np.array([[0.98, 0.06],[1.31, 0.09],[0.87, 0.05],[0.92, 0.05]])
        print(StretchZ,'StretchZ')
            
        # Inclulde in the photometric redshift -shifted and stretched
        mu,muStd,sigma,sigmaStd =  self.GimmeShiftAndStretch(LensZ,StretchZ,zRange)
        print(mu,muStd,sigma,sigmaStd,'mu,muStd,sigma,sigmaStd')
        if photoShiftAndStretch == cosmos: # Planck22 
            Δz  =0 #FUDICIAL or Planck22
            σz = 1 #FUDICIAL or Planck22
        else:
            Δz  = mu  + muStd # with shift
            σz = sigma  + sigmaStd # with stretch

        # MAKE THE NEW WEIGHTS HERE 
        print('We use the new W11 = fnew(Z11) to make the new weights Wg_new')
        AveZ  = np.sum(W11*zBins*znew)   
        #   Here we go to find the new weights 
        ZNEW =σz*(znew-AveZ)+AveZ
        Wg = f11(ZNEW-Δz)
        Wg_new = σz*Wg
        return (Δz,σz,Wg_new)
        
    def GimmeShiftAndStretch(self,LensZ,StretchZ,zRange):
        '''This will give you the shift and stretch for the photoZ of GALAXIES ONLY'''

        if zRange  =='020-040':
            i=0
        elif zRange  =='040-055':
            i=1
        elif zRange  =='055-070':
            i=2
        elif zRange  =='070-085':    
            i=3
        mu,muStd = LensZ[i,0],LensZ[i,1]
        sigma,sigmaStd = StretchZ[i,0],StretchZ[i,1]

        return(mu,muStd,sigma,sigmaStd)        

class Chi_squared:
    """
    This class performs the checks for you. 
    """        
    def __init__(self):
        """
        """      

    def step5_getChi2(self, LoScale,HiScale,turn_off_deltacdeltag):
        """
        This gets you the chisquared.
        """
        bgAverage = np.load('data/bgAverage.npy')
        bgCovMatrix= np.load('data/bgCovMatrix.npy')
        Directory =["020-040", "040-055","055-070", "070-085", ]
        Redshifts = Directory
        CCshape = 3
        CGshape = 7
        TotSHAPE = CCshape+CGshape
        indices_to_keep = np.array([4,5,6,             #cluster-cluster
                                    7,8,9,10,11,12,13])#clulster-galaxy 
        dcdc_selection = np.array([0,1,2,10,11,12,20,21,22,30,31,32])
        D,M,θ, θ_GC = MCMC().GiveMeDataAndModel_FullCov_COSMO(TotSHAPE,CCshape,Directory,
                                  Redshifts,indices_to_keep,cosmos)
        NoOfRedshiftBins=4
        Initial_bc = 4
        Initial_bg = 1.5
        
        
        #  THIS  IS  YOUR  WALKING  PARAMETER !!!!!!!!!!!!
        theta = MCMC().GiveMeWalkingParameters(NoOfRedshiftBins,
                                          Initial_bc,Initial_bg)
        """
        This is the covariance matrix from our calculaations of TreeCorr 
        """
        
        C = MCMC().GiveMeCovariance4MCMC(TotSHAPE,Directory,indices_to_keep)

        BestFitParameters= minimize(MCMC().log_probabilityMIN_FullCov,x0=np.zeros(8)+2,
                            args=(D,M,C,bgAverage,bgCovMatrix,CCshape,
                                  LoScale,HiScale,Directory,
                                  turn_off_deltacdeltag, dcdc_selection))
        bestFitParam = BestFitParameters['x']
        print(bestFitParam,'bestFitParam')
        
        chi2 = -2*MCMC().lnlikelihood_FullCov_LP(bestFitParam,D,M,C,bgAverage,bgCovMatrix,
                    CCshape,LoScale,HiScale,Directory,turn_off_deltacdeltag,dcdc_selection)
        print(chi2,'chi2')
        print(chi2/40, 'chi2 per degree of freedom')
        

"""""    
main Function
"""""        
class main:
    """
    This runs the programs step-by-step for you.
    """
    def __init__(self,zRange,LoScale,HiScale,path,outputDir,outputData,nside,npix,nJack,NoOfBins,
                minSepGal,maxSepGal,minSepCluster,maxSepCluster,minSepCross,maxSepCross,
                nbinsCluster,nbinsGalaxy,AS,OMBH2,OMCH2,lspace,
                 photoShiftAndStretch,Initial_bc,Initial_bg,turn_off_deltacdeltag,cosmos):
        """
        You can select to run all or just run parts
        """
        gettingData().step0_getRelevantInfo(zRange,path)
        FindArea().step1_runHealPix(zRange,outputDir,outputData,nside,npix)
        Tree().step2_runTreeCorr(zRange,outputDir,outputData,nside,npix,nJack,NoOfBins,
                                 minSepGal,maxSepGal,minSepCluster,
                               maxSepCluster,minSepCross,maxSepCross)
        Limber().step3_runCAMB(path,zRange,outputDir,outputData,nbinsCluster,nbinsGalaxy,
                               AS,OMBH2,OMCH2,lspace,photoShiftAndStretch,cosmos) 
        #MCMC().step4_runMCMC(Initial_bc,Initial_bg,turn_off_deltacdeltag,cosmos)
        #Chi_squared().step5_getChi2( LoScale,HiScale,turn_off_deltacdeltag)

"""""    
START
"""""
################  D O   N O T   C H A N G E   T H E S E ################  

# How much of the angular scale you want 
# this is an old code, but it is important that the difference is SEVEN
LoScale=0
HiScale=7

#The pixels for healpix
nside =64
npix  = 12*nside**2

# The Jacknife patch  
nJack = 200  

# For TreeCorr
NoOfBins = 7 # Number of Correlation bins 

# Correlations bins range 
minSepGal = 30 #arcmin
maxSepGal = 70 #arcmin
minSepCluster = 10 #arcmin
maxSepCluster = 70 #arcmin
minSepCross = 30 # arcmin
maxSepCross = 70 # arcmin

# Weights 
nbinsCluster=10
nbinsGalaxy =50

# Get the multipole space for interpolation
lspace = np.logspace(-2,np.log10(30000),num=1000,endpoint=True)

# For the MCMC
Initial_bc = 4
Initial_bg = 1.5

################  D O   N O T   C H A N G E   A B O V E  ################ 
"""
You can go to main() itself to select which parts to run : 
        0) Getting and cutting the data to relevant info
        1) Runs HealPix
        2) Runs TreeCorr
        3) Runs Camb ( using the Limber Approx ) & the angular Correlation
        4) Runs MCMC

"""
################  M A N U A L L Y   C H A N G E   T H E S E ################  

#COSMOLOGICAL PARAMETERS                                    
AS = 2.1005829e-9    # PLANCK2022 1ST RUN mod
OMBH2=0.022383       # PLANCK2022 1ST RUN mod
OMCH2=0.12011        # PLANCK2022 1ST RUN mod
#############################################
#AS = 2.1005829e-9*1.05  # 5% INCREASE  
#OMBH2=0.022383       # PLANCK2022 1ST RUN mod
#OMCH2=0.12011        # PLANCK2022 1ST RUN mod
#############################################
#AS = 2.1005829e-9    # PLANCK2022 1ST RUN mod
#OMBH2 = 0.022383*1.05 # 5% INCREASE 3RD RUN mod
#OMCH2 = 0.12011*1.05  # 5% INCREASE 3RD RUN mod

# set your cosmology 
cosmos='Planck22'

# SHIFT&STRETCH of photometric data
photoShiftAndStretch=cosmos# not shift nor stretch 
#photoShiftAndStretch='SHIFTnSTRETCH' # It has beeen shifted and stretched

# False means use galaxy priors // True means don't use it // for MCMC only
turn_off_deltacdeltag = False

# Choose the range you want to run it at. 
zRange  ='020-040'
#zRange  =  '040-055'
#zRange  = '055-070'
#zRange  = '070-085'

################  E N D    O F    A N Y    C H A N G E  #################### 

#Where the data is 
path = 'data/'
outputDir,outputData = gettingData().getOutputDirectory(zRange) 
print(outputData,'outputData ') # 'OutputData/zRange/'
print(outputDir,'outputDir') # 'OutputPlots/zRange/'

#Make the Directory if it doesn't exist
try:
    os.mkdir(outputDir) 
except:
    pass
try:
     os.mkdir(outputData) 
except:
    pass



############################## S T A R T  ##################################

if __name__=='__main__':
    print("Running ... ... TzeFinalProj.py project on Cluster Cosmology " )
    
    #MAIN
    #############################################
    # Following needs selection on redshift range 
    main(zRange,LoScale,HiScale,path,outputDir,outputData,nside,npix,nJack,NoOfBins,
        minSepGal,maxSepGal,minSepCluster,maxSepCluster,minSepCross,maxSepCross,
        nbinsCluster,nbinsGalaxy,AS,OMBH2,OMCH2,lspace,photoShiftAndStretch,
         Initial_bc,Initial_bg,
        turn_off_deltacdeltag,cosmos)
    
    #CHECKS
    #############################################    
    ####Following is for all redshift range 
    #check().checkSNR(path) # This is for all redshift range
    #check().checkAftCutAndHealPixAndSurveyMapALL(outputDir,outputData)
    #############################################
    ####Following needs selection on redshift range
    #check().checkB4Cut(outputDir,outputData,zRange)
    #check().checkAftCut(outputDir,outputData,zRange)
    #check().checkJacknifePatch(zRange,nside,npix,nJack,outputData)
    #check().checkTheSurveyMapAndMask(zRange,nside,outputData,outputDir)
    #check().checkMolllweideMaps(zRange,nside,outputData,outputDir)
    #check().checkAftCutAndHealPix(outputDir,outputData,zRange)
    #check().checkPhotoShiftAndStretch(zRange,nside,outputData,photoShiftAndStretch)
    
    #BE CAREFUL, THIS ONE WILL TAKE HOURS 
    #check().checkCorrelationGalaxies_ThisTakesHours(zRange)



