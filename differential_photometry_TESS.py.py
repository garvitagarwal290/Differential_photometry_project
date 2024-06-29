#created by Garvit Agarwal
#29th July 2023

import sep
import numpy as np
from glob import glob
from astropy.io import fits
from astropy.time import Time
from copy import deepcopy
import pathlib
import datetime
import os
from tqdm import trange
import sys
import math
from astropy import wcs, stats
import matplotlib.pyplot as plt
import cv2
import matplotlib
import warnings
import lightkurve as lk
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord


#get the fluxes of each star from each image using the user-decided aperture
def get_fluxes(aperture):

    data = np.empty([num_images, num_stars], dtype=(np.float64)) #array to store data
    
    if(aperture=='plus'):
        #calculating the positions of the pixels in the plus aperture
        plus_pixels = np.ndarray((num_stars, 5,2), dtype=int)
        for i in range(num_stars):
            plus_pixels[i,0] = stars_pos[i]
            plus_pixels[i,1] = [stars_pos[i,0]-1, stars_pos[i,1]]
            plus_pixels[i,2] = [stars_pos[i,0], stars_pos[i,1]-1]
            plus_pixels[i,3] = [stars_pos[i,0]+1, stars_pos[i,1]]
            plus_pixels[i,4] = [stars_pos[i,0], stars_pos[i,1]+1]


        for i in range(num_images):

            image = images[i]
            fluxes = np.zeros((num_stars))
            for j in range(num_stars):
                for k in range(5):
                    fluxes[j] += image[plus_pixels[j,k,0],plus_pixels[j,k,1]] #adding the fluxes in the aperture
        
            data[i] = fluxes


    elif(aperture=='point'):
        
        for i in range(num_images):

            image = images[i]
            fluxes = np.zeros((num_stars))
            for j in range(num_stars):
                fluxes[j] = image[stars_pos[j,0], stars_pos[j,1]]
        
            data[i] = fluxes

    return data


target_radec = [224.161770871613,-50.985501453475]
savepath = '/home/gagarwal/Downloads/relative_LC.txt'
#choose aperture shape
aperture= 'point' #single pixel
# aperture = 'plus'

#The user has to provide the pixel positions of the target first, followed by the reference stars' positions. The stars_pos has
# to be in the shape (num of stars including the target, 2) 
stars_pos = np.array([[5,5],[7,3],[1,7],[3,2]])
num_stars = stars_pos.shape[0]
initialx, initialy= stars_pos[:,0], stars_pos[:,1]


#Load the TESS data from the target's ra dec. The data is assumed to be a data cube
obj = SkyCoord(target_radec[0], target_radec[1], unit="deg")

search_result = lk.search_targetpixelfile(obj)

tpf_file = search_result[4].download(quality_bitmask='default')
time = tpf_file.time
time = time.value
images = tpf_file.flux
images = images.value
images = images.byteswap().newbyteorder()
        
num_images = images.shape[0]
print ("Imported", num_images, "frames")


#get the fluxes of each star from each image using the user-decided aperture
data = get_fluxes(aperture)


#weighted mean lightcurve of the reference stars and target's flux by it to get relative flux
weights = []
star_fluxes=[]
for i in range(num_stars-1):
    weights.append((1/stats.sigma_clipped_stats(data[:,i+1]/np.mean(data[:,i+1]))[2])**2)
    star_fluxes.append(data[:,i+1]/np.mean(data[:,i+1]))

mean_lc=np.average(star_fluxes,weights=weights, axis=0)
rel_lc = (data[:,0]/np.mean(data[:,0]))/mean_lc

#saving the relative lightcurve
f = open(savepath, 'w')
f.write('time,relative_flux\n')
np.savetxt(f, np.column_stack((time, rel_lc)), delimiter=',')
    

