#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:32:08 2022

@author: Roman A.

Perform multiaperture photometry of 1-minute stacked frames
"""

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

def getXY(transform_header, ra_dec):
    '''Use transform taken from fits image header to convert from ra dec to pixel coords'''
    
    #load in transformation information
    transform = wcs.WCS(transform_header)
    
    ra=ra_dec[:,0]
    dec=ra_dec[:,1]

    px = transform.all_world2pix(ra, dec, 0, ra_dec_order=True,quiet=True)
    
    return px[0], px[1]


def getRAdec(transform, xy):
    '''Use transform taken from fits image header to convert from pixel coords to ra dec'''
    
    x=xy[:,0]
    y=xy[:,1]
    
    #get transformation
    world = transform.all_pix2world(x,y, 0,ra_dec_order=True) #2022-07-21 Roman A. changed solution function to fit SIP distortion
    
    coords = np.array([world[0], world[1]]).transpose()
    
    return coords



def initialFindFITS(data, detect_thresh, target_mask):
    """ Locates the stars in the given image
    input: flux data in 2D array for a fits image, star detection threshold (float)
    returns: [x, y, half light radius, npix, semi-major/semi-minor ratio, flux] of all stars in pixels"""

    ''' Background extraction for initial time slice'''
    data_new = deepcopy(data)           #make copy of data
    bkg = sep.Background(data_new)      #get background array
    bkg.subfrom(data_new)               #subtract background from data
    thresh = detect_thresh * bkg.globalrms      # set detection threshold to mean + 3 sigma


    ''' Identify stars in initial time slice '''
    objects = sep.extract(data_new, thresh, mask=target_mask)

    ''' Characterize light profile of each star '''
    halfLightRad = np.sqrt(objects['npix'] / np.pi) / 2.  # approximate half light radius as half of radius

    
    ''' Generate tuple of (x,y,r) positions for each star'''
    positions = zip(objects['x'], objects['y'], halfLightRad, objects['npix'], objects['a']/objects['b'], objects['flux'])

    return positions


def refineCentroid(data, time, coords, sigma):
    """ Refines the centroid for each star for an image based on previous coords, used for tracking
    input: flux data in 2D array for single fits image, header time of image, 
    coord of stars in previous image, weighting (Gauss sigma)
    returns: new [x, y] positions, header time of image """

    '''initial x, y positions'''
    x_initial = [pos[0] for pos in coords]
    y_initial = [pos[1] for pos in coords]
    
    '''use an iterative 'windowed' method from sep to get new position'''
    new_pos = np.array(sep.winpos(data, x_initial, y_initial, sigma, subpix=5))[0:2, :]
    x = new_pos[:][0].tolist()
    y = new_pos[:][1].tolist()
    
    '''returns tuple x, y (python 3: zip(x, y) -> tuple(zip(x,y))) and time'''
    return x,y



def clipCutStars(x, y, x_length, y_length):
    """ When the aperture is near the edge of the field of view sets flux to zero to prevent 
    fadeout
    input: x coords of stars, y coords of stars, length of image in x-direction, 
    length of image in y-direction
    returns: indices of stars to remove"""

    edgeThresh = 20.          #number of pixels near edge of image to ignore
    
    '''make arrays of x, y coords'''
    xeff = np.array(x)
    yeff = np.array(y) 
    
    '''get list of indices where stars too near to edge'''
    ind = np.where(edgeThresh > xeff)
    ind = np.append(ind, np.where(xeff >= (x_length - edgeThresh)))
    ind = np.append(ind, np.where(edgeThresh > yeff))
    ind = np.append(ind, np.where(yeff >= (y_length - edgeThresh)))
    
    return ind



def getSizeFITS(filenames):
    """ gets dimensions of fits 'video' 
    input: list of filenames in directory
    returns: width, height of fits image, number of images in directory, 
    list of header times for each image"""
    
    '''get names of first and last image in directory'''
    filename_first = filenames[0]
    frames = len(filenames)     #number of images in directory

    '''get width/height of images from first image'''
    file = fits.open(filename_first)
    header = file[1].header
    width = header['NAXIS1']
    height = header['NAXIS2']

    return width, height, frames


def importFramesFITS(parentdir, filenames, start_frame, num_frames, flip=False):
    """ reads in frames from fits files starting at frame_num
    input: parent directory (minute), list of filenames to read in, starting frame number, how many frames to read in, 
    bias image (2D array of fluxes)
    returns: array of image data arrays, array of header times of these images"""

    imagesData = []    #array to hold image data
    imagesTimes = []   #array to hold image times
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [filename for i, filename in enumerate(filenames) if i >= start_frame and i < start_frame + num_frames]

    '''get data from each file in list of files to read, subtract bias frame'''
    for filename in files_to_read:
        file = fits.open(filename)
        
        try:
            header = file[1].header
            
            ''' Calibration frame correction '''
            data = (file[1].data) #- bias #- dark)/flat 
            headerTime = header['DATE-OBS']
        except KeyError:                #Garvit Ag. In rare cases it was seen as that the data and header was in the first index of the 'file' object instead of the usual 1st index
            header = file[0].header
            
            ''' Calibration frame correction '''
            data = (file[0].data) #- bias #- dark)/flat 
            headerTime = header['DATE-OBS']
            
        file.close()

        if(flip):data = np.flip(data)
        imagesData.append(data)
        imagesTimes.append(headerTime)
         
    '''make into array'''
    imagesData = np.array(imagesData, dtype='float64')
    
    '''reshape, make data type into floats'''
    if imagesData.shape[0] == 1:
        imagesData = imagesData[0]
        imagesData = imagesData.astype('float64')
        
    return imagesData, imagesTimes


def get_correct_hdrfile_idx(num_images, filenames, data_path): 
    '''Garvit Ag: Finds the image with the highest flux for the target, when the flux is calculated using the pixel 
    coords derived from the ra dec. This gives us the one image in the sequence that is guaranteed to have a correct transform from
    ra dec to pixels'''


    target_fluxes = []
    flux_errs = []
    for i in range(num_images):

        hdu = fits.open(filenames[i])
        x,y=getXY(hdu[1].header, np.array([target_radec[k]]))

        frame = importFramesFITS(data_path, filenames, i, 1)
        bkg = sep.Background(frame[0]) #create background profile
        bkg_rms = bkg.rms()

        GaussSigma = 2.5
        newx, newy = refineCentroid(*frame, [[x,y]], GaussSigma)
        newx = newx[0]
        newy = newy[0]

        ap_r = 5
        flux, flux_err, flags = sep.sum_circle(frame[0], newx, newy, ap_r, err=bkg_rms, bkgann = (inner_annulus, outer_annulus))
        target_fluxes.append(flux)
        flux_errs.append(flux_err)
        
    max_flux = max(target_fluxes)
    index = target_fluxes.index(max_flux)
    if(max_flux/flux_errs[index] < 10.0): 
        print("Couldnt find any image with reliable header. Best image: ", filenames[index])
        # print(max_flux,flux_errs[index], index)
        # print(target_fluxes)
        sys.exit()

    return index

    

def individual_telescope_obs():

    num_images_pertelescope = []

    for j in range(len(dirs)):
        data_path = data_parentpath + stars[k] + '/'+ dirs[j] + '/'
        filenames = sorted(glob(data_path+'*.fits.fz'))

        num_images_pertelescope.append([])
        prev_telescope_name = ''
        for i in range(len(filenames)):
            hdu = fits.open(filenames[i])

            try:
                telescope_name = hdu[1].header['TELESCOP']
            except KeyError:
                telescope_name = hdu[0].header['TELESCOP']

            if telescope_name != prev_telescope_name:
                num_images_pertelescope[j].append(1)
                prev_telescope_name = telescope_name
            else:
                num_images_pertelescope[j][-1] +=1

    return num_images_pertelescope


def get_radec_stars():

    data_path = data_parentpath + stars[k] + '/'+ dirs[0] + '/'
    filenames = sorted(glob(data_path+'*.fits.fz'))[:num_images_pertelescope[0][0]]
    num_images = len(filenames)

    x_length, y_length, num_images = getSizeFITS(filenames)

    correct_hdrfile_idx = get_correct_hdrfile_idx(num_images, filenames, data_path)

    correct_hdrframe = importFramesFITS(data_path, filenames, correct_hdrfile_idx, 1)      #data and time from 1st image

    hdu = fits.open(filenames[correct_hdrfile_idx])
    x,y=getXY(hdu[1].header, np.array([target_radec[k]]))
    target_x=x[0]
    target_y=y[0]
    target_mask = np.zeros_like(correct_hdrframe[0])
    target_mask[round(target_x-100):round(target_x+100), round(target_y-100):round(target_y+100)] = 1.0
    positions = initialFindFITS(correct_hdrframe[0], detect_thresh, target_mask)

    star_find_results = tuple(positions) #find stars on first frame

    if len(star_find_results)<20: #quit when no stars
        print("Not good image! ", filenames[0])
        sys.exit()


    before_ellipticity_filter = len(star_find_results)

    global ap_r
    if(before_ellipticity_filter) > 10000: ap_r = 5
    else: ap_r = 10

    star_find_results = tuple(star_find_results[i] for i in range(before_ellipticity_filter) if star_find_results[i][4] < 1.3)
    after_ellipticity_filter = len(star_find_results)
    print('Number of stars before and after ellipticity filter: ', before_ellipticity_filter, after_ellipticity_filter)


    #remove stars where centre is too close to edge of frame
    before_trim = len(star_find_results)
    star_find_results = tuple(x for x in star_find_results if x[0] + Edge_buffer < x_length and x[0] - Edge_buffer > 0)
    star_find_results = tuple(y for y in star_find_results if y[1] + Edge_buffer < x_length and y[1] - Edge_buffer > 0)
    after_trim = len(star_find_results)
    print('Number of stars before and after trim: ', before_trim, after_trim)

    star_find_results = np.array(star_find_results)
    sep_fluxes = star_find_results[:,5]
    if(len(star_find_results) > 2000):
        top2000idx= sep_fluxes.argsort()[-2000:]
        star_find_results = star_find_results[top2000idx]

    radii = star_find_results[:,2]  #get radius of stars in px
    global GaussSigma
    GaussSigma = np.mean(radii * 2. / 2.35) #only for refining centroid which works awfully
    print("gausssigma: ",GaussSigma)
    
    initial_positions = star_find_results[:,:2] #initial star position in x and y
    initial_positions = np.append(np.array([[target_x, target_y]]), initial_positions, axis=0)

    num_stars = len(initial_positions)      #number of stars in image
    print('number of stars found, including the target: ', num_stars) 

    hdu = fits.open(filenames[correct_hdrfile_idx]) #open fits file
    transform = wcs.WCS(hdu[1].header) #open wcs from headers
    hdu.close()
    ra_dec=getRAdec(transform, initial_positions)  #get ra dec of each star 

    return ra_dec, num_stars

def choose_referencestars():
    chosenstars_idx = []
    weights = []
    
    for measured_star in range(0,num_stars): #loop through each star
        print("On Star {} out of {}".format(measured_star+1, num_stars), end='\r')
        chosenstars_idx.append([])
        median = []
        stdev=[]
        weights.append([]) #for weighted average

        measured_star_allflux = np.concatenate([dirdata[:,measured_star,2] for dirdata in alldir_data])
        star_flux=measured_star_allflux/np.mean(measured_star_allflux) #normalized lightcurve of target star
        for star in range(0, num_stars): #loop through all stars
            if star!=measured_star: #but not our
                reference_star_allflux = np.concatenate([dirdata[:,star,2] for dirdata in alldir_data])
                ref_flux= reference_star_allflux/np.mean(reference_star_allflux)  #get their relative flux
                residual=np.subtract(ref_flux,star_flux) #substract target and reference lightcurve
                median.append(abs(np.median(residual)))
                stdev.append(np.std(residual))

        thresh_index = int(0.1 * num_stars)
        median_thresh = np.sort(median)[thresh_index]
        stdev_thresh = np.sort(stdev)[thresh_index]
        
        for star in range(0, len(median)):
            if(median[star] < median_thresh and stdev[star]<stdev_thresh):
                if(star>= measured_star): star+=1
                chosenstars_idx[measured_star].append(star)

        # if(measured_star==0):
        #         matplotlib.use('TkAgg')
        #         plt.scatter(median, stdev, s=3)
        #         plt.xlabel('median of residual fluxes')
        #         plt.ylabel('std dev of residual fluxes')
        #         plt.xscale('log')
        #         plt.yscale('log')
        #         plt.show()

        compstars_idx = chosenstars_idx[measured_star]
        for star in compstars_idx:
            reference_star_allflux = np.concatenate([dirdata[:,star,2] for dirdata in alldir_data])
            ref_flux= reference_star_allflux/np.mean(reference_star_allflux) #normalized lightcurve of target star
            try:
                weights[measured_star].append((1/stats.sigma_clipped_stats(ref_flux)[2])**2)
            except ZeroDivisionError:
                weights[measured_star].append(1)


    return chosenstars_idx, weights




'''Garvit Ag: My modified version of Roman's code:
        1) takes the targets' ra dec from the user, finds the one reference image in the sequence that is guaranteed to have the correct 
        ra dec to pixel transformation in the header using the function get_correct_hdrfile_idx. Then it cross-correlates each 
        image with the reference image to find any shifts in the pixel coords
        
        2) replaces the 'chisquare' score with a new score that finds the median and stdev of the residuals of each star 
        with the target and picks the stars with the best 10% of median and stdev as the comparison stars to compute the
        relative flux'''



data_parentpath = '/home/gagarwal/Downloads/Data/'
os.system(" ls -d {}*/ > /home/gagarwal/Downloads/temp.txt".format(data_parentpath))
stars = [starpath.split('/')[-2] for starpath in open('/home/gagarwal/Downloads/temp.txt', 'r').read().split('\n')[:-1]]

'''The user needs to provide the most accurate ra dec of the targets in a list of shape [no of targets, no of stares for each target]'''
target_radec = [[216.4454064, -3.987855961], [233.73115833, -14.31575278], [255.071998, -16.91276414], [262.8633877, -7.997966847], [267.212726, -0.649488056], [269.38794583, -12.89951389], [276.0985121, -5.615670083], [283.0641767, -3.097493986], [316.0877851, -69.8785669], [163.6290162, 25.96294055], [296.4011603, -25.95686641], [205.9105716, 8.429159262], [97.18257233, 5.485348294], [241.8800931, -4.704552696], [258.0163403, -3.392545346]]

detect_thresh=4 #star finding threshold
inner_annulus = 17 #photometry bkg inner anulus
outer_annulus = 25 #photometry bkg outer anulus
Edge_buffer= 250 #10 #to remove stars that are close to edge, I think this should be more than 20px


listt = list(range(0,len(stars)))
listt.remove(5)
for k in listt:
    print("\nStar: ", stars[k])
    
    os.system(" ls -d {}/*/ > /home/gagarwal/Downloads/temp.txt".format(data_parentpath + stars[k]))
    dirs = [dirpath.split('/')[-2]+'/' for dirpath in open('/home/gagarwal/Downloads/temp.txt', 'r').read().split('\n')[:-1]]

    savefolder = '/home/gagarwal/Downloads/relative_LCs/' + stars[k] + '/'
    os.system('mkdir -p '+savefolder)
    radec_file = open(savefolder+'ra_dec.txt', 'w')
    chosenstaridx_file = open('{}chosen_ref_stars_index.txt'.format(savefolder), 'w')

    num_images_pertelescope = individual_telescope_obs()

    radec_stars, num_stars = get_radec_stars()

    alldir_flux = []
    alldir_time = []
    alldir_data = []
    all_headertimes = []
    initial_pixelpos = []
    all_filenames = []
    numimages_cumul = [0]

    for j in range(0,len(dirs)):
        print("\nDirectory: ", dirs[j])

        ''' get list of image names to process'''
        data_path = data_parentpath + stars[k] + '/'+ dirs[j] + '/'
        all_filenames.append(sorted(glob(data_path+'*.fits.fz'))) #list of mean-stacked frames
        

        ''' get 2d shape of images, number of image in directory'''
        x_length, y_length, total_num_images = getSizeFITS(all_filenames[j])
        print ("Imported", total_num_images, "frames")

        numimages_cumul.append(numimages_cumul[j]+ total_num_images)

        data = np.empty([total_num_images, num_stars], dtype=(np.float64, 4)) #array to store data
        all_headertimes.append([])

        cumulative = 0
        for i in range(len(num_images_pertelescope[j])):
            num_images = num_images_pertelescope[j][i]
            
            filenames = all_filenames[j][cumulative: cumulative+num_images]
                
            correct_hdrfile_idx = get_correct_hdrfile_idx(num_images, filenames, data_path)
            hdu = fits.open(filenames[correct_hdrfile_idx])
            initialx, initialy=getXY(hdu[1].header, radec_stars)

            correct_hdrframe = importFramesFITS(data_path, filenames, correct_hdrfile_idx, 1)      #data and time from 1st image
            for t in range(cumulative, cumulative+num_images):

                imageFile = importFramesFITS(data_path, all_filenames[j], t, 1)
                all_headertimes[j].append(imageFile[1])  #add header time to list

                slice_size = 500
                slice = [2048 - slice_size, 2048 + slice_size]
                corr = cv2.filter2D(correct_hdrframe[0][slice[0]:slice[1], slice[0]:slice[1]], kernel=imageFile[0][slice[0]:slice[1], slice[0]:slice[1]], ddepth=-1, borderType=cv2.BORDER_CONSTANT)
                max_val = np.max(corr)
                max_idx = np.where(corr == max_val)
                # print(max_idx)

                shift = np.array([500, 500]) - np.array([max_idx[1][0], max_idx[0][0]]) 
                # print(shift)

                currentx, currenty = initialx + shift[0], initialy + shift[1]

                # if(t==0):
                #     matplotlib.use('TkAgg')

                #     plt.imshow(corr)
                #     plt.show()

                '''add up all flux within aperture'''
                bkg = sep.Background(imageFile[0]) #create background profile
                bkg_rms = bkg.rms()
                
                newx, newy = refineCentroid(*imageFile, [[currentx, currenty]], GaussSigma)
                newx = np.array(newx[0])
                newy = np.array(newy[0])

                if(i==0 and t==0):
                    initial_pixelpos.append([newx, newy])

                fluxes, flux_err, flags = sep.sum_circle(imageFile[0], newx, newy, ap_r, err=bkg_rms, bkgann = (inner_annulus, outer_annulus))
                
                data[t] = tuple(zip(newx, newy, fluxes, flux_err))


            cumulative += num_images

        alldir_data.append(data)


    chosenstars_idx, weights = choose_referencestars()
    sort_arr = np.flip(np.argsort(weights[0]))
    weights[0] = np.array(weights[0])[sort_arr]
    chosenstars_idx[0] = np.array(chosenstars_idx[0])[sort_arr]
    for i in range(len(chosenstars_idx[0])):
            chosenstaridx_file.write('{}\n'.format(chosenstars_idx[0][i]))
    print('Number of comparison stars chosen for mean lightcurve: ',len(chosenstars_idx[0]))


    rel_lcs=[]
    rel_lc_errs=[]

    for measured_star in range(0,num_stars): #loop through each star
        star_fluxes=[] #for mean lightcurve
        star_errs=[]
        
        for i in range(len(chosenstars_idx[measured_star])):
            star_flux = np.concatenate([dirdata[:,chosenstars_idx[measured_star][i],2] for dirdata in alldir_data])
            star_fluxes.append(star_flux/np.mean(star_flux)) #append those lightcurves
            star_err = np.concatenate([dirdata[:,chosenstars_idx[measured_star][i],3] for dirdata in alldir_data])
            star_errs.append(star_err/np.mean(star_flux))

        try: #get weighted average lightcurve
            mean_lc=np.average(star_fluxes,weights=weights[measured_star], axis=0)
        except ZeroDivisionError:
            mean_lc=np.average(star_fluxes, axis=0)
        weights_sq = [weight**2 for weight in weights[measured_star]]
        mean_lc_err = np.average([np.power(star_err,2) for star_err in star_errs] , weights=weights_sq, axis=0)
        mean_lc_err = mean_lc_err * np.sum(weights_sq) / np.power(np.sum(weights[measured_star]), 2)

        measuredstar_flux = np.concatenate([dirdata[:,measured_star,2] for dirdata in alldir_data])
        rel_lc=(measuredstar_flux/np.mean(measuredstar_flux))/mean_lc #get relative lightcurve
        rel_lcs.append(rel_lc)
        measuredstar_err = np.concatenate([dirdata[:,measured_star,3] for dirdata in alldir_data])
        measuredstar_err = measuredstar_err/np.mean(measuredstar_flux)
        rel_lc_err = rel_lc * np.power(np.power(measuredstar_err/(measuredstar_flux/np.mean(measuredstar_flux)), 2) + np.power(mean_lc_err/mean_lc, 2), 0.5)
        rel_lc_errs.append(rel_lc_err)


    for j in range(len(dirs)):
        data = alldir_data[j]    
        results = []
        for star in range(0,num_stars): #append results [flux, star, relative flux, flux error, x, y]
            results.append((data[:, star, 2], star, rel_lcs[star][numimages_cumul[j]: numimages_cumul[j+1]], data[:, star, 3], data[:, star, 0], data[:, star, 1], rel_lc_errs[star][numimages_cumul[j]: numimages_cumul[j+1]]))
            
        results = np.array(results, dtype = object)


        headerJD=[t[0] for t in all_headertimes[j]]
        headerJD=Time(headerJD,format='isot', scale='utc')
        headerJD.format='jd'
        headerJD = [headerJD[i].value for i in range(len(headerJD))]
        alldir_time.append(headerJD)

        
        ''' data archival '''

        lightcurve_savepath = savefolder+ dirs[j]+'/'
        # lightcurve_savepath = savefolder
        os.system('mkdir -p '+lightcurve_savepath)      #make folder to hold master bias images in

        for row in results:  # loop through each detected event
            
            savefile = lightcurve_savepath+'star' + str(row[1])+ ".txt"

            #open file to save results
            with open(savefile, 'w') as filehandle:
                
                # file header
                filehandle.write('#\n#\n#\n#\n')
                filehandle.write('#    RA Dec: %f, %f\n' %(radec_stars[row[1],0], radec_stars[row[1],1]))
                filehandle.write('#    Pixel Position: %d, %d\n' %(initial_pixelpos[j][0][row[1]], initial_pixelpos[j][1][row[1]]))
                filehandle.write('#    DATE-OBS (JD): %s\n' %(all_headertimes[j][0]))
                filehandle.write('#    Aperture radius: %s\n' %(ap_r))
                filehandle.write('#\n#\n#\n')
                filehandle.write('#filename\ttime\tflux\tflux_err\trel_flux\trel_flux_errt\tx\ty\n')

                data_err=row[3]
                x=row[4]
                y=row[5]
                relative_lightcurve=row[0]
                relative_lightcurve_err=row[6]
                
                files_to_save = [filename.split('/')[-1] for filename in all_filenames[j]]
                star_save_flux = row[0]            #part of light curve to save
        
                #loop through each frame to be saved
                for i in range(0, relative_lightcurve.shape[0]):  
                    filehandle.write('%s\t%f\t%.6f\t%f\t%f\t%f\t%ft%f\n' % (files_to_save[i], float(headerJD[i]), float(star_save_flux[i]), float(data_err[i]), float(relative_lightcurve[i]), float(relative_lightcurve_err[i]), float(x[i]), float(y[i])))


    radec_file.write('Star Index\tRa (deg)\tDec (deg)\n')
    for i in range(radec_stars.shape[0]):
        radec_file.write('%d\t%f\t%f\n' % (i, radec_stars[i][0], radec_stars[i][1]))

print ("\n")

# </3