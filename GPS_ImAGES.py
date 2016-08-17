# -*- coding: utf-8 -*-
"""
GUINEA PIG SKIN IMAGE ANALYSIS GESTATION ESTIMATION SUITE (GPS ImAGES)

Created on Wed Aug 17 09:16:22 2016

Last updated 8/17/2016

@author: Aaron Au
"""
import dicom
import glob as gb
import numpy as np
from math import *
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndimage
from scipy import signal;
import time
import tifffile

class ImagePrep():
    #All Postprocessing Algorithms Go Here
    def importDiacom(self,directory, max_size):
        """
        Imports a folder of diacom files into a np array
        INPUTS:
        directory = folder containing all the images
        max_size = 3 x 1 array containing maximum size of image; [x, y, z]
        OUTPUT:
        3D array of values corresponding to image
        """
        [Max_X, Max_Y, Max_Z] = max_size;    
        
        start = time.clock();
        image_names = gb.glob(directory+"/*"); #Get all files in that directory
        image_names.sort(); #Sort according to image name
        total_images = len(image_names);
        if total_images > Max_Z:
            total_images = Max_Z;
        
        if total_images > 0:
            RefDs = dicom.read_file(image_names[0]);
            Img = RefDs.pixel_array; #  Call one image to get the image size
        
            Pix_X = Img.shape[1];    #Check for X 
            if Img.shape[1] > Max_X:
                Pix_X = Max_X;
            Pix_Y = Img.shape[0];    #Check for Y 
            if Img.shape[0] > Max_Y:
                Pix_Y = Max_Y;
            
            #Import images into array
            OrImage = np.zeros([Pix_X,Pix_Y,total_images+1]);
            for i in range(0, total_images):
                RefDs = dicom.read_file(image_names[i]);
                OrImage[:,:,i] = RefDs.pixel_array[:Pix_X, :Pix_Y];
            print("importDiacom done, time elapsed :" + str(time.clock()-start));
            return OrImage;
        else:
            print("importDiacom fail, no images");
    
    def gaussFilter(self, array, sig):
        """
        Creates a gaussian filter of 3d array
        INPUTS:
        array = 3d array of image
        sig = 3 x 1 array containg std; [x, y, z]
        OUTPUTS:
        Gaussian Filtered array 
        """        
        start = time.clock();
        ConvoArray = ndimage.gaussian_filter(array, sigma=(sig[0], sig[1], sig[2]), order=0);
        print("gaussFilter done, time elapsed :" + str(time.clock()-start));
        return ConvoArray;
    
    def dogFilter(self, array, small_psf, large_psf):
        """
        Performs a difference of gaussians on the image
        INPUTS:
        array = 3d array of image
        small_psf = 3 x 1 array containing std; [x,y,z]
        large_psf = 3 x 1 array containing std of larger psf; [x,y,z]
        OUTPUTS:
        Difference of Gaussians array
        """
        start = time.clock();
        SmallArray = ndimage.gaussian_filter(array, sigma=(small_psf[0], small_psf[1], small_psf[2]), order=0);
        LargeArray = ndimage.gaussian_filter(array, sigma=(large_psf[0], large_psf[1], large_psf[2]), order=0);
        DoGArray = SmallArray - LargeArray;
        print("dogFilter done, time elapsed :" + str(time.clock()-start));
        return DoGArray;    
    
    def saveAsTiff(self, file_name, array, types=np.int8):
        """
        Save 3d array into tiff format
        INPUTS:
        file_name = directory + filename with .tiff as the ending
        array = 3d array
        types = OPTIONAL save values as different types (default: np.int8)
        OUTPUTS:
        none
        """
        tifffile.imsave(file_name, array.astype(types));

class DataAcqusition():
    #All Data Acqusition Algorithms Go Here
    def __none(self):
        print "place holder";
    def max1d(self, Arr1D, x, y, j, Pk_Thres):
        """
        Determine the First Peak and distance of all following peaks afterwards
        INPUTS:
        Arr1D = 1D Array describing profile in Z direction
        x, y = integer describing location of 1D profile
        j = integer describing index of FP
        Pk_Thres = threshold where peaks only occur where > max(Arr1D)*thresh 
        OUTPUTS:
        First Peak values [x,y, z], List of Distances [index of FP, location, amplitude]
        """
        Loc = [];
        Amp = [];
        js = [];
        
        Arr1D[Arr1D < np.max(Arr1D) * Pk_Thres] = np.max(Arr1D) * Pk_Thres;
        Maxes = signal.argrelmax(Arr1D)[0];
        
        if Maxes.shape[0] <= 0:
          FP = float('nan');
        elif Maxes.shape[0] == 1:
          FP = Maxes[0];
          Loc = [float('nan')];
          Amp = [float('nan')];
        else: #Maxes.shape[0] > 1
          FP = Maxes[0];
          for i in range(1, Maxes.shape[0]):
              Loc.append(Maxes[i] - Maxes[0]);
              Amp.append(Arr1D[Maxes[i]]/np.max(Arr1D));
              js.append(j);
        
        Dis = np.array([js, Loc, Amp]).T;
        
        if len(js) <= 0:
          Dis = float('nan');
        
        return [x, y, FP], Dis;
    
    def halfMax1d(self, Arr1D, x, y, j):
        """
        Determine the First Peak (based on Half Maximum) and distance of all following peaks afterwards
        INPUTS:
        Arr1D = 1D Array describing profile in Z direction
        x, y = integer describing location of 1D profile
        j = integer describing index of FP
        OUTPUTS:
        First Peak values [x,y, z], List of Distances [index of FP, location, amplitude]
        """
        Loc = [];
        Amp = [];
        js = [];
        
        #Arr1D[Arr1D < np.max(Arr1D) * Thres] = np.max(Arr1D) * Thres;
        Maxes = signal.argrelmax(Arr1D, order=5)[0];
        difMaxes = signal.argrelmax(np.diff(Arr1D), order=5)[0];
        
        if Maxes.shape[0] <= 0:
          FP = float('nan');
        if Maxes.shape[0] == 1:
          [FP, temp] = self.__detHM(Maxes[0], difMaxes, Arr1D);
          Loc = [float('nan')];
          Amp = [float('nan')];
        if Maxes.shape[0] > 1:
          l = True;
          k = 0;
          while l:
              FP, temp = self.__detHM(Maxes[k], difMaxes, Arr1D);
              if not np.isnan(FP):
                  l = False; #exit loop
              elif k == (Maxes.shape[0] - 1):
                  l = False; #too big
              k += 1;
          for i in range(k, Maxes.shape[0]):
              tLoc, tAmp = self.__detHM(Maxes[i], difMaxes, Arr1D);
              if not np.isnan(tLoc):
                  Loc.append(tLoc - FP);
                  Amp.append(tAmp);
                  js.append(j);
        
        Dis = np.array([js, Loc, Amp]).T;
        
        if len(js) <= 0:
          Dis = float('nan');
        
        return [x, y, FP], Dis;
    
    def __detHM(argMax, diffMaxes, Arr1D):
        """
        Private method to determine the Halfmax point
        """
        if argMax < diffMaxes[0]:
            return [float('nan'), float('nan')]
        else:
            diffMax = np.max(diffMaxes[diffMaxes<argMax]); #Location of the POI
            return [diffMax, Arr1D[argMax] - Arr1D[diffMax]];
    
    def histogram(distances, numbins):
        """
        Create histogram of distances
        INPUT:
        distances = the panada table for distances [fp_index, loc, amplitude]
        bins = number of divisions in histogram
        OUTPUT: 
        n = array of counts
        bins = edges of bins
        """
        n, bins, temp = plt.hist(distances['loc']);
        return n, bins;
    
    def coverage(fps, distances, upr_bnd, lwr_bnd):
        """
        Determines coverage of peaks in a given upper and lower bound of peaks
        INPUT:
        fps = panda table of first peak >> x || y || z
        distances = panda table of distances >> fp_id || loc || amp
        upr_bnd, lwr_bnd = upper and lower bounds of peaks (loc)
        OUTPUT:
        coverage = 2d array of counts at the respective (x,y) coordinate
        """
        temp_dis = dis.ix[(distances['loc'] <= upr_bnd) & (distances['loc'] >= lwr_bnd)];

        coverage = np.zeros([int(max(fps['x']))+1, int(max(fps['y']))+1]);
        
        for row_dis in temp_dis.iterrows():
            row_fp = fps.loc[row_dis[1]['fp_id']];
            coverage[int(row_fp['x']), int(row_fp['y'])] = coverage[int(row_fp['x']), int(row_fp['y'])] + 1;
            
        if coverage.max() > 0:
            coverage = coverage / coverage.max() * 255;
        
        return coverage;
        

#ip = ImagePrep()
      
#im = ip.importDiacom("Y:\\Au_Aaron\\06-Guinea Pig OCT\\Raw Data\\AA-01046-N13_Abd_Ear-032416\\N13Ear\\PAT1\\20160324\\2_OCT",[256,256, 200])