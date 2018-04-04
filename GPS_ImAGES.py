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
import bisect

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
    
    def testLayers(self, const_x, const_y, d, shape, gaus=False):
        """
        Creates a single 'continuous' plane based on the given parameters
        anything in the z direction that goes close to 5 and max_z-5 becomes 5 and max_z-5 respectively
        INPUTS:
        const_x = array of constants for the x axis such that [a,b, ..., n] = ax + bx^2 + ... + nx^
        const_y = array of constants for the y axis MAKE SURE THEY ARE THE SAME SIZE!
        d = distance between layers
        """
        g = np.zeros(shape);
        if (gaus == True):
            gx = signal.gaussian(shape[0], std=shape[0]/9);
            gy = signal.gaussian(shape[1], std=shape[1]/9);
            gm = np.meshgrid(gx, gy);
            g = gm[0]*gm[1]*d;
                        
            
        x, y = np.meshgrid(range(0,shape[0]), range(0,shape[1]));
        i = 1;
        z = np.zeros(shape);
        dx = np.zeros(shape);
        dy = np.zeros(shape);
        for a, b in zip(const_x, const_y):
            z = a*x**i + b*y**i + z;
            dx = i*a*(x**(i-1)) + dx;
            dy = i*b*(y**(i-1)) + dy;
            i += 1;
        
        m = (d+g)/np.sqrt(dx**2 + dy**2 + 1**2);        
        
        output = np.zeros([shape[0],shape[1], z.max()+2*d+15]);
        for xi in range(0,shape[0]):
             for yi in range(0,shape[1]):
                output[xi,yi,z[xi,yi]+5] = 255;
                sx = xi-dy[xi,yi]*m[xi,yi];
                sy = yi-dx[xi,yi]*m[xi,yi];
                if sx >= 0 and sx < shape[0] and sy >= 0 and sy <= shape[1]:
                    output[sx, sy, z[xi,yi]+m[xi,yi]] = 125;
        
        return output;
    
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
        
    def map_peaks(self, fp, dis):
        """
        Creates a 3d array of the location of peaks
        INPUTS:
        fp, dis = pickle arrays
        OUTPUS:
        3d array of peaks
        """
        plots = np.zeros([max(fp['x'])+1, max(fp['y']+1), 300]);
        for f in fp.index:
            for p in dis[dis['fp_id']==f].iterrows():
                plots[fp.loc[f]['x'], fp.loc[f]['y'], p[1]['loc']]=255;
        
        return plots;
    
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
    
    def saveAsPickle(self, image, name):
        """
        Extracts the FP and Dis and saves it in a pickle DF format
        INPUTS:
        image = 3d array representing image
        OUTPUTS:
        none
        """
        Col_FP = ['x','y','z'];
        Col_Dis = ['fp_id', 'loc', 'amp'];
        i = 0;
  
        #DF_FP = pd.DataFrame(columns=Col_FP);
        #DF_Dis = pd.DataFrame(columns=Col_Dis);
        fps = np.array([[0,0,0]]);
        dis = np.array([[0,0,0]]);
        counter = 0;

        for x in range(0,image.shape[0]):
            for y in range(0, image.shape[1]):
                #fp, dist = da.halfMax1d(image[x,y,:], x, y, i);
                fp, dist = da.peaks(image[x,y,:],x,y,i);
                #DF_FP = DF_FP.append(pd.DataFrame([fp], columns = Col_FP), ignore_index = True);
                if not fp==None:                
                    fps = np.insert(fps,len(fps),fp,axis=0);
                    if dist == None:
                        counter += 1;
                    else:
                        #DF_Dis = DF_Dis.append(pd.DataFrame(dist, columns = Col_Dis), ignore_index = True);
                        dis = np.insert(dis, len(dis),dist, axis=0);
                    i += 1;
        
        DF_FP = pd.DataFrame(data=fps, columns=Col_FP);
        DF_Dis = pd.DataFrame(data=dis, columns=Col_Dis);
    
        DF_FP.to_pickle(name+"FP_pd");
        DF_Dis.to_pickle(name+"Dis_pd");
    
    def saveAsTiff_Dis(self, image, name, types=np.int8):
        """
        Extracts the FP and Dis and saves it in a pickle DF format
        INPUTS:
        image = 3d array representing image
        OUTPUTS:
        none
        """
        i = 0;

        fps = np.zeros([image.shape([0]), image.shape([1])]);
        dis = np.zeros([image.shape([0]), image.shape([1])]);
        counter = 0;

        for x in range(0,image.shape[0]):
            for y in range(0, image.shape[1]):
                #fp, dist = da.halfMax1d(image[x,y,:], x, y, i);
                fp, dist = da.peaks(image[x,y,:],x,y,i);
                #DF_FP = DF_FP.append(pd.DataFrame([fp], columns = Col_FP), ignore_index = True);
                if not fp==None:                
                    fps[x,y] = fp;
                    if dist == None:
                        counter += 1;
                    else:
                        dis[x,y] = dist;
                    i += 1;

        tifffile.imsave(name+'_fps.tiff', fps.astype(types));
        tifffile.imsave(name+'_dis.tiff', fps.astype(types));
        return fps, dis;
        

class DataAcqusition():
    #All Data Acqusition Algorithms Go Here
    def __none(self):
        print "place holder";
    def max1d(self, Arr1D, x, y, j, Pk_Thres=0.1):
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
        Maxe = [];
        
        #Arr1D[Arr1D < np.max(Arr1D) * Thres] = np.max(Arr1D) * Thres;
        Maxes = signal.argrelmax(Arr1D, order=5)[0];
        difMaxes = signal.argrelmax(np.diff(Arr1D), order=5)[0];
        
        if Maxes.shape[0] <= 0:
          FP = float('nan');
        if Maxes.shape[0] == 1:
          [FP, temp, nahh] = self.__detHM(Maxes[0], difMaxes, Arr1D);
          Loc = [float('nan')];
          Amp = [float('nan')];
        if Maxes.shape[0] > 1:
          l = True;
          k = 0;
          """while l:
              FP, temp = self.__detHM(Maxes[k], difMaxes, Arr1D);
              if not np.isnan(FP):
                  l = False; #exit loop
              elif k == (Maxes.shape[0] - 1):
                  l = False; #too big
              k += 1;"""
          for i in range(k, Maxes.shape[0]):
              tLoc, tAmp, tMax = self.__detHM(Maxes[i], difMaxes, Arr1D);
              """if not np.isnan(tLoc):
                  Loc.append(tLoc - FP);
                  Amp.append(tAmp);
                  js.append(j);"""
              if not np.isnan(tLoc):
                  Loc.append(tLoc);
                  Amp.append(tAmp);
                  js.append(j);
                  Maxe.append(tMax);
          ampMax = np.argmax(Maxe);
          FP = Loc[ampMax];
          del Loc[:ampMax+1];
          Loc[:] = [x - FP for x in Loc];
          del Amp[:ampMax+1];
          del js[:ampMax+1];
        
        Dis = np.array([js, Loc, Amp]).T;
        
        if len(js) <= 0:
          Dis = float('nan');
        
        return [x, y, FP], Dis;
    
    def __detHM(self, argMax, diffMaxes, Arr1D):
        """
        Private method to determine the Halfmax point
        """
        if diffMaxes.shape[0] > 0:
            if argMax < diffMaxes[0]:
                return [float('nan'), float('nan'), float('nan')];
            else:
                #diffMax = np.max(diffMaxes[diffMaxes<argMax]); #Location of the POI
                diffMax = diffMaxes[bisect.bisect_left(diffMaxes, argMax) - 1];
                return [diffMax, Arr1D[argMax] - Arr1D[diffMax], Arr1D[argMax]];
        else:
            return [float('nan'), float('nan'), float('nan')];
    
    def peaks(self, Arr1D, x, y , j):
        #Determine FP
        difMaxes = signal.argrelmax(np.diff(Arr1D), order=5)[0]; #Use this to determine the number of half maximums
        if len(difMaxes) > 0:        
            FP = difMaxes[bisect.bisect_left(difMaxes, Arr1D.argmax()) - 1]; #location of FP, by looking for left half_max from largest peak
        
            FP_ind = np.where(difMaxes==FP)[0][0]; #location of FP wrt difMaxes
            dis_np = None;

            if len(difMaxes) > FP_ind + 1: #has distances after FP
                diffs = difMaxes[FP_ind:];
                dis_np = np.zeros([len(diffs)-1, 3]);        
            
                for i in range(len(diffs)-1):
                    dis_np[i,0] = j; #index
                    dis_np[i,1] = diffs[i+1] - FP; #z-difference
                    dis_np[i,2] = Arr1D[diffs[i+1]]/Arr1D[FP];#amplitude/Max_amplitude
        
            return [x, y, FP], dis_np;
        else: 
            return None, None
    
    def histogram(self, distances, numbins):
        """
        Create histogram of distances
        INPUT:
        distances = the panada table for distances [fp_index, loc, amplitude]
        bins = number of divisions in histogram
        OUTPUT: 
        n = array of counts
        bins = edges of bins
        """
        n, bins, temp = plt.hist(distances['loc'], bins=numbins);
        return n, bins;
    
    def coverage(self, fps, distances, upr_bnd, lwr_bnd):
        """
        Determines coverage of peaks in a given upper and lower bound of peaks
        INPUT:
        fps = panda table of first peak >> x || y || z
        distances = panda table of distances >> fp_id || loc || amp
        upr_bnd, lwr_bnd = upper and lower bounds of peaks (loc)
        OUTPUT:
        coverage = 2d array of counts at the respective (x,y) coordinate
        """
    
        temp_dis = distances.ix[(distances['loc'] <= upr_bnd) & (distances['loc'] >= lwr_bnd)];

        coverage = np.zeros([int(max(fps['x']))+1, int(max(fps['y']))+1]);
        
        i = 0;
        for row_dis in temp_dis.iterrows():
            row_fp = fps.loc[int(row_dis[1]['fp_id'])];
            coverage[int(row_fp['x']), int(row_fp['y'])] = coverage[int(row_fp['x']), int(row_fp['y'])] + 1;
            i+=1;            
            
        if coverage.max() > 0:
            coverage = coverage / coverage.max() * 255;
        
        return coverage;
    
    def second_peak_difference(self, fps1, distances1, fps2, distances2):
        """
        Determines height of sp1 and subtracts it from the height of sp2 .
        INPUT:
        fps1, distances1, fps2, distances2 = are all panda tables of first peak and distances
        OUTPUT:
        difference = 2d array of differences sp2 - sp1
        """
        sp1 = np.ones([max(fps1['x'])+1,max(fps1['y'])+1])*100
        sp2 = np.ones([max(fps1['x'])+1,max(fps1['y'])+1])*200
        
        for x in range(max(fps1.x)):
            for y in range(max(fps1.y)):
                i = fps1[(fps1.x == x) & (fps1.y == y)].index;
                j = fps2[(fps2.x == x) & (fps2.y == y)].index;
                
                if len(i) == 1: 
                    dis = distances1[distances1.fp_id == i[0]]["loc"]
                    if len(dis) > 0:
                        sp1[x,y] = min(dis);
                if len(j) == 1: 
                    dis = distances2[distances2.fp_id == j[0]]["loc"]
                    if len(dis) > 0:
                        sp2[x,y] = min(dis);
        
        return sp2-sp1;
    
    def flattenFP(self, image_array):
        """
        Creates an image with flattened and spliced
        INPUT:
        image_array = 3d array of image stack
        OUTPUT:
        fp = 3d array of first peaks, value of 255 at each peak
        output_array = 3d array of flattened image stack
        """
        dis = 200;
        order = 5;
        #GET FIRST PEAKS
        zs = np.ones([image_array.shape[0],image_array.shape[1]])* float('nan');
        
        for x in range(0, image_array.shape[0]):
            for y in range(0, image_array.shape[1]):
                fp, nm = self.halfMax1d(image_array[x,y],x,y,0);
                #fp, nm = self.max1d(image_array[x,y], x, y, 0);
                zs[x,y] = fp[2];
        
        grad = np.gradient(zs.astype(float));
        output_array = np.zeros([image_array.shape[0], image_array.shape[1], dis]);
        #fp = np.zeros([image_array.shape[0], image_array.shape[1], np.nanmax(zs)-np.nanmin(zs) + 10]);
        print('done upa');
        
        #Place in new array 
        for x in range(0, image_array.shape[0]):
            for y in range(0, image_array.shape[1]):
                dx, dy, dz = grad[0][x,y], grad[1][x,y], -1.;
                unit_dis = sqrt(dx**2+dy**2+dz**2);
                m = 20./unit_dis
                x0, y0, z0 = x + m*dx, y + m*dy, zs[x,y] + m*dz;
                m = (dis-20.)/unit_dis
                x1, y1, z1 = x - m*dx, y - m*dy, zs[x,y] - m*dz;
                xi = np.linspace(x0, x1, dis); 
                yi = np.linspace(y0, y1, dis);
                zj = np.linspace(z0, z1, dis);
                
                if grad[0][x,y] <= 10. and grad[0][x,y] >= -10. and grad[1][x,y] <= 10. and grad[1][x,y]>= -10. and not np.isnan(zs[x,y]):
                    output_array[x,y,:] = ndimage.map_coordinates(image_array, np.vstack((xi,yi,zj)), prefilter=False);
                    #fp[x,y,zs[x,y]+5] = 255;                
                
        return [zs, output_array, grad];

ip = ImagePrep()
da = DataAcqus

#Import Image (Windows or Linux)
#im = ip.importDiacom("Z:\\Au_Aaron\\06-Guinea Pig OCT\\Raw Data\\AA-01046-N13_Abd_Ear-032416\\N13Ear\\PAT1\\20160324\\2_OCT",[256,356,150]);
#im = ip.importDiacom("//home//yipgroup//Current//Au_Aaron//06-Guinea Pig OCT//Raw Data//AA-01046-N13_Abd_Ear-032416//N13Ear//PAT1//20160324//2_OCT",[1024,1024,300])

#Perform Gaussian Images
t_s =time.clock();
small_psf = [8,8,8];
large_psf = [14, 14, 14];
small_gauss = ip.gaussFilter(im, small_psf);
large_guass = ip.gaussFilter(im, large_psf);
t_g = time.clock();
im_dog = ip.dogFilter(im, small_psf, large_psf);
t_dog = time.clock();
ip.saveAsTiff(small_gauss, 'smallGauss.tiff', np.float32);
ip.saveAsTiff(large_gauss, 'largeGuass.tiff', np.float32);
ip.saveAsTiff(im_dog, 'DoF.tiff', np.float32);
t_save = time.clock();

print("Gauss Filter: {}, DoG: {}, Save: {}, Total: {}".format(t_g-t_s, t_dog-t_g, t_save-t_dog, t_save-t_s));

#Acquire two layers
t_s2 = time.clock();
fps, dis = ip.saveAsTiff_Dis(im_dog, 'Unflattened', np.float32);
t_2l = time.clock();
#Overlay fps and dis 
unflat_overlay = im;
for x in range(len(im.shape[0])):
	for y in range(len(im.shape[1])):
		unflat_overlay[x,y,fps[x,y]] = 255;
		unflat_overlay[x,y,dis[x,y]] = 255;
t_o = time.clock();
ip.saveAsTiff(unflat_overlay, 'unflat_overlay.tiff', np.float32);
t_save = time.clock();

print("Layers: {}, Overlay: {}, Save: {}, Total: {}".format(t_2l-t_s2, t_o-t_2l, t_save-t_o, t_save-t_s2));

#Flatten
t_s2 = time.clock();
[fps, oa, grad] = da.flattenFP(im_dog);
#Acquire two layers
t_f = time.clock();
fps, dis = ip.saveAsTiff_Dis(oa, 'Flattened', np.float32);
t_2l = time.clock();
#Overlay fps and dis
flat_overlay = oa;
for x in range(len(oa.shape[0])):
	for y in range(len(oa.shape[1])):
		flat_overlay[x,y,fps[x,y]] = 255;
		flat_overlay[x,y,dis[x,y]] = 255;
t_o = time.clock();
ip.saveAsTiff('flat.tiff', oa, np.float32);
ip.saveAsTiff('flat_overlay.tiff', flat_overlay, np.float32);
ip.saveAsTiff("gradx.tiff", grad[0]);
ip.saveAsTiff("grady.tiff", grad[1]);
t_save = time.clock();

print("Flatten: {}, Layers: {}, Overlay: {}, Save: {}, Total {}".format(t_f-ts2, t_2l-t_f, t_o-t_2l, t_save-t_o, t_save-t_s2));
print("FINISHED: {}", format(t_save-t_s));

'''#img = tifffile.imread("/home/yipgroup/image_store/Au_Aaron/DoG/G8-G14-1.tif");
img = np.transpose(img);
Start_X = 125;
Start_Y = 625;
img = np.transpose(img[:,Start_Y:Start_Y+256,Start_X:Start_X+256]);

#img = ip.testLayer([5,10,20],[20,10,5],[256,256,256])
#img2 = ip.gaussFilter(img, [5,5,5])

img = tifffile.imread("C:\Users\Aaron Au\Desktop\with_pandas\img.tiff")

[fps, oa, grad] = da.flattenFP(img)

#ip.saveAsTiff("/home/yipgroup/image_store/Au_Aaron/DoG/img.tiff", img, np.float32);

ip.saveAsPickle(img, "C:\Users\Aaron Au\Desktop\with_pandas\\img_");

for x in range(0,oa.shape[0]):
    for y in range(0, oa.shape[1]):
        img[x,y,fps[x,y]] = 255.;

ip.saveAsTiff("C:\Users\Aaron Au\Desktop\with_pandas\\flattened.tiff", oa, np.float32);
ip.saveAsTiff("C:\Users\Aaron Au\Desktop\with_pandas\\fp.tiff", fps);
ip.saveAsTiff("C:\Users\Aaron Au\Desktop\with_pandas\\img_layer.tiff", img, np.float32);
ip.saveAsTiff("C:\Users\Aaron Au\Desktop\with_pandas\\gradx.tiff", grad[0]);
ip.saveAsTiff("C:\Users\Aaron Au\Desktop\with_pandas\\grady.tiff", grad[1]);

ip.saveAsPickle(oa, "C:\Users\Aaron Au\Desktop\with_pandas\\layer_");
        
print("done all")"""
"""da = DataAcqusition();
ip = ImagePrep();
pic = tifffile.imread("C:\Users\Aaron Au\Desktop\with_pandas\\flattened.tiff")
ip.saveAsPickle(pic, "C:\Users\Aaron Au\Desktop\with_pandas\\test")"""
