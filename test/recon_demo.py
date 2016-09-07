# -*- coding: utf-8 -*-
"""
Created on Sat May 23 12:26:06 2015

@author: xhxiao
"""

#from tomopy.io.reader import *
#import numpy as np
#import os.path
#from numpy.testing import assert_allclose
import tomopy 
import os
import numpy as np
import dxchange

import timeit
import tomocuda

from tomopy.recon.rotation import write_center
from tomopy.recon.algorithm import recon
from scipy import misc


####----------------- Section 1: Input file -- Staragt -----------------  
#data_dir = '/media/2BM_Backup42/2015_02/Florian/gypsum/'
data_dir = '/data2/XiaoData/reader1/'
#data_dir = '/media/2BM_Backup36/2014_11/Commissioning/Bobby3/a17_pos1_20x_edge_90mm_0.75DegPerSec_180Deg_100msecExpTime_2000proj_Rolling_10umLuAG_USArm1.1_1mmAl_1mmGlass_2mrad_mirror_AHutch'

#data_dir = '/media/OCT14A/APS_2-BM_2014Oct/SPMEL00_0'
#file = 'proj_111.hdf'
file = 'proj_4.hdf'

output_dir = data_dir
file_name = os.path.join(data_dir, file)
output_file = output_dir+'/recon_'+file.split(".")[-2]+'/recon_'+file.split(".")[-2] +'_'
####----------------- Section 1: Input file -- End -----------------  

####----------------- Section 2: Parameter configuration -- Start -----------------  
offset = 100
total_num_slices = 4 # total slice want to do 1 trial 50  initial 300
chunk_size = 2 # memory 50 initial 100
if chunk_size > total_num_slices:
    chunk_size = total_num_slices

margin_slices = 2 # set to 0 fft boundary  initial set to 30,
num_chunk = np.int(total_num_slices/chunk_size) + 1
if total_num_slices == chunk_size:
    num_chunk = 1
    
z = 8
eng = 60
pxl = 1.3e-4
rat = .25e-04
    
#z = 5
#eng = 25
#pxl = 1.3e-4
#rat = .2e-03    

zinger_level = 200
####----------------- Section 2: Parameter configuration -- End -----------------   

#######----------------- Section 3: Finding center -- Start -----------------
#data, white, dark = tomopy.io.exchange.read_aps_2bm(file_name,sino=(700,720,1))
#data_size = data.shape
#theta = np.linspace(0,np.pi,num=data_size[0])
#
###data[0:11,:,:] = data[12:23,:,:]
#data = tomopy.misc.corr.remove_outlier(data,zinger_level,size=15,axis=0)
#white = tomopy.misc.corr.remove_outlier(white,zinger_level,size=15,axis=0)
#                            
## there is modification in below normalize routine
#data = tomopy.prep.normalize.normalize(data,white,dark)
#data = tomopy.prep.normalize.normalize_bg(data,air=10)
#
#data = tomopy.prep.stripe.remove_stripe_fw(data,level=6,wname='sym16',sigma=2,pad=True)
#
#data = tomopy.prep.phase.retrieve_phase(data,pixel_size=pxl,dist=z,energy=eng,alpha=rat,pad=True)
#
#data_size = data.shape
# 
#cs = data_size[2]/2-0
## there is modification in the below routine
#write_center(data[:,9:11,:], theta, dpath='/media/2BM_Backup42/data_center/', 
#             cen_range=(cs,cs+10,0.5),emission=False)
#######----------------- Section 3: Finding center -- End -----------------

center = 1284.5

####----------------- Section 4: Full reconstruction -- Start -----------------
for ii in xrange(num_chunk):
    if ii == 0:
        SliceStart = offset + ii*chunk_size
        SliceEnd = offset + (ii+1)*chunk_size
    else:
        SliceStart = offset + ii*(chunk_size-margin_slices)
        SliceEnd = offset + SliceStart + chunk_size
        if SliceEnd > (offset+total_num_slices):
            SliceEnd = offset+total_num_slices

    # read chunk data
    filtersize=15

    # start = timeit.default_timer()

    # data, white, dark = tomopy.io.exchange.read_aps_2bm(file_name,sino=(SliceStart,SliceEnd,1))
    datasl, white, dark = dxchange.read_aps_2bm(file_name,sino=(SliceStart,SliceEnd,1))

    data= datasl[10:1000,:,:]

    data_size = data.shape
    # theta = np.linspace(0,np.pi,num=data_size[0])

    stop = timeit.default_timer()
    # print("end read 2BM data", (stop - start))
    #print stop1 - start1

    # replace corrupted images if there is any
    # data[0,:,:] = data[1,:,:]



    print("start remove outlier")
    start = timeit.default_timer()

    data = data.astype(np.float32)

    # print data

    # remove zingers (pixels with abnormal counts), size initial is 15
    data1 = tomopy.misc.corr.remove_outlier(data,zinger_level,size=filtersize,axis=0)
    # white = tomopy.misc.corr.remove_outlier(white,zinger_level,size=15,axis=0)

    # print data1

    stop = timeit.default_timer()
    diff1 = stop - start
    print("end remove outlier", diff1)


    print("start gpu remove outlier")

    loffset = filtersize/2
    roffset = (filtersize-1)/2

    start = timeit.default_timer()

    filter = tomocuda.mFilter(data_size[2], data_size[1], data_size[0], filtersize)
    im_noisecu = np.lib.pad(data,((0,0), (loffset, roffset),(loffset, roffset)), 'symmetric')
    # print im_noisecu

    im_noisecu = im_noisecu.flatten();
    im_noisecu = im_noisecu.astype(np.float32)

    filter.setCuImage(im_noisecu)
    filter.run3DRemoveOutliner(filtersize, zinger_level)
    data2 = filter.retreive()
    stop = timeit.default_timer()

    data2 = data2.reshape(data_size[0], data_size[1],data_size[2])
    # print data2

    diff2 = stop - start
    print("end gpu remove ouliner", diff2)
    print("the times gpu over cpu is", diff1/diff2)

    print not np.any(data1-data2)

    # print("start normalize")
    # # normalize projection images; for now you need to do below two operations in sequence
    # data = tomopy.prep.normalize.normalize(data,white,dark)
    # data = tomopy.prep.normalize.normalize_bg(data,air=2)
    #
    # stop3 = timeit.default_timer()
    # print("end normalize", (stop3 - stop2))
    #
    # # remove stripes in sinograms
    # data = tomopy.prep.stripe.remove_stripe_fw(data,level=9,wname='sym16',sigma=2,pad=True)
    #
    # stop4 = timeit.default_timer()
    # print("end remove stripe", (stop4 - stop3))
    #
    # # phase retrieval
    # data = tomopy.prep.phase.retrieve_phase(data,pixel_size=pxl,dist=z,energy=eng,alpha=rat,pad=True)
    #
    # stop5 = timeit.default_timer()
    # print("end retrieve phase", (stop5 - stop4))
    #
    # # tomo reconstruction
    # data_recon = recon(data,theta,center=center, algorithm='gridrec')
    #
    # stop6 = timeit.default_timer()
    # print("end gridrec", (stop6 - stop5))
    #
    # # save reconstructions
    # # tomopy.io.writer.write_tiff_stack(data_recon[np.int(margin_slices/2):(SliceEnd-SliceStart-np.int(margin_slices/2)),:,:],
    # #                                              axis = 0,
    # #                                              fname = output_file,
    # #                                              start = SliceStart+np.int(margin_slices/2),
    # #                                              overwrite = True)
    # dxchange.write_tiff_stack(data_recon[np.int(margin_slices/2):(SliceEnd-SliceStart-np.int(margin_slices/2)),:,:],
    #                          axis = 0,
    #                          fname = output_file,
    #                          start = SliceStart+np.int(margin_slices/2),
    #                          overwrite = True)
    # stop7 = timeit.default_timer()
    # print("end write tiff", (stop7 - stop6))
########----------------- Section 4: Full reconstruction -- End -----------------
