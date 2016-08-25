__author__ = 'kyue'

import tomopy

import tomoCuda
import numpy as np
import scipy
from scipy import ndimage

import matplotlib.pyplot as plt
import timeit

def testMedianFilter1():
    '''test with random array, 2d cuda kernel with a loop inside'''
    # prjsize is z, imsize is x, y.

    # print combined
    size = 15 # window size for the filter
    imsizex =2016 # image size for the input
    imsizey = 2560
    prjsize=4
    diff = 20


    # the left and right offset of the image to do the reflect mode
    loffset = size/2
    roffset = (size-1)/2

    #create test 3d array, filter size -1 = loffset+roffset
    combinedMed = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    combined = np.zeros(shape=(prjsize,imsizey+size-1,imsizex+size-1), dtype=np.float32)
    results1 = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)

    # create a gpu median filter object
    filter = tomoCuda.mFilter(imsizex, imsizey, prjsize, size)



    # create combined noise matrix 3D
    for step in range (5,5+prjsize):
        im_noise = np.arange( 10, imsizey*imsizex*step+10, step ).reshape(imsizey, imsizex)
        im_noise = im_noise.astype(np.float32)
        combinedMed[step-5]=im_noise


    start = timeit.default_timer()
    # im_med = ndimage.median_filter(im_noise, size)
    # results1 = tomopy.misc.corr.remove_outlier(combinedMed, diff, size )
    results1 = tomopy.median_filter(combinedMed,size=size)
    stop = timeit.default_timer()
    diff1 = stop - start

    print("end scipy median filter", diff1 )


    combined = np.lib.pad(combinedMed, ((0,0), (loffset, roffset),(loffset, roffset)), 'symmetric')

    im_noisecu = combined.flatten()
    im_noisecu = im_noisecu.astype(np.float32)

    start = timeit.default_timer()

    # reset the cuda image in the median filter
    filter.setCuImage(im_noisecu)

    # start to run the filter with window size
    filter.run2DLoopFilter(size)
    results2 = filter.retreive()

    stop = timeit.default_timer()

    results2 = results2.reshape(prjsize, imsizey,imsizex)

    diff2 = stop - start
    print("end cuda filter", diff2)
    print("the times gpu over cpu is", diff1/diff2)

    print not np.any(results1-results2)

def testMedianFilter2D():
    '''test with random array, loop outside with a 2d cuda kernel'''
    # prjsize is z, imsize is x, y.

    # print combined
    size = 7 # window size for the filter
    imsizex =2560 # image size for the input
    imsizey = 204
    prjsize= 721
    diff = 2000



    # the left and right offset of the image to do the reflect mode
    loffset = size/2
    roffset = (size-1)/2

    #create test 3d array, filter size -1 = loffset+roffset
    combinedMed = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    # combined = np.zeros(shape=(prjsize,imsizey+size-1,imsizex+size-1), dtype=np.float32)
    results1 = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    resultcombine = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    # create a gpu median filter object
    filter = tomoCuda.mFilter(imsizex, imsizey, prjsize, size)


    diff2 = 0
    # create combined noise matrix 3D
    for step in range (5,5+prjsize):
        # im_noise = np.arange( 10, imsizey*imsizex*step+10, step ).reshape(imsizey, imsizex)
        im_noise = np.random.rand(imsizey, imsizex)
        im_noise = im_noise.astype(np.float32)
        combinedMed[step-5]=im_noise

        im_noisecu = im_noise.astype(np.float32)
        im_noisecu=np.lib.pad(im_noisecu, ((loffset, roffset),(loffset, roffset)), 'symmetric')

        # print im_noisecu

        im_noisecu = im_noisecu.flatten()

        start = timeit.default_timer()
        filter.setCuImage(im_noisecu)
        filter.run2DFilter(size)
        results2 = filter.retreive()
        stop = timeit.default_timer()
        diff2 += stop - start


        results2=results2.reshape(imsizey,imsizex)
        resultcombine[step-5]=results2

    print("end cuda filter", diff2)
    # print combinedMed

    start = timeit.default_timer()
    results1 = tomopy.median_filter(combinedMed,size=size)
    stop = timeit.default_timer()
    diff1 = stop - start

    print("end scipy median filter", diff1 )

    print("the times gpu over cpu is", diff1/diff2)

    # print results1
    # print resultcombine
    # print results1-resultcombine
    print not np.any(results1-resultcombine)

def testMedianFilter3():
    '''test with random array, 3D cuda kernel '''
    # prjsize is z, imsize is x, y.

    # print combined
    size = 15 # window size for the filter
    imsizex =2016 # image size for the input
    imsizey = 300
    prjsize=1
    diff = 20

    # the left and right offset of the image to do the reflect mode
    loffset = size/2
    roffset = (size-1)/2

    #create test 3d array, filter size -1 = loffset+roffset
    combinedMed = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    combined = np.zeros(shape=(prjsize,imsizey+size-1,imsizex+size-1), dtype=np.float32)
    results1 = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)





    # create combined noise matrix 3D
    for step in range (5,5+prjsize):
        im_noise = np.arange( 10, imsizey*imsizex*step+10, step ).reshape(imsizey, imsizex)
        im_noise = im_noise.astype(np.float32)
        combinedMed[step-5]=im_noise


    start = timeit.default_timer()
    # im_med = ndimage.median_filter(im_noise, size)
    # results1 = tomopy.misc.corr.remove_outlier(combinedMed, diff, size )
    results1 = tomopy.median_filter(combinedMed,size=size)
    stop = timeit.default_timer()
    diff1 = stop - start

    print("end scipy median filter", diff1 )


    combined = np.lib.pad(combinedMed, ((0,0), (loffset, roffset),(loffset, roffset)), 'symmetric')

    im_noisecu = combined.flatten()
    im_noisecu = im_noisecu.astype(np.float32)

        # create a gpu median filter object
    filter = tomoCuda.mFilter(imsizex, imsizey, prjsize, size)

    start = timeit.default_timer()

    # reset the cuda image in the median filter
    filter.setCuImage(im_noisecu)

    # start to run the filter with window size
    filter.run3DFilter(size)
    results2 = filter.retreive()

    stop = timeit.default_timer()

    results2 = results2.reshape(prjsize, imsizey,imsizex)

    diff2 = stop - start
    print("end cuda filter", diff2)
    print("the times gpu over cpu is", diff1/diff2)

    print not np.any(results1-results2)

def testMedianFilterXZ():
    '''test with random array, loop outside with a 2d cuda kernel, eacho along the z axis and x'''
    # prjsize is z, imsize is x, y.

    # print combined
    size = 15 # window size for the filter
    imsizex =2560 # image size for the input
    imsizey = 204
    prjsize=3000
    diff = 20

    # the left and right offset of the image to do the reflect mode
    loffset = size/2
    roffset = (size-1)/2

    #create test 3d array, filter size -1 = loffset+roffset
    combinedMed = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    combined = np.zeros(shape=(prjsize,imsizey+size-1,imsizex+size-1), dtype=np.float32)
    results1 = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)

    # create combined noise matrix 3D
    for step in range (5,5+prjsize):
        im_noise = np.arange( 10, imsizey*imsizex*step+10, step ).reshape(imsizey, imsizex)
        im_noise = im_noise.astype(np.float32)
        combinedMed[step-5]=im_noise


    start = timeit.default_timer()
    # im_med = ndimage.median_filter(im_noise, size)
    # results1 = tomopy.misc.corr.remove_outlier(combinedMed, diff, size )
    results1 = tomopy.median_filter(combinedMed,size=size)
    stop = timeit.default_timer()
    diff1 = stop - start

    print("end scipy median filter", diff1 )


    combined = np.lib.pad(combinedMed, ((0,0), (loffset, roffset),(loffset, roffset)), 'symmetric')
    combined = combined.swapaxes(0,1)

    im_noisecu = combined.flatten()
    im_noisecu = im_noisecu.astype(np.float32)

        # create a gpu median filter object
    filter = tomoCuda.mFilter(imsizex, imsizey, prjsize, size)

    start = timeit.default_timer()

    # reset the cuda image in the median filter
    filter.setCuImage(im_noisecu)

    # start to run the filter with window size
    filter.run3DFilterXZ(size)
    results2 = filter.retreive()

    stop = timeit.default_timer()

    results2 = results2.reshape(prjsize, imsizey,imsizex)

    diff2 = stop - start
    print("end cuda filter", diff2)
    print("the times gpu over cpu is", diff1/diff2)

    # print results1
    # print results2
    # print results1-results2
    print not np.any(results1-results2)

def testRemoveOutliner1():
    '''test with random array, loop outside with a 2d cuda kernel'''
    # prjsize is z, imsize is x, y.

    # print combined
    size = 7 # window size for the filter
    imsizex =2560 # image size for the input
    imsizey = 204
    prjsize=721
    diff = 20


    # the left and right offset of the image to do the reflect mode
    loffset = size/2
    roffset = (size-1)/2

    #create test 3d array, filter size -1 = loffset+roffset
    combinedMed = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    # combined = np.zeros(shape=(prjsize,imsizey+size-1,imsizex+size-1), dtype=np.float32)
    results1 = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    resultcombine = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    # create a gpu median filter object
    filter = tomoCuda.mFilter(imsizex, imsizey, prjsize, size)

    diff2 = 0

    # create combined noise matrix 3D
    for step in range (5,5+prjsize):
        # im_noise = np.arange( 10, imsizey*imsizex*step+10, step ).reshape(imsizey, imsizex)
        im_noise = np.random.rand(imsizey, imsizex)
        im_noise = im_noise.astype(np.float32)
        combinedMed[step-5]=im_noise

        im_noisecu = im_noise.astype(np.float32)
        im_noisecu=np.lib.pad(im_noisecu, ((loffset, roffset),(loffset, roffset)), 'symmetric')
        im_noisecu = im_noisecu.flatten()

        start = timeit.default_timer()
        filter.setCuImage(im_noisecu)
        filter.run2DRemoveOutliner(size, diff)
        results2 = filter.retreive()
        stop = timeit.default_timer()
        diff2 += stop - start


        results2=results2.reshape(imsizey,imsizex)
        resultcombine[step-5]=results2
        # combinedMed[step-5]=im_noise

    print("end remove outlier", diff2)

    start = timeit.default_timer()

    results1 = tomopy.misc.corr.remove_outlier(combinedMed, diff, size )

    stop = timeit.default_timer()
    diff1 = stop - start

    print("end scipy outlier", diff1 )

    print("the times gpu over cpu is", diff1/diff2)

    print not np.any(results1-resultcombine)


# testMedianFilter1()

testMedianFilter2D()

# testMedianFilter3()

# testMedianFilterXZ()

testRemoveOutliner1()