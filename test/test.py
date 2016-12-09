__author__ = 'kyue'

import numpy as np
import timeit
import tomocuda
import tomopy

def testMedianFilter2D(size):
    '''test with random array, loop outside with a 2d cuda kernel'''
    # prjsize is z, imsize is x, y.

    # size = 13 # window size for the filter
    imsizex =2048 # image size for the input
    imsizey = 2048
    prjsize= 1

    combinedMed = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    resultscuda = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    resultscpu = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)

    # create combined noise matrix 3D
    for step in range (5, 5 + prjsize):
        im_noise = np.arange( 10, imsizey*imsizex*step+10, step ).reshape(imsizey, imsizex)
        # im_noise = np.random.rand(imsizey, imsizex)
        im_noise = im_noise.astype(np.float32)
        combinedMed[step-5]=im_noise

    start = timeit.default_timer()
    resultscpu= tomopy.median_filter_cuda(combinedMed,size=size)
    stop = timeit.default_timer()
    diff2 = stop - start
    print("end cuda filter", diff2)

    start = timeit.default_timer()
    resultscuda = tomopy.median_filter(combinedMed,size=size, ncore=1)
    stop = timeit.default_timer()
    diff1 = stop - start

    print("end cpu median filter", diff1 )

    print("the times gpu over cpu is", diff1/diff2)

    print not np.any(resultscuda-resultscpu)

def testOutlierRemoval(size):
    '''test with random array, loop outside with a 2d cuda kernel'''
    # prjsize is z, imsize is x, y.

    # size = 13 # window size for the filter
    imsizex =2048 # image size for the input
    imsizey = 2048
    prjsize= 1

    combinedMed = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    resultscuda = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    resultscpu = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)
    diff = 100

    # create combined noise matrix 3D
    for step in range (5, 5 + prjsize):
        im_noise = np.arange( 10, imsizey*imsizex*step+10, step ).reshape(imsizey, imsizex)
        # im_noise = np.random.rand(imsizey, imsizex)
        im_noise = im_noise.astype(np.float32)
        combinedMed[step-5]=im_noise

    start = timeit.default_timer()
    resultscpu= tomopy.remove_outlier_cuda(combinedMed, diff, size=size)
    stop = timeit.default_timer()
    diff2 = stop - start
    print("end cuda outlier removal", diff2)

    start = timeit.default_timer()
    resultscuda = tomopy.remove_outlier(combinedMed, diff, size=size, ncore = 1)
    stop = timeit.default_timer()
    diff1 = stop - start

    print("end cpu outlier removal", diff1 )

    print("the times gpu over cpu is", diff1/diff2)

    print not np.any(resultscuda-resultscpu)

if __name__ == '__main__':

    size = 2

    testMedianFilter2D(size)

    testOutlierRemoval(size)

