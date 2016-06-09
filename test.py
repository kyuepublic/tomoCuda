import tomoCuda
import numpy as np
import scipy
from scipy import ndimage

import matplotlib.pyplot as plt
import timeit
#import numpy.testing as npt

def test1():
    '''test with random array'''
    #arr = np.array(np.arange( 10, 49*5+10, 5 ), dtype=np.float32)

    size = 4 # window size for the filter
    imsize = 7 # image size for the input


    # the left and right offset of the image to do the reflect mode
    loffset = size/2
    roffset = (size-1)/2

    # create a gpu median filter object
    filter = tomoCuda.mFilter(imsize, imsize, size)



    for step in range (5,7):

        im_noise = np.arange( 10, imsize*imsize*step+10, step ).reshape(7, 7)
        im_noise = im_noise.astype(np.float32)

        # np.array( , dtype=np.float32 )

        start = timeit.default_timer()
        im_med = ndimage.median_filter(im_noise, size)
        stop = timeit.default_timer()
        diff1 = stop - start
        print("end scipy filter", diff1 )


        im_noisecu = im_noise.astype(np.float32)

        # window 2
        #im_noisecu=np.lib.pad(im_noisecu, ((1, 0),(1,0)), 'symmetric')
        # window 4
        im_noisecu=np.lib.pad(im_noisecu, ((loffset, roffset),(loffset, roffset)), 'symmetric')
        im_noisecu = im_noisecu.flatten()

        start = timeit.default_timer()

        # reset the cuda image in the median filter
        filter.setCuImage(im_noisecu)

        # start to run the filter with window size
        filter.runFilter(size)
        results2 = filter.retreive()
        stop = timeit.default_timer()

        results2 = results2.reshape(imsize,imsize)

        diff2 = stop - start
        print("end cuda filter", diff2)
        print("the times gpu over cpu is", diff1/diff2)

        print im_med
        print results2
        print results2-im_med


    #adder.retreive_inplace()
    # results2 = filter.retreive()
    #
    # print("the result is ", results2)
    #print arr
    #npt.assert_array_equal(arr, [2,3,3,3])
    #npt.assert_array_equal(results2, [2,3,3,3])



def test2():
    '''test with randome image and scipy median filter'''
    size = 4
    imsize = 3000


    loffset = size/2
    roffset = (size-1)/2


    im = np.zeros((imsize, imsize))
    im[300:-300, 300:-300] = 1
    im = ndimage.distance_transform_bf(im)
    im_noise = im + 0.2*np.random.randn(*im.shape)
    im_noise = im_noise.astype(np.float32)




    start = timeit.default_timer()
    im_med = ndimage.median_filter(im_noise, (size, size))
    stop = timeit.default_timer()
    diff1 = stop - start
    print("end scipy filter", diff1 )


    filter = tomoCuda.mFilter(imsize, imsize, size)

    im_noisecu = im_noise.astype(np.float32)

    # start to reflect mode first
    start = timeit.default_timer()
    im_noisecu=np.lib.pad(im_noisecu, ((loffset, roffset),(loffset, roffset)), 'symmetric')
    im_noisecu = im_noisecu.flatten()


    # reset the cuda image in the median filter
    filter.setCuImage(im_noisecu)
    filter.runFilter(size)
    results2 = filter.retreive()

    stop = timeit.default_timer()
    diff2 = stop - start
    print("end cuda filter", diff2)
    print("the times gpu over cpu is", diff1/diff2)

    results2 = results2.reshape(imsize,imsize)

    # print results2[6]
    # print im_med[6]
    # print results2[6]-im_med[6]

    print results2-im_med
    # np.savetxt('results2.out', results2, delimiter=',')
    # np.savetxt('im_med.out', im_med, delimiter=',')
    # np.savetxt('diff.out', results2-im_med, delimiter=',')

    plt.figure(figsize=(26, 5))

    plt.subplot(141)
    plt.imshow(im, interpolation='nearest')
    plt.axis('off')
    plt.title('Original image', fontsize=20)
    plt.subplot(142)
    plt.imshow(im_noise, interpolation='nearest', vmin=0, vmax=5)
    plt.axis('off')
    plt.title('Noisy image', fontsize=20)
    plt.subplot(143)
    plt.imshow(im_med, interpolation='nearest', vmin=0, vmax=5)
    plt.axis('off')
    plt.title('Median filter', fontsize=20)
    plt.subplot(144)
    plt.imshow(results2, interpolation='nearest', vmin=0, vmax=5)
    plt.axis('off')
    plt.title('Cuda Median filter', fontsize=20)

    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0, left=0,
                        right=1)

    # plt.show()

def test3():
    '''test with random array, 3d cuda kernel'''
    # prjsize is z, imsize is x, y.

    size = 2 # window size for the filter
    imsize = 3 # image size for the input
    prjsize=2


    # the left and right offset of the image to do the reflect mode
    loffset = size/2
    roffset = (size-1)/2

    #create test 3d array, filter size -1 = loffset+roffset
    combined = np.zeros(shape=(prjsize,imsize+size-1,imsize+size-1), dtype=np.float32)
    results1 = np.zeros(shape=(prjsize,imsize,imsize), dtype=np.float32)

    # create a gpu median filter object
    filter = tomoCuda.mFilter(imsize, imsize, prjsize, size)

    diff1 = 0

    for step in range (5,5+prjsize):

        im_noise = np.arange( 10, imsize*imsize*step+10, step ).reshape(imsize, imsize)
        im_noise = im_noise.astype(np.float32)

        # print im_noise

        start = timeit.default_timer()
        im_med = ndimage.median_filter(im_noise, size)
        stop = timeit.default_timer()
        diff1 += stop - start

        results1[step-5]=im_med


    print("end scipy filter", diff1 )

        # window 2
        #im_noisecu=np.lib.pad(im_noisecu, ((1, 0),(1,0)), 'symmetric')
        # window 4


    for step in range (5,5+prjsize):
        im_noise = np.arange( 10, imsize*imsize*step+10, step ).reshape(imsize, imsize)

        im_noisecu = im_noise.astype(np.float32)
        im_noisecu=np.lib.pad(im_noisecu, ((loffset, roffset),(loffset, roffset)), 'symmetric')
        combined[step-5]=im_noisecu



        # print results2-im_med


    # print combined

    im_noisecu = combined.flatten()
    im_noisecu = im_noisecu.astype(np.float32)

    start = timeit.default_timer()

    # reset the cuda image in the median filter
    filter.setCuImage(im_noisecu)

    # start to run the filter with window size
    filter.run3DFilter(size)
    results2 = filter.retreive()
    stop = timeit.default_timer()

    results2 = results2.reshape(prjsize, imsize,imsize)

    # print results1
    # print results2

    diff2 = stop - start
    print("end cuda filter", diff2)
    print("the times gpu over cpu is", diff1/diff2)

    print not np.any(results1-results2)

def test4():
    '''test with a list of random array'''
    #arr = np.array(np.arange( 10, 49*5+10, 5 ), dtype=np.float32)

    size = 15 # window size for the filter
    imsize = 500 # image size for the input
    prjsize=200

    # the left and right offset of the image to do the reflect mode
    loffset = size/2
    roffset = (size-1)/2

    # create a gpu median filter object
    filter = tomoCuda.mFilter(imsize, imsize, prjsize, size)



    for step in range (5,5+prjsize):

        im_noise = np.arange( 10, imsize*imsize*step+10, step ).reshape(imsize, imsize)
        im_noise = im_noise.astype(np.float32)

        # np.array( , dtype=np.float32 )

        start = timeit.default_timer()
        im_med = ndimage.median_filter(im_noise, size)
        stop = timeit.default_timer()
        diff1 = stop - start
        print("end scipy filter", diff1 )


        im_noisecu = im_noise.astype(np.float32)

        # window 2
        #im_noisecu=np.lib.pad(im_noisecu, ((1, 0),(1,0)), 'symmetric')
        # window 4
        im_noisecu=np.lib.pad(im_noisecu, ((loffset, roffset),(loffset, roffset)), 'symmetric')
        im_noisecu = im_noisecu.flatten()

        start = timeit.default_timer()

        # reset the cuda image in the median filter
        filter.setCuImage(im_noisecu)

        # start to run the filter with window size
        filter.run2DFilter(size)
        results2 = filter.retreive()
        stop = timeit.default_timer()

        results2 = results2.reshape(imsize,imsize)

        diff2 = stop - start
        print("end cuda filter", diff2)
        print("the times gpu over cpu is", diff1/diff2)

        # print im_med
        # print results2
        print results2-im_med

def test5():
    '''test with random array, 2d cuda kernel with a loop inside'''
    # prjsize is z, imsize is x, y.

    size = 15 # window size for the filter
    imsize = 200 # image size for the input
    prjsize=3000


    # the left and right offset of the image to do the reflect mode
    loffset = size/2
    roffset = (size-1)/2

    #create test 3d array, filter size -1 = loffset+roffset
    combined = np.zeros(shape=(prjsize,imsize+size-1,imsize+size-1), dtype=np.float32)
    results1 = np.zeros(shape=(prjsize,imsize,imsize), dtype=np.float32)

    # create a gpu median filter object
    filter = tomoCuda.mFilter(imsize, imsize, prjsize, size)

    diff1 = 0

    for step in range (5,5+prjsize):

        im_noise = np.arange( 10, imsize*imsize*step+10, step ).reshape(imsize, imsize)
        im_noise = im_noise.astype(np.float32)

        # print im_noise

        start = timeit.default_timer()
        im_med = ndimage.median_filter(im_noise, size)
        stop = timeit.default_timer()
        diff1 += stop - start

        results1[step-5]=im_med

    print("end scipy filter", diff1 )

        # window 2
        #im_noisecu=np.lib.pad(im_noisecu, ((1, 0),(1,0)), 'symmetric')
        # window 4


    for step in range (5,5+prjsize):
        im_noise = np.arange( 10, imsize*imsize*step+10, step ).reshape(imsize, imsize)

        im_noise = im_noise.astype(np.float32)
        im_noisecu=np.lib.pad(im_noise, ((loffset, roffset),(loffset, roffset)), 'symmetric')
        combined[step-5]=im_noisecu

        # print im_med

        # print results2-im_med
    # print combined

    im_noisecu = combined.flatten()
    im_noisecu = im_noisecu.astype(np.float32)

    start = timeit.default_timer()

    # reset the cuda image in the median filter
    filter.setCuImage(im_noisecu)

    # start to run the filter with window size
    filter.run2DLoopFilter(size)
    results2 = filter.retreive()

    stop = timeit.default_timer()

    results2 = results2.reshape(prjsize, imsize,imsize)

    diff2 = stop - start
    print("end cuda filter", diff2)
    print("the times gpu over cpu is", diff1/diff2)

    print not np.any(results1-results2)

import tomopy

def testRemoveoutliner1():
    '''test with random array, 3d cuda kernel for removing outiner'''
    # prjsize is z, imsize is x, y.

    size = 15 # window size for the filter
    imsize =10 # image size for the input

    prjsize= 1
    diff = 20


    # the left and right offset of the image to do the reflect mode
    loffset = size/2
    roffset = (size-1)/2

    #create test 3d array, filter size -1 = loffset+roffset
    combinedMed = np.zeros(shape=(prjsize,imsize,imsize), dtype=np.float32)
    combined = np.zeros(shape=(prjsize,imsize+size-1,imsize+size-1), dtype=np.float32)
    results1 = np.zeros(shape=(prjsize,imsize,imsize), dtype=np.float32)

    # create a gpu median filter object
    filter = tomoCuda.mFilter(imsize, imsize, prjsize, size)



    # create combined noise matrix 3D
    for step in range (5,5+prjsize):
        im_noise = np.arange( 10, imsize*imsize*step+10, step ).reshape(imsize, imsize)
        im_noise = im_noise.astype(np.float32)
        combinedMed[step-5]=im_noise
        # print im_noise



        # results1[step-5]=im_med




        # window 2
        #im_noisecu=np.lib.pad(im_noisecu, ((1, 0),(1,0)), 'symmetric')
        # window 4



    # get teh combined test data matrix
    # for step in range (5,5+prjsize):
    #     im_noise = np.arange( 10, imsize*imsize*step+10, step ).reshape(imsize, imsize)
    #     im_noisecu = im_noise.astype(np.float32)
    #     im_noisecu=np.lib.pad(im_noisecu, ((loffset, roffset),(loffset, roffset)), 'symmetric')
    #     combined[step-5]=im_noisecu

    combined = np.lib.pad(combinedMed, ((0,0), (loffset, roffset),(loffset, roffset)), 'symmetric')

    start = timeit.default_timer()
    # im_med = ndimage.median_filter(im_noise, size)
    results1 = tomopy.misc.corr.remove_outlier(combinedMed, diff, size )
    stop = timeit.default_timer()
    diff1 = stop - start

    print("end scipy remove oulier", diff1 )

    # print combined

    im_noisecu = combined.flatten()
    im_noisecu = im_noisecu.astype(np.float32)

    start = timeit.default_timer()

    # reset the cuda image in the median filter
    filter.setCuImage(im_noisecu)
    # start to run the outliner with window size
    filter.run3DRemoveOutliner(size, diff)
    results2 = filter.retreive()
    stop = timeit.default_timer()

    results2 = results2.reshape(prjsize, imsize,imsize)

    # print results1
    # print results2

    diff2 = stop - start
    print("end cuda filter", diff2)
    print("the times gpu over cpu is", diff1/diff2)

    print not np.any(results1-results2)

# test1()
# test2()
# test3()
# test4()
# test5()
testRemoveoutliner1()