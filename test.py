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
    size = 4
    imsize = 7


    loffset = size/2
    roffset = (size-1)/2

    im_noise = np.arange( 10, 49*5+10, 5 ).reshape(7, 7)
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
    print im_noisecu


    im_noisecu = im_noisecu.flatten()
    filter = tomoCuda.mFilter(im_noisecu, imsize, imsize, size)
    start = timeit.default_timer()
    filter.runFilter(size)
    stop = timeit.default_timer()
    diff2 = stop - start
    print("end cuda filter", diff2)
    print("the times gpu over cpu is", diff1/diff2)
    results2 = filter.retreive()
    results2 = results2.reshape(imsize,imsize)

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

test1()

def test2():
    '''test with randome image and scipy median filter'''
    imsize = 200
    im = np.zeros((imsize, imsize))
    im[300:-300, 300:-300] = 1
    im = ndimage.distance_transform_bf(im)
    im_noise = im + 0.2*np.random.randn(*im.shape)
    im_noise = im_noise.astype(np.float32)


    size = 4

    start = timeit.default_timer()
    im_med = ndimage.median_filter(im_noise, (size, size))
    stop = timeit.default_timer()
    diff1 = stop - start
    print("end scipy filter", diff1 )



    im_noisecu = im_noise.astype(np.float32)
    im_noisecu = im_noisecu.flatten()
    filter = tomoCuda.mFilter(im_noisecu, imsize, imsize)
    start = timeit.default_timer()
    filter.runFilter(size)
    stop = timeit.default_timer()
    diff2 = stop - start
    print("end cuda filter", diff2)
    print("the times gpu over cpu is", diff1/diff2)
    results2 = filter.retreive()
    results2 = results2.reshape(imsize,imsize)

    print results2[6]
    print im_med[6]
    print results2[6]-im_med[6]

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

# test2()
