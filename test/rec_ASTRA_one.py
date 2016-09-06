import tomopy
# import astra
import time
import numpy as np
import timeit
import tomoCuda

import tomopy.util.dtype as dtype

proj_type='cuda'

def median_filter_GPU(arr, size=3):
    """
    Apply median filter to 3D array along 0 axis with GPU support.
    The winAllow 4 to 7 is for A6000, for Tian X support 3 to 8
    Parameters
    ----------
    arr : ndarray
        Input array.
    size : int, optional
        The size of the filter.

    Returns
    -------
    ndarray
        Median filtered 3D array.
    """

    winAllow = [4,5,6,7,15]

    if size in winAllow:
        prjsize = arr.shape[0]
        loffset = int(size/2)
        roffset = int((size-1)/2)
        imsizex =arr.shape[2] # image size for the input
        imsizey = arr.shape[1]

        filter = tomoCuda.mFilter(imsizex, imsizey, prjsize, size)
        out = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)

        for step in range (prjsize):
            im_noisecu=arr[step].astype(np.float32)
            im_noisecu=np.lib.pad(im_noisecu, ((loffset, roffset),(loffset, roffset)), 'symmetric')
            im_noisecu = im_noisecu.flatten()

            filter.setCuImage(im_noisecu)
            filter.run2DFilter(size)
            results = filter.retreive()
            results=results.reshape(imsizey,imsizex)
            out[step]=results
    else:
        out = tomopy.median_filter(arr, size)

    return out

def remove_outlier_GPU(arr, dif, size=3):
    """
    Remove high intensity bright spots from a 3D array along specified
    dimension.

    Parameters
    ----------
    arr : ndarray
        Input array.
    dif : float
        Expected difference value between outlier value and
        the median value of the array.
    size : int
        Size of the median filter.
    axis : int, optional
        Axis along which median filtering is performed.
    ncore : int, optional
        Number of cores that will be assigned to jobs.
    out : ndarray, optional
        Output array for result.  If same as arr, process will be done in-place.


    Returns
    -------
    ndarray
       Corrected array.
    """
    arr = dtype.as_float32(arr)
    dif = np.float32(dif)

    winAllow = [4,5,6,7,15]

    if size in winAllow:
        prjsize = arr.shape[0]
        loffset = int(size/2)
        roffset = int((size-1)/2)
        imsizex =arr.shape[2] # image size for the input
        imsizey = arr.shape[1]

        filter = tomoCuda.mFilter(imsizex, imsizey, prjsize, size)
        out = np.zeros(shape=(prjsize,imsizey,imsizex), dtype=np.float32)

        for step in range (prjsize):
            im_noisecu=arr[step].astype(np.float32)
            im_noisecu=np.lib.pad(im_noisecu, ((loffset, roffset),(loffset, roffset)), 'symmetric')
            im_noisecu = im_noisecu.flatten()

            filter.setCuImage(im_noisecu)
            filter.run2DRemoveOutliner(size, dif)
            results = filter.retreive()
            results=results.reshape(imsizey,imsizex)
            out[step]=results
    else:
        print("using cpu remove outlier")
        out = tomopy.remove_outlier(arr, dif, size)

    return out


def rec_test(file_name, sino_start, sino_end, astra_method, extra_options, num_iter=1):

    print '\n#### Processing '+ file_name
    print "Test reconstruction of slice [%d]" % sino_start
    # Read HDF5 file.
    prj, flat, dark = tomopy.io.exchange.read_aps_32id(file_name, sino=(sino_start, sino_end))

    # Manage the missing angles:
    theta  = tomopy.angles(prj.shape[0])
#    prj = np.concatenate((prj[0:miss_angles[0],:,:], prj[miss_angles[1]+1:-1,:,:]), axis=0)
#    theta = np.concatenate((theta[0:miss_angles[0]], theta[miss_angles[1]+1:-1]))
    prj = prj[miss_angles[0]:miss_angles[1],:,:]
    theta = theta[miss_angles[0]:miss_angles[1]]

    # normalize the prj
    prj = tomopy.normalize(prj, flat, dark)
    
    # remove ring artefacts
#    prj = tomopy.remove_stripe_fw(prj)
    
    # remove ring artefacts
    prj = tomopy.remove_stripe_sf(10)
    
    # Median filter:
    if medfilt_size:
        prj = tomopy.median_filter(prj,size=medfilt_size)

    if level>0:
        prj = tomopy.downsample(prj, level=level)

    # reconstruct 
    # rec = tomopy.recon(prj, theta, center=best_center/pow(2,level), algorithm=tomopy.astra, options={'proj_type':proj_type,'method':astra_method,'extra_options':extra_options,'num_iter':num_iter}, emission=False)
    
    # Write data as stack of TIFs.
    tomopy.io.writer.write_tiff_stack(rec, fname=output_name)

    print "Slice saved as [%s_00000.tiff]" % output_name
    
    
def rec_full(file_name, sino_start, sino_end, astra_method, extra_options, num_iter=1):

    start_time = time.time()
    print '\n#### Processing '+ file_name

    chunks = 10 # number of data chunks for the reconstruction

    nSino_per_chunk = (sino_end - sino_start)/chunks
    print "Reconstructing [%d] slices from slice [%d] to [%d] in [%d] chunks of [%d] slices each" % ((sino_end - sino_start), sino_start, sino_end, chunks, nSino_per_chunk)
    strt = 0
    for iChunk in range(0,chunks):
        diff = 0
        print '\n  -- chunk # %i' % (iChunk+1)
        sino_chunk_start = sino_start + nSino_per_chunk*iChunk 
        sino_chunk_end = sino_start + nSino_per_chunk*(iChunk+1)
        print '\n  --------> [%i, %i]' % (sino_chunk_start, sino_chunk_end)
        
        if sino_chunk_end > sino_end: 
            break

        start = timeit.default_timer()
        # Read HDF5 file.
        prj, flat, dark = tomopy.io.exchange.read_aps_32id(file_name, sino=(sino_chunk_start, sino_chunk_end))
        stop = timeit.default_timer()
        print prj.shape
        diff += stop - start
        print("end read data", stop-start)


        start = timeit.default_timer()
        # Manage the missing angles:
        theta  = tomopy.angles(prj.shape[0])
        stop = timeit.default_timer()
        diff += stop - start
        print("end angles", stop-start)

#        theta = np.concatenate((theta[0:miss_angles[0]], theta[miss_angles[1]+1:-1]))
#        theta = np.concatenate((theta[0:miss_angles[0]], theta[miss_angles[1]+1:-1]))
        prj = prj[miss_angles[0]:miss_angles[1],:,:]
        theta = theta[miss_angles[0]:miss_angles[1]]

        start = timeit.default_timer()
        # normalize the prj
        prj = tomopy.normalize(prj, flat, dark)
        stop = timeit.default_timer()
        diff += stop - start
        print("end normalize", stop-start)


        print '\n start remove stripe'
        # remove ring artefacts
        start = timeit.default_timer()
        print prj.shape

        prj = tomopy.remove_stripe_ti(prj, 2)
        stop = timeit.default_timer()
        diff += stop - start
        print("end remove stripe", stop-start)


        print '\n start median_filter'
        print prj.shape
        print medfilt_size



        start = timeit.default_timer()
        resultcombine= median_filter_GPU(prj, medfilt_size)
        stop = timeit.default_timer()
        mdiff2= stop - start
        print("end gpu median filter", stop-start)




        start = timeit.default_timer()
        # Median filter:
        if medfilt_size:
            prj = tomopy.median_filter(prj,size=medfilt_size)
        stop = timeit.default_timer()
        mdiff1=stop-start
        alldiff=mdiff1
        diff += mdiff1
        print("end median filter", stop-start)

        print("the times gpu over cpu is", mdiff1/mdiff2)
        print not np.any(prj-resultcombine)



        print '\n start outlier remove'
        odiff = 20
        start = timeit.default_timer()
        resultcombine = remove_outlier_GPU(prj, odiff, medfilt_size)
        stop = timeit.default_timer()
        rdiff2= stop - start
        print("end gpu outlier remove", stop-start)


        start = timeit.default_timer()
        # Median filter:
        if medfilt_size:
            prj = tomopy.misc.corr.remove_outlier(prj, odiff, medfilt_size)
            # prj = tomopy.median_filter(prj,size=medfilt_size)

        stop = timeit.default_timer()
        rdiff1=stop-start
        alldiff+=rdiff1
        diff+=rdiff1
        print("end outlier removal", stop-start)

        print("the times gpu over cpu is", rdiff1/rdiff2)
        print not np.any(prj-resultcombine)



        start = timeit.default_timer()
        print '\n start downsample'
        if level>0:
            prj = tomopy.downsample(prj, level=level)
            prj = tomopy.downsample(prj, level=level, axis=1)
        stop = timeit.default_timer()
        diff += stop - start
        print("end downsample", stop-start)

        # reconstruct 
        print '\n start reconstruction'

        start = timeit.default_timer()
#        astra_method='FBP_CUDA'
#        proj_type='cuda'
        # extra_options='MinConstraint':0
        # num_iter=100
        # rec = tomopy.recon(prj, theta, center=best_center/pow(2,level), algorithm=tomopy.astra, options={'proj_type':proj_type,'method':astra_method,'extra_options':extra_options,'num_iter':num_iter}, emission=False)
        print prj.shape, theta.shape, best_center/pow(2,level)
        rec = tomopy.recon(prj,theta,center=best_center/pow(2,level), algorithm='gridrec')

        stop = timeit.default_timer()
        # diff += stop - start
        print("end reconstruction", stop-start)
        # print "the percentage is %f, %f" % (mdiff1/diff, rdiff1/diff)
        # print("the total diff is", diff)
        print output_name

        # Write data as stack of TIFs.
        tomopy.io.writer.write_tiff_stack(rec, fname=output_name, start=strt)
        strt += prj.shape[1]

    print("%i minutes" % ((time.time() - start_time)/60))



astra_method='SIRT-FBP'
extra_options = {'filter_dir':'./filters'}
num_iter = 100

# if astra_method=='SIRT-FBP':
#     import sirtfbp
#     astra.plugin.register(sirtfbp.plugin)
#     extra_options = {'filter_dir':'./filters'}
#     num_iter = 100

##################################### Inputs ##########################################################
file_name = '/data2/XiaoData/reader1/proj_4.hdf' # best center = 1268
output_name = '/data2/XiaoData/reader1/Test/test_Astra_recon_'

# file_name = './proj_4.hdf'
# output_name = './Test/test_Astra_recon_'

#file_name = '/local/dataraid/2015_11/Debbie/F11_60nmZP_721proj_8000eV_5X_3s_61.h5' # best center = 1198
#output_name = '/local/dataraid/2015_11/Debbie/F11_60nmZP_721proj_8000eV_5X_3s_61_binned2_Astra_recon/F11_60nmZP_721proj_8000eV_5X_3s_61_binned2_Astra_recon_'
#file_name = '/local/dataraid/2015_11/Debbie/F14_60nmZP_721proj_8000eV_5X_3s_62.h5' # best center = 1266
#output_name = '/local/dataraid/2015_11/Debbie/F14_60nmZP_721proj_8000eV_5X_3s_62_no_bin_Astra_recon/F14_60nmZP_721proj_8000eV_5X_3s_62_no_bin_Astra_recon_'
#file_name = '/local/dataraid/2015_11/Debbie/B14_60nmZP_721proj_8000eV_5X_3s_65_.h5' # best center = 1298
#output_name = '/local/dataraid/2015_11/Debbie/B14_60nmZP_721proj_8000eV_5X_3s_65__binned2_Astra_recon/B14_60nmZP_721proj_8000eV_5X_3s_65__binned2_Astra_recon_'
#file_name = '/local/dataraid/2015_11/Debbie/F31_60nmZP_721proj_8000eV_5X_5s_72.h5' # best center = 1300
#output_name = '/local/dataraid/2015_11/Debbie/F31_60nmZP_721proj_8000eV_5X_5s_72_binned2_Astra_recon/F31_60nmZP_721proj_8000eV_5X_5s_72_binned2_Astra_recon_'
#output_name = '/local/dataraid/2015_11/Debbie/test/test_'

reconstruction_test = False
best_center = 1268; sino_start = 0; sino_end = 2048; miss_angles = [0,721]; level = 1; medfilt_size = 5
if reconstruction_test: rec_test(file_name, sino_start, sino_end, astra_method, extra_options, num_iter)
else: rec_full(file_name, sino_start, sino_end, astra_method, extra_options, num_iter)
