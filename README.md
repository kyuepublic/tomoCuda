# Tomocuda

CUDA version of median_filter and outlier_removal for tomopy

# Install instruction:

1 download the source " git clone https://github.com/kyuepublic/tomocuda.git"

2 go to the source code directory and issue the command "python setup install"

# Prerequisite

 CUDA
 
 Cython
 
# Usage:

import tomocuda

tomocuda.median_filter_cuda(arr, size)

tomocuda.remove_outlier_cuda(arr, dif, size)
