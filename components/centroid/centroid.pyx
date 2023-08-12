import numpy as np
cimport numpy as np
from matplotlib import pyplot as plt
import cython
from cython.parallel import prange
from libc.stdio cimport printf
import sys,os
from sys import exit

# Importing round is failing in windows for some reason; it may have
# something to do with the MSVS c compiler and a 32-bit 64-bit clash.
# If we want to implement parallelism at some point, we need to figure
# this out, because the python round function cannot be used in a
# 'with nogil:' context (since it's a python object), but we need some
# way to round values and it's risky to write our own c round function
# because it might be inefficient compared to clang's version.
# from libc.math cimport round as c_round

# Changes required to compile parallel version (Linux only):
# 1. uncomment the "from libc.math cimport round as c_round" line above
# 2. change all calls to round below to c_round
# 3. add back the 'with nogil:' context, just above the prange
# 4. change range to prange in the outermost loop (the one that iterates
#    over the spots

ctypedef np.uint16_t uint16_t

# Function compute_centroids:
# this function takes the following arguments:
# 1. spots_image (int array): the spots image
# 2. sb_x_vec (float array): the x coordinates of search box centers
# 3. sb_y_vec (float array): the y coordinates of search box centers
# 4. sb_bg_vec (float array): the background values for each search box
# 5. sb_half_width (integer): if the width of the search box (inclusive),
#    is odd integer N, this is (N-1)//2
# 6. iterations (integer): the number of centroid iterations, in
#    which the search boxes are recentered about the
#    previous center of mass measurement
# 7. iteration_step_px (integer): the number of pixels by which
#    to reduce the search box half-width on each iteration
# 8. x_out (float array): an array in which to store the
#    resulting x coordinates of centers of mass
# 9. y_out (float array): an array in which to store the
#    resulting y coordinates of centers of mass
# 10. mean_intensity (float array): an array in which to store each
#    search box's mean intensity
# 11. maximum_intensity (float array): (ditto)
# 12. minimum_intensity (float array): (ditto)
# 13. num_threads (integer): number of threads to use; currently not
#     implemented because it doesn't improve speed. Use of prange may
#     not be optimal way to thread this.

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_centroids(np.ndarray[np.int16_t,ndim=2] spots_image,
                        np.ndarray[np.float_t,ndim=1] sb_x_vec,
                        np.ndarray[np.float_t,ndim=1] sb_y_vec,
                        np.ndarray[np.float_t,ndim=1] sb_bg_vec,
                        sb_half_width_p,
                        iterations_p,
                        iteration_step_px_p,
                        np.ndarray[np.float_t,ndim=1] x_out,
                        np.ndarray[np.float_t,ndim=1] y_out,
                        np.ndarray[np.float_t,ndim=1] mean_intensity,
                        np.ndarray[np.float_t,ndim=1] maximum_intensity,
                        np.ndarray[np.float_t,ndim=1] minimum_intensity,
                        num_threads_p = 1):


    cdef np.int_t n_spots = len(sb_x_vec)
    cdef np.int_t num_threads = int(num_threads_p)
    cdef np.int_t iterations = int(iterations_p)
    cdef np.int_t iteration_step_px = int(iteration_step_px_p)
    cdef np.int_t sb_half_width = int(sb_half_width_p)

    cdef np.float_t intensity
    cdef np.float_t background
    cdef np.float_t xprod
    cdef np.float_t yprod
    cdef np.int_t x
    cdef np.int_t x1
    cdef np.int_t x2
    cdef np.int_t y
    cdef np.int_t y1
    cdef np.int_t y2
    cdef np.int_t sy
    cdef np.int_t sx
    cdef np.int_t half_width
    cdef np.float_t imax
    cdef np.float_t imin
    cdef np.float_t pixel
    cdef np.float_t counter
    cdef np.int_t k_iteration
    cdef np.int_t k_spot

    sy = spots_image.shape[0]
    sx = spots_image.shape[1]

    # Populate x_out,y_out with the sb centers, for starters; this allows
    # us to use the out arrays as places to both read the current sb center
    # (in the case of multiple iterations, where we want to recenter the
    # sb on each iteration
    # This serves an additional purpose--preventing sb_x_vec or sb_y_vec from
    # being altered; this is critical, as these represent the loop's search box
    # centers and reference coordinates.
    x_out[:] = sb_x_vec[:]
    y_out[:] = sb_y_vec[:]

    # first, we iterate through the number of iterations
    for k_iteration in range(0,iterations):

        for k_spot in range(n_spots):

            imin = 2**15
            imax = -2**15
            xprod = 0.0
            yprod = 0.0
            intensity = 0.0
            counter = 0.0

            x1 = int(round(x_out[k_spot]))-sb_half_width+k_iteration*iteration_step_px
            x2 = int(round(x_out[k_spot]))+sb_half_width-k_iteration*iteration_step_px
            y1 = int(round(y_out[k_spot]))-sb_half_width+k_iteration*iteration_step_px
            y2 = int(round(y_out[k_spot]))+sb_half_width-k_iteration*iteration_step_px

            if x1<0 or x2>sx-1 or y1<0 or y2>sy-1:
                printf("Search box x=(%ld,%ld),y=(%ld,%ld) extends beyond image edge. Possibly search box width too large.\n",x1,x2,y1,y2)
                #exit(0)

            if x1>=x2 or y1>=y2:
                printf("Search box x=(%ld,%ld),y=(%ld,%ld) too small. Possibly search box width too large, number of iterations too high, or iteration step size too high.\n",x1,x2,y1,y2)
                #exit(0)

            background = sb_bg_vec[k_spot]
            #printf("background=%0.1f\n",background)
                    
            for x in range(x1,x2+1):
                for y in range(y1,y2+1):

                    # not sure if it's better to cast with python's float()
                    # or c's <float>:
                    pixel = float(spots_image[y,x])-background
                    
                    if pixel<0.0:
                        pixel = 0.0
                    xprod = xprod + pixel*x
                    yprod = yprod + pixel*y
                    intensity = intensity + pixel
                    if pixel<imin:
                        imin = pixel
                    elif pixel>imax:
                        imax = pixel
                    counter = counter + 1.0

            if intensity==0 or xprod==0 or yprod==0:
                printf("Warning: search box intensity low; skipping.\n")
                continue

            mean_intensity[k_spot] = intensity/counter
            maximum_intensity[k_spot] = imax
            minimum_intensity[k_spot] = imin
            x_out[k_spot] = xprod/intensity
            y_out[k_spot] = yprod/intensity

            
# Function estimate_backgrounds:
# this function takes the following arguments:
# 1. spots_image (int array): the spots image
# 2. sb_x_vec (float array): the x coordinates of search box centers
# 3. sb_y_vec (float array): the y coordinates of search box centers
# 4. sb_bg_vec (float array): array for writing output
# 5. sb_half_width (integer): if the width of the search box (inclusive)
#    is odd integer N, this is (N-1)//2 

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef estimate_backgrounds(np.ndarray[np.int16_t,ndim=2] spots_image,
                           np.ndarray[np.float_t,ndim=1] sb_x_vec,
                           np.ndarray[np.float_t,ndim=1] sb_y_vec,
                           np.ndarray[np.float_t,ndim=1] sb_bg_vec,
                           sb_half_width_p):

    cdef np.int_t n_spots = len(sb_x_vec)
    cdef np.int_t sb_half_width = int(sb_half_width_p)

    cdef np.float_t intensity
    cdef np.int_t x
    cdef np.int_t x1
    cdef np.int_t x2
    cdef np.int_t y
    cdef np.int_t y1
    cdef np.int_t y2
    cdef np.int_t sy
    cdef np.int_t sx
    cdef np.float_t counter
    cdef np.int_t k_spot

    sy = spots_image.shape[0]
    sx = spots_image.shape[1]

    # first, we iterate through the number of iterations
    for k_spot in range(n_spots):

        intensity = 0.0
        counter = 0.0

        x1 = int(round(sb_x_vec[k_spot]))-sb_half_width
        x2 = int(round(sb_x_vec[k_spot]))+sb_half_width
        y1 = int(round(sb_y_vec[k_spot]))-sb_half_width
        y2 = int(round(sb_y_vec[k_spot]))+sb_half_width

        if x1<0 or x2>sx-1 or y1<0 or y2>sy-1:
            printf("Search box x=(%ld,%ld),y=(%ld,%ld) extends beyond image edge. Possibly search box width too large.\n",x1,x2,y1,y2)
            exit(0)

        if x1>=x2 or y1>=y2:
            printf("Search box x=(%ld,%ld),y=(%ld,%ld) too small. Possibly search box width too large, number of iterations too high, or iteration step size too high.\n",x1,x2,y1,y2)
            exit(0)

        for x in range(x1,x2+1):
            intensity = intensity + float(spots_image[y1,x]) + float(spots_image[y2,x])
            counter = counter + 2.0
        for y in range(y1,y2+1):
            intensity = intensity + float(spots_image[y,x1]) + float(spots_image[y,x2])
            counter = counter + 2.0

        sb_bg_vec[k_spot] = intensity/counter



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fast_centroids(np.ndarray[np.int16_t,ndim=2] spots_image,
                     np.ndarray[np.float_t,ndim=1] sb_x_vec,
                     np.ndarray[np.float_t,ndim=1] sb_y_vec,
                     sb_half_width_p,
                     centroiding_half_width_p,
                     np.ndarray[np.float_t,ndim=1] x_out,
                     np.ndarray[np.float_t,ndim=1] y_out,
                     np.ndarray[np.float_t,ndim=1] sb_max_vec,
                     np.ndarray[np.int16_t,ndim=1] valid_vec,
                     verbose_p = 0,
                     num_threads_p = 1):

    """Function fast_centroids:
    this function takes the following arguments:
    1. spots_image (int array): the spots image
    2. sb_x_vec (float array): the x coordinates of search box centers
    3. sb_y_vec (float array): the y coordinates of search box centers
    4. sb_half_width_p (integer): if the width of the search box (inclusive)
       is odd integer N, this is (N-1)//2
    5. centroiding_half_width_p: this is the width of the region over which
       to compute the center of mass; should be at least twice the expected
       subaperture diffraction-limited spot size (to handle elongation of spots
       due to axial distribution of retinal reflectors)
    6. x_out (float array): an array in which to store the
       resulting x coordinates of centers of mass
    7. y_out (float array): an array in which to store the
       resulting y coordinates of centers of mass
    8. maximum_intensity (float array): (ditto)
    9. valid_vec (int array): a numpy array with type int16 into which
       the validity of a centroid is recorded; on output 0 means the
       measurement was invalid, and 1 means valid
    10. verbose (integer): determines verbosity of debugging messages
    11. num_threads (integer): number of threads to use; currently not
        implemented."""

    cdef int sy
    cdef int sx
    
    sy = spots_image.shape[0]
    sx = spots_image.shape[1]
    
    # expose memory location of all numpy arrays using typed memoryviews
    # this improves speed because it removes numpy overhead, and more
    # importantly it's required for parallelism because we need to put
    # prange in a 'with nogil:' context
    cdef short [:,:] spots_image_view = spots_image
    cdef double [:] sb_x_vec_view = sb_x_vec
    cdef double [:] sb_y_vec_view = sb_y_vec
    cdef double [:] x_out_view = x_out
    cdef double [:] y_out_view = y_out
    cdef double [:] sb_max_vec_view = sb_max_vec
    cdef short [:] valid_vec_view = valid_vec
    
    cdef int n_spots = len(sb_x_vec)
    cdef int spot_index
    cdef int sb_half_width_c = sb_half_width_p
    cdef int centroiding_half_width_c = centroiding_half_width_p
    cdef int verbose_c = verbose_p
    
    # declare some variables for the inner loops; don't forget to initialize
    # on every iteration
    cdef int y
    cdef int x
    cdef int max_y
    cdef int max_x
    cdef int current_max
    cdef short pixel

    cdef int x1
    cdef int x2
    cdef int y1
    cdef int y2

    cdef double xnum
    cdef double ynum
    cdef double denom

    # first we have to iterate through the spots; do this without the gil, for
    # future parallelization
    #with nogil:
    # changed this for the time being to prevent linking errors in windows due
    # to 32-bit libc.math.round being incompatile with 64 bit python. Replacing
    # 'with nogil:' with 'if True:' to avoid having to unindent everything after.
    if True:
        # For parallel version change range to prange
        for spot_index in range(n_spots):
            current_max = -2**16+1

            # This is how we have to do it if we want nogil and parallelism:
            #x1 = <int>c_round(sb_x_vec_view[spot_index]-sb_half_width_c)
            #x2 = <int>c_round(sb_x_vec_view[spot_index]+sb_half_width_c)
            #y1 = <int>c_round(sb_y_vec_view[spot_index]-sb_half_width_c)
            #y2 = <int>c_round(sb_y_vec_view[spot_index]+sb_half_width_c)

            # This is how we do it (with Python round) to avoid the Windows
            # problem but this prevents parallelizing:
            x1 = <int>round(sb_x_vec_view[spot_index]-sb_half_width_c)
            x2 = <int>round(sb_x_vec_view[spot_index]+sb_half_width_c)
            y1 = <int>round(sb_y_vec_view[spot_index]-sb_half_width_c)
            y2 = <int>round(sb_y_vec_view[spot_index]+sb_half_width_c)

            if verbose_c>0:
                printf('cython A %d,%d,%d,%d,%d\n',spot_index,x1,x2,y1,y2)
            
            if (x1<0 or x2<0 or x1>sx-1 or x2>sx-1 or
                y1<0 or y2<0 or y1>sy-1 or y2>sy-1):
                printf('centroid.fast_centroids: search box coordinates x=[%d,%d], y=[%d,%d] not valid for spots image with size %dx%d. Check search_box_half_width.\n',x1,x2,y1,y2,sx,sy)
                valid_vec_view[spot_index] = 0
                x_out[spot_index] = -1
                y_out[spot_index] = -1
                break
                
            for y in range(y1,y2+1):
                for x in range(x1,x2+1):
                    pixel = spots_image_view[y,x]
                    if pixel>current_max:
                        current_max = pixel
                        max_y = y
                        max_x = x

            sb_max_vec[spot_index] = current_max

            # See note abouve about c_round vs round
            # x1 = <int>c_round(max_x-centroiding_half_width_c)
            # x2 = <int>c_round(max_x+centroiding_half_width_c)
            # y1 = <int>c_round(max_y-centroiding_half_width_c)
            # y2 = <int>c_round(max_y+centroiding_half_width_c)

            # Using Python round instead of libc.math.round for
            # windows compatibility
            x1 = <int>round(max_x-centroiding_half_width_c)
            x2 = <int>round(max_x+centroiding_half_width_c)
            y1 = <int>round(max_y-centroiding_half_width_c)
            y2 = <int>round(max_y+centroiding_half_width_c)

            if verbose_c>0:
                printf('cython B %d,%d,%d,%d,%d\n',spot_index,x1,x2,y1,y2)
            
            if (x1<0 or x2<0 or x1>sx-1 or x2>sx-1 or
                y1<0 or y2<0 or y1>sy-1 or y2>sy-1):
                printf('centroid.fast_centroids: centroiding coordinates x=[%d,%d], y=[%d,%d] not valid for spots image with size %dx%d. Check centroiding_half_width.\n',x1,x2,y1,y2,sx,sy)
                valid_vec_view[spot_index] = 0
                x_out[spot_index] = -1
                y_out[spot_index] = -1
            else:
                
                xnum = 0.0
                ynum = 0.0
                denom = 0.0

                #printf('box %d x=[%d,%d] y=[%d,%d]\n',spot_index,x1,x2,y1,y2)
                for y in range(y1,y2+1):
                    for x in range(x1,x2+1):
                        pixel = spots_image_view[y,x]
                        xnum = xnum + <double>(pixel*x)
                        ynum = ynum + <double>(pixel*y)
                        denom = denom + <double>pixel
                        #printf('\t%d,%d,%0.2f,%0.2f,%0.2f\n',x,y,xnum,ynum,denom)

                if denom>0:
                    #printf('%f,%f,%f,\n',xnum,ynum,denom)
                    x_out[spot_index] = xnum/denom
                    y_out[spot_index] = ynum/denom
                    #printf('%f,%f\n',x_out[spot_index],y_out[spot_index])
                    #printf('\n')
                    valid_vec[spot_index] = 1
                else:
                    printf('centroid.fast_centroids: centroiding coordinates x=[%d,%d], y=[%d,%d] produce search box with zero intensity. Check image.\n',x1,x2,y1,y2,sx,sy)
                    valid_vec[spot_index] = 0
                    x_out[spot_index] = -1
                    y_out[spot_index] = -1
        
            
    return 1
