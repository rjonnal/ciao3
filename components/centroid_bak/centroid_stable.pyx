import numpy as np
cimport numpy as np
from matplotlib import pyplot as plt
import cython
from cython.parallel import prange
from libc.stdio cimport printf
ctypedef np.uint16_t uint16_t


# Function compute_centroids:
# this function takes the following arguments:
# 1. spots_image (int array): the spots image
# 2. sb_x_vec (float array): the x coordinates of search box centers
# 3. sb_y_vec (float array): the y coordinates of search box centers
# 4. sb_bg_vec (float array): the background values for each search box
# 5. sb_width (integer): the width of the search box (inclusive),
#    which should be an odd integer
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
# 5. sb_width (integer): the width of the search box (inclusive),
#    which should be an odd integer

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
