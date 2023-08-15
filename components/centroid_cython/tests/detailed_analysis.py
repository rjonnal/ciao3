import centroid
from matplotlib import pyplot as plt
import glob,sys,os
import numpy as np
from time import time
import centroid

image_width = 33
sb_x_vec = np.array([image_width/2.0-0.5])
sb_y_vec = np.array([image_width/2.0-0.5])
valid_vec = np.ones(sb_x_vec.shape,dtype=np.int16)
sb_half_width = image_width//2-1
centroiding_half_width = sb_half_width-2
xout = np.zeros(sb_x_vec.shape)
yout = np.zeros(sb_y_vec.shape)
max_intensity = np.zeros(sb_x_vec.shape)
n_spots = len(sb_x_vec)
A = 2000 # express in camera ADU
DC = 100 # express in camera ADU
x = np.arange(image_width)
x = x-np.mean(x)
XX,YY = np.meshgrid(x,x)
bit_depth=12

def make_spot(dx=0.0,dy=0.0,sigma=3.0,dc=DC,amplitude=A,noise_gain=1.0):
    xx = XX-dx
    yy = YY-dy
    light = np.exp(-(xx**2+yy**2)/(2*sigma**2))*amplitude
    shot_noise = np.random.randn(light.shape[0],light.shape[1])*np.sqrt(light)
    read_noise = np.random.randn(light.shape[0],light.shape[1])
    # assume discretization @ read noise level, so read noise should have STD of 1
    total_noise = noise_gain*(shot_noise+read_noise)
    im = light+dc+total_noise
    im = np.clip(np.round(im).astype(np.int16),0,2**bit_depth)
    return im


def center_of_mass(spots_image,refx,refy,sb_half_width,do_plot=False):
    x1 = int(round(refx-sb_half_width))
    x2 = int(round(refx+sb_half_width))
    y1 = int(round(refy-sb_half_width))
    y2 = int(round(refy+sb_half_width))

    subim = spots_image[y1:y2+1,x1:x2+1]
    v = np.arange(-sb_half_width,sb_half_width+1)
    xx,yy = np.meshgrid(refx+v,refy+v)
    xc = np.sum(xx*subim)/np.sum(subim)
    yc = np.sum(yy*subim)/np.sum(subim)

    if do_plot:
        plt.imshow(spots_image,cmap='gray')
        plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'r-')
        plt.plot(xc,yc,'gx')
    
    return xc,yc

spot = make_spot(dx=0.5,dy=0.5,noise_gain=0.0,sigma=0.5,dc=0,amplitude=1000.0)
xcom,ycom = center_of_mass(spot,sb_x_vec,sb_y_vec,sb_half_width,do_plot=False)


def fast_centroids(spots_image,sb_x_vec,sb_y_vec,sb_half_width,centroiding_half_width,verbose=False):
    n_spots = len(sb_x_vec)
    x_out = np.zeros(n_spots)
    y_out = np.zeros(n_spots)
    for spot_index in range(n_spots):
        current_max = -2**16+1
        
        x1 = int(round(sb_x_vec[spot_index]-sb_half_width))
        x2 = int(round(sb_x_vec[spot_index]+sb_half_width))
        y1 = int(round(sb_y_vec[spot_index]-sb_half_width))
        y2 = int(round(sb_y_vec[spot_index]+sb_half_width))

        if verbose:
            print 'python A',spot_index,x1,x2,y1,y2
        
        for y in range(y1,y2+1):
            for x in range(x1,x2+1):
                pixel = spots_image[y,x]
                if pixel>current_max:
                    current_max = pixel
                    max_y = y
                    max_x = x
                    
        x1 = int(round(max_x-centroiding_half_width))
        x2 = int(round(max_x+centroiding_half_width))
        y1 = int(round(max_y-centroiding_half_width))
        y2 = int(round(max_y+centroiding_half_width))

        if verbose:
            print 'python B',spot_index,x1,x2,y1,y2
        
        xnum = 0.0
        ynum = 0.0
        denom = 0.0

        for y in range(y1,y2+1):
            for x in range(x1,x2+1):
                pixel = spots_image[y,x]
                xnum = xnum + pixel*x
                ynum = ynum + pixel*y
                denom = denom + pixel

        x_out[spot_index] = xnum/denom
        y_out[spot_index] = ynum/denom

    return x_out,y_out


centroid.fast_centroids(spot,sb_x_vec,sb_y_vec,sb_half_width,centroiding_half_width,xout,yout,max_intensity,valid_vec,0,1)

pxout,pyout = fast_centroids(spot,sb_x_vec,sb_y_vec,sb_half_width,centroiding_half_width,verbose=False)

print xcom,ycom
print xout,yout
print pxout,pyout
sys.exit()


centroid.fast_centroids(spots_image,sb_x_vec,sb_y_vec,sb_half_width,centroiding_half_width,xout,yout,max_intensity,valid_vec,0,1)
pxout,pyout = fast_centroids(spots_image,sb_x_vec,sb_y_vec,sb_half_width,centroiding_half_width)

python_cython_err = (xout-pxout).tolist()+(yout-pyout).tolist()
if any(python_cython_err):
    sys.exit('Error between Cython centroiding and Python centroiding. Please fix.')
else:
    print 'Cython centroid centers of mass match pure Python calculations.'
    
cython_ground_truth_err = (xout-x_spot_location).tolist()+(yout-y_spot_location).tolist()
if any(cython_ground_truth_err):
    sys.exit('Error between Cython centroiding and ground truth. Please fix.')
else:
    print 'Cython centroid centers of mass match ground truth.'

N = 1000
t0 = time()
for k in range(N):
    centroid.fast_centroids(spots_image,sb_x_vec,sb_y_vec,sb_half_width,centroiding_half_width,xout,yout,max_intensity,valid_vec,0,1)

t_total = time()-t0
t_iteration = t_total/float(N)
fps = 1.0/t_iteration

print '%d spots, %d iterations, total time %0.1f, iteration time %0.1e, fps %0.1f'%(n_spots,N,t_total,t_iteration,fps)
