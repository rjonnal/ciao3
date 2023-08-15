from matplotlib import pyplot as plt
import numpy as np
import sys,os

sensor_size = 100
x = np.arange(sensor_size)
ref_x = x.mean()

def create_spot(sensor_size=15,x0=None,
                sigma=None,amplitude=1000.0,
                dc=100.0,round_output=False,noise_gain=1.0,do_plot=0.0):
    if x0 is None:
        x0 = (sensor_size-1)/2.0
    if sigma is None:
        sigma = sensor_size/100.0
    x = np.arange(sensor_size)
    light = amplitude*np.exp(-(x-x0)**2/(2*sigma**2))
    shot_noise = np.sqrt(light)*np.random.randn(len(light))*noise_gain
    read_noise = np.random.randn(len(light))*noise_gain
    noise = shot_noise+read_noise
    signal = light+noise+dc
    if round_output:
        signal = np.round(signal)
    if do_plot:
        plt.cla()
        plt.plot(signal)
        plt.pause(do_plot)
    return signal


def centroid(signal,x=None):
    if x is None:
        x = np.arange(len(signal))
    return np.sum(x*signal)/np.sum(signal)


def iterative_centroid(signal,ref_x,sb_half_width,n_iterations,iteration_step_px):
    x = np.arange(len(signal))
    ref_x_temp = ref_x
    for k in range(n_iterations):
        print ref_x_temp,sb_half_width,k,iteration_step_px
        left = int(round(ref_x_temp-sb_half_width+k*iteration_step_px))
        right = int(round(ref_x_temp+sb_half_width-k*iteration_step_px))
        sig_temp = signal[left:right+1]
        x_temp = x[left:right+1]
        ref_x_temp = centroid(sig_temp,x_temp)
    return ref_x_temp


def maxcentered_centroid(signal,ref_x,sb_half_width,c_half_width):
    x = np.arange(len(signal))
    left = int(round(ref_x-sb_half_width))
    right = int(round(ref_x+sb_half_width))
    # copy to avoid modifying signal, since we'll be using the
    # same signal to test multiple algorithms
    # also, put zeros outside the search box so that we can
    # just use argmax on the whole vector
    temp = np.zeros(signal.shape)
    temp[left:right+1] = signal[left:right+1]
    x_peak = np.argmax(temp)

    # recenter the smaller centroiding box
    # on x_peak
    left = int(round(x_peak-c_half_width))
    right = int(round(x_peak+c_half_width))
    temp = temp[left:right+1]
    x = x[left:right+1]
    return centroid(temp,x)
    #return x_peak



def spot_and_com(x0,sigma,dc,sb_half_width,n_iterations,iteration_step_px,c_half_width):
    sensor_size = 100
    amplitude = 1000.0
    round_output = True
    noise_gain = 1.0
    do_plot=False
    x = np.arange(sensor_size)
    ref_x = x.mean()
    spot = create_spot(x0=x0,sensor_size=sensor_size,noise_gain=noise_gain,dc=dc,round_output=round_output,amplitude=amplitude,sigma=sigma)
    results = [None]*3
    results[0] = centroid(spot)
    results[1] = iterative_centroid(spot,ref_x,sb_half_width,n_iterations,iteration_step_px) # this leads to a final size of FWHM
    results[2] = maxcentered_centroid(spot,ref_x,sb_half_width,c_half_width)
    return np.array(results)


def compare_methods(func,variable_vector,variable_name,solution_vector,title_string=''):

    if not type(solution_vector)==np.ndarray:
        solution_vector = np.ones(len(variable_vector))*solution_vector
    
    results = []
    for v,s in zip(variable_vector,solution_vector):
        results.append(func(v)-s)

    labels = ['simple COM','iterative COM','max-centered COM']
    plt.figure()
    plt.plot(variable_vector,results)
    plt.legend(labels)
    plt.xlabel(variable_name)
    plt.ylabel('error (px)')
    plt.title(title_string)
    plt.pause(.1)


x0_default = 45.0
sigma_default = 3.0
dc_default = 100.0

sb_half_width_default = 10
n_iterations_default = 3
iteration_step_px_default = 2

c_half_width_default = 7
    
f_foo = lambda foo: spot_and_com(x0_default,sigma_default,dc_default,
                            sb_half_width_default,n_iterations_default,
                            iteration_step_px_default,c_half_width_default)
f_dc = lambda dc: spot_and_com(x0_default,sigma_default,dc,
                               sb_half_width_default,n_iterations_default,
                               iteration_step_px_default,c_half_width_default)

f_x0 = lambda x0: spot_and_com(x0,sigma_default,dc_default,
                               sb_half_width_default,n_iterations_default,
                               iteration_step_px_default,c_half_width_default)
f_sigma = lambda sigma: spot_and_com(x0_default,sigma,dc_default,
                            sb_half_width_default,n_iterations_default,
                            iteration_step_px_default,np.ceil(sigma)*2)
f_niter = lambda niter: spot_and_com(x0_default,sigma_default,dc_default,
                            sb_half_width_default,niter,
                            1,c_half_width_default)
f_sbhw = lambda sbhw: spot_and_com(x0_default,sigma_default,dc_default,
                            sbhw,n_iterations_default,
                            iteration_step_px_default,c_half_width_default)


compare_methods(f_dc,np.arange(0,2500),'DC level (ADU)',x0_default,'DC-related error (spot peak 1000 ADU)')
compare_methods(f_x0,np.arange(40,60,.1),'spot location',np.arange(40,60,.1),'aberration-dependent error')
compare_methods(f_sigma,np.arange(.1,10.0,.1),'spot size',x0_default,'spot-size dependent error')
compare_methods(f_niter,range(1,8),'number of iterations',x0_default,'iteration n error')
compare_methods(f_sbhw,range(3,12),'search box half width',x0_default,'search box size')

plt.show()
print f
sys.exit()

    
sigma = 3.0
fwhm = 2*np.sqrt(2*np.log(2))*sigma
c_half_width = np.ceil(fwhm)//2
noise_gain = 1.0

# test the impact of DC on the three methods
dc_range = np.arange(0,500)
results = np.zeros((len(dc_range),3))
for idx,dc in enumerate(dc_range):
    x0 = 45.0
    spot = create_spot(x0=x0,sensor_size=sensor_size,noise_gain=noise_gain,dc=dc,round_output=True,amplitude=1000.0,sigma=3.0)
    results[idx,0] = centroid(spot)
    results[idx,1] = iterative_centroid(spot,ref_x,10,3,2) # this leads to a final size of FWHM
    results[idx,2] = steve_centroid(spot,ref_x,10,c_half_width)

plt.plot(dc_range,results)
plt.legend(labels)

# test the impact of DC on the three methods
x0_range = np.arange(40.0,60.0,.05)
results = np.zeros((len(x0_range),3))
for idx,x0 in enumerate(x0_range):
    spot = create_spot(x0=x0,sensor_size=sensor_size,noise_gain=noise_gain,dc=100.0,round_output=True,amplitude=1000.0,sigma=3.0)
    results[idx,0] = centroid(spot)
    results[idx,1] = iterative_centroid(spot,ref_x,10,3,2) # this leads to a final size of FWHM
    results[idx,2] = steve_centroid(spot,ref_x,10,c_half_width)

plt.figure()
plt.plot(x0_range-ref_x,(results.T-x0_range).T)
plt.legend(labels)








plt.show()



sys.exit()
