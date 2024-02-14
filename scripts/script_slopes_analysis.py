import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import ciao_config as ccfg
import os,sys


# To make movies, install fig2gif and imagemagick:
# To install fig2gif, run `git clohe https://github.com/rjonnal/fig2gif`
#   in a PYTHONPATH folder (e.g., `c:\code`).
# To install imagemagick, see https://imagemagick.org/script/download.php#windows
#   for Windows, or run `sudo apt install imagemagick` for Debian or Ubuntu linux.

try:
    from fig2gif import GIF
    make_movies = True
except:
    make_movies = False


figure_folder = 'slopes_figures'
data_folder = 'slopes_data'
os.makedirs(figure_folder,exist_ok=True)
os.makedirs(data_folder,exist_ok=True)
    
pupil_size = ccfg.beam_diameter_m

try:
    buf_fn = sys.argv[1]
except IndexError as ie:
    #buf_fn = './slopes_buffer/RSJ_eye_closed_loop_five_iterations.csv'
    sys.exit('Please pass the slopes buffer file name as a command line argument, e.g.: python script_slopes_analysis.py path/to/slopes/buffer.csv')
    
tag = os.path.splitext(os.path.split(buf_fn)[1])[0]

# get time array and slope array from csv, via pandas
df = pd.read_csv(buf_fn)
arr = np.array(df)
t = arr[:,1]
t = t - np.min(t)
slopes_all = arr[:,2:]

# load reconstructors and masks
zernike_matrix = np.loadtxt('etc/wf/zernike_matrix.txt')
wavefront_matrix = np.loadtxt('etc/wf/wavefront_matrix.txt')
slope_matrix = np.loadtxt('etc/wf/slope_matrix.txt')
ref_mask = np.loadtxt('etc/ref/reference_mask.txt')
ref = np.loadtxt('etc/ref/reference.txt')


# plot zernike coefficients for a single measurement
slopes = slopes_all[0,:]
coefs = np.dot(zernike_matrix,slopes)
coef_indices = np.arange(len(coefs))
plt.figure()
plt.bar(coef_indices,coefs)
plt.xlabel('zernike index')
plt.ylabel('zernike coefficient')
plt.savefig(os.path.join(figure_folder,'%s_zernike_coefficients.png'%tag))

# plot wavefront
wavefront_arr = np.dot(wavefront_matrix,coefs)*(pupil_size/2.0)
wavefront = np.zeros(ref_mask.shape)
wavefront[np.where(ref_mask)] = wavefront_arr
error = np.std(wavefront_arr)*1e9

plt.figure()
plt.subplot(1,2,1)
plt.imshow(wavefront)
plt.colorbar()
plt.title('rms error = %0.1f nm'%error)
ax = plt.axes([0.6,.2,0.35,.8],projection='3d')
mask_width = ref_mask.shape[1]
coords = np.linspace(-pupil_size/2.0,pupil_size/2.0,mask_width)
xx,yy = np.meshgrid(coords,coords)
surf = ax.plot_wireframe(xx,yy,wavefront,rstride=1,cstride=1,color='k')
ax.view_init(elev=60., azim=40)
ax.axes.set_zlim3d(bottom=-1e-6, top=1e-6)
plt.savefig(os.path.join(figure_folder,'%s_wavefront.png'%tag))


# compute all coefs
coefs_all = []

for k in range(slopes_all.shape[0]):
    slopes = slopes_all[k,:]
    coefs = np.dot(zernike_matrix,slopes)
    coefs_all.append(coefs)
    
coefs_all = np.array(coefs_all)


# temporal power spectrum of coefs
coefs_all_fft = np.fft.fft(coefs_all,axis=0)
tps = np.mean(np.abs(coefs_all_fft)**2,axis=1)
dt = np.median(np.diff(t))
freq = np.fft.fftfreq(len(tps),dt)

positive_idx = np.where(freq>=0)[0]
tps = tps[positive_idx]
freq = freq[positive_idx]

#coefs_all_fft = np.fft.fftshift(coefs_all_fft)
#freq = np.fft.fftshift(freq)

plt.figure()
plt.loglog(freq,tps)
plt.xlabel('frequency (Hz)')
plt.ylabel('power')
plt.savefig(os.path.join(figure_folder,'%s_temporal_power_spectrum.png'%tag))
tps_out = np.vstack((freq,tps)).T
np.savetxt(os.path.join(data_folder,'%s_tps_freq_power.txt'%tag),tps_out)

plt.show()
sys.exit()

coef_indices = np.arange(len(coefs))
wavefront = np.zeros(ref_mask.shape)
if make_movies:
    mov = GIF('figures/%s_aberrations.gif'%tag,fps=round(1/dt),autoclean=False)

z_min = np.min(coefs_all)
z_max = np.max(coefs_all)
zernike_ylim = (z_min-1e-5,z_max+1e-5)
zernike_ylim = (-1e-5,1e-5)

fig = plt.figure()
errs = []
tmax = 40.0

kmax = np.where(t>tmax)[0][0]

for k in range(slopes_all.shape[0]):
    if t[k]>tmax:
        break
    plt.clf()
    slopes = slopes_all[k,:]
    coefs = coefs_all[k,:]
    wavefront_arr = np.dot(wavefront_matrix,coefs)*(pupil_size/2.0)
    wavefront[np.where(ref_mask)] = wavefront_arr
    error = np.std(wavefront_arr)*1e9 # nm

    # zernike bar plot
    plt.subplot(2,2,1)
    plt.cla()
    plt.xlabel('zernike index')
    plt.ylabel('zernike coefficient')
    plt.bar(coef_indices[:15],coefs[:15])
    plt.ylim(zernike_ylim)
    plt.title('Zernike coefs')
    
    # error strip chart
    errs.append(error)
    plt.subplot(2,2,2)
    plt.cla()
    plt.plot(t[:k+1],errs,'g-',linewidth=2)
    plt.xlabel('time (s)')
    plt.ylabel('RMS error (nm)')
    plt.xlim((0,tmax))
    plt.ylim((0,400))
    #plt.xlim((t[k]-1,t[k]+1))
    plt.title('RMS error')
    #plt.xticks([])
    
    ax = plt.axes([0.1,.1,0.4,.3],projection='3d')
    ax.clear()
    surf = ax.plot_wireframe(xx,yy,wavefront,rstride=1,cstride=1,color='k')
    ax.view_init(elev=40., azim=40)
    ax.set_title('wavefront')
    ax.axes.set_zlim3d(bottom=-1e-7, top=1e-7)

    amp = np.zeros(ref_mask.shape)
    amp[np.where(ref_mask)] = 1.0
    pupil_function = amp*np.exp(1j*wavefront*2*np.pi/850e-9)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(pupil_function)))
    plt.subplot(2,2,4)
    plt.cla()
    plt.imshow(psf)
    plt.title('PSF')
    
    plt.tight_layout()
    if make_movies:
        mov.add(fig)
    plt.pause(.1)

if make_movies:
    mov.make()




