import sys,os
sys.path.append(os.path.split(__file__)[0])
from ciao import cameras,simulator
import ciao_config as ccfg
from matplotlib import pyplot as plt
import numpy as np
from ciao import tools

# Call this with 'python record_reference_coordinages.py output_filename.txt'

if len(sys.argv)<2:
    print "Call this with 'python record_reference_coordinages.py output_filename.txt,"
    print "  where output_filename.txt is the file in which the reference coordinates are saved."
    sys.exit()
    
try:
    N = int(ccfg.reference_n_measurements)
except Exception as e:
    print e
    N = 1

if ccfg.simulate:
    cam = simulator.Simulator()
else:
    cam = cameras.get_camera()
    # if ccfg.camera_id=='pylon':
        # cam = cameras.PylonCamera()
    # elif ccfg.camera_id=='ace':
        # cam = cameras.AOCameraAce()
    # else:
        # sys.exit('camera_id %s not recognized.'%ccfg.camera_id)

output_filename = sys.argv[1]

# collect and average N images
im = cam.get_image().astype(np.float)
for k in range(N-1):
    im = im + cam.get_image()
im = im/float(N)

# Load data and initialize variables.
reference_mask = np.loadtxt(ccfg.reference_mask_filename)
d_lenslets = reference_mask.shape[0] # assumed to be square
pixels_per_lenslet = float(ccfg.lenslet_pitch_m)/float(ccfg.pixel_size_m)
total_width = d_lenslets*pixels_per_lenslet
border = (im.shape[0]-total_width)/2.0+pixels_per_lenslet/2.0

# Create a empty array for reference coordinates.
ref_xy = []

# Create a template image with simulated spots based on the reference mask
# and pixel size on the sensor.
# First, create an empty array with the same size as the camera image, and
# put 1's at the expected spot centers, based on the lenslet array
# geometry and sensor pixel size. Since these may not be centered on pixels,
# interpolate among the four pixels containing the 1.
template = np.zeros(im.shape)
for ly in range(d_lenslets):
    for lx in range(d_lenslets):
        if reference_mask[ly,lx]:
            py = border+pixels_per_lenslet*ly
            px = border+pixels_per_lenslet*lx
            ref_xy.append([px,py])
            for ky in [int(np.floor(py)),int(np.ceil(py))]:
                for kx in [int(np.floor(px)),int(np.ceil(px))]:
                    template[ky,kx] = (1.0-abs(ky-py))*(1.0-abs(kx-px))

# Blur and normalize the fake spots image, and normalize the real image.
lenslet_pitch_m = ccfg.lenslet_pitch_m
f = ccfg.lenslet_focal_length_m
L = ccfg.wavelength_m
pixel_size_m = ccfg.pixel_size_m
spot_sigma = (1.22*L*f/lenslet_pitch_m)/pixel_size_m
template = tools.gaussian_convolve(template,spot_sigma)
template = (template-template.mean())/(template.std())
im = (im-im.mean())/(im.std())

# ref_xy contains the coordinates of the lenslet centers, if the lenslet
# array were centered on the sensor.
ref_xy = np.array(ref_xy)

# Compute the corss-correlation between the real image and the fake one,
# in order to determine the displacements of the real spots from those
# stored in ref_xy.
nxc = np.abs(np.fft.ifft2(np.fft.fft2(template)*np.fft.fft2(im)))
py,px = np.unravel_index(np.argmax(nxc),nxc.shape)
sy,sx = nxc.shape
if py>=sy//2:
    py = py - sy
if px>=sx//2:
    px = px - sx

# Correct ref_xy accordingly.
ref_xy = ref_xy+np.array([px,py])

# Plot
m1x = 'Please adjust x coordinates of reference as necessary.'
m1y = 'Please adjust y coordinates of reference as necessary.'
m2x = 'To shift by a whole column, enter an integer.'
m2y = 'To shift by a whole row, enter an integer.'
m3 = 'To shift by pixels enter a floating point number, e.g. 1.5, -3.0.'
m4 = 'Enter 0 to finish.'
while True:
    plt.subplot(1,2,1)
    plt.cla()
    plt.imshow(template,cmap='gray')
    plt.subplot(1,2,2)
    plt.cla()
    plt.imshow(im,cmap='gray')
    plt.plot(ref_xy[:,0],ref_xy[:,1],'rx')
    plt.pause(.1)
    answer = raw_input('%s\n%s\n%s\n%s\n'%(m1x,m2x,m3,m4))
    if answer=='0':
        break
    elif answer.find('.')>-1:
        ref_xy[:,0] = ref_xy[:,0] + float(answer)
    else:
        ref_xy[:,0] = ref_xy[:,0] + int(answer)*pixels_per_lenslet
while True:
    plt.subplot(1,2,1)
    plt.cla()
    plt.imshow(template,cmap='gray')
    plt.subplot(1,2,2)
    plt.cla()
    plt.imshow(im,cmap='gray')
    plt.plot(ref_xy[:,0],ref_xy[:,1],'rx')
    plt.pause(.1)
    answer = raw_input('%s\n%s\n%s\n%s\n'%(m1y,m2y,m3,m4))
    if answer=='0':
        break
    elif answer.find('.')>-1:
        ref_xy[:,1] = ref_xy[:,1] + float(answer)
    else:
        ref_xy[:,1] = ref_xy[:,1] + int(answer)*pixels_per_lenslet

# And save the output
np.savetxt(output_filename,ref_xy,fmt='%0.3f')
