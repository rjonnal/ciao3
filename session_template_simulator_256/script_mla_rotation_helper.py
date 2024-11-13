import sys,os
sys.path.append(os.path.split(__file__)[0])
import ciao_config as ccfg
from matplotlib import pyplot as plt
import numpy as np
from ciao3.components import tools

from ciao3.components import simulator
from ciao3.components import cameras

# Call this with 'python record_reference_coordinages.py output_filename.txt'

ylim = (0,400)
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


# collect and average N images
while True:
    im = cam.get_image().astype(float)
    hprof = np.mean(im,axis=0)
    vprof = np.mean(im,axis=1)
    plt.subplot(1,2,1)
    plt.cla()
    plt.plot(hprof)
    plt.ylim(ylim)
    plt.subplot(1,2,2)
    plt.cla()
    plt.plot(vprof)
    plt.ylim(ylim)
    plt.pause(.01)
    
    