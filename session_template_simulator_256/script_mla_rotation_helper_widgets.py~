import sys,os
sys.path.append(os.path.split(__file__)[0])
import ciao_config as ccfg
from matplotlib import pyplot as plt
import numpy as np
from ciao3.components import tools

from ciao3.components import simulator
from ciao3.components import cameras


from matplotlib.widgets import Button, Slider

if ccfg.simulate:
    cam = simulator.Simulator()
else:
    cam = cameras.get_camera()

test = cam.get_image().astype(float)
sy,sx = test.shape
hpos = np.arange(sx)
vpos = np.arange(sy)

# Define initial parameters
init_ylim = 4095
init_exposure_time = 10000

def f(cam):
    im = cam.get_image().astype(float)
    hprof = np.mean(im,axis=0)
    vprof = np.mean(im,axis=1)
    return hprof,vprof

# Create the figure and the line that we will manipulate
fig, (hax, vax) = plt.subplots(1,2)
hprof,vprof = f(cam)

hline, = hax.plot(hpos,hprof)
vline, = vax.plot(vpos,vprof)

hax.set_xlabel('horizontal position')
vax.set_xlabel('vertical position')
hax.set_ylabel('amplitude')
hax.set_ylim((0,init_ylim))
vax.set_ylim((0,init_ylim))
vax.set_yticklabels([])

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)


# Make a horizontal slider to control the frequency.
axexp = fig.add_axes([0.25, 0.1, 0.65, 0.03])
exp_slider = Slider(
    ax=axexp,
    label='Camera exposure time',
    valmin=100,
    valmax=100000,
    valinit=init_exposure_time,
)

# Make a vertically oriented slider to control the amplitude
axylim = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
ylim_slider = Slider(
    ax=axylim,
    label="Plot Y limit",
    valmin=0,
    valmax=4095,
    valinit=init_ylim,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    cam.set_exposure(int(exp_slider.val))
    hprof,vprof = f(cam)
    hax.set_ylim((0,ylim_slider.val))
    vax.set_ylim((0,ylim_slider.val))
    hline.set_ydata(hprof)
    vline.set_ydata(vprof)
    fig.canvas.draw_idle()


# register the update function with each slider
exp_slider.on_changed(update)
ylim_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    exp_slider.reset()
    ylim_slider.reset()
    update(0)
    
button.on_clicked(reset)

plt.show()
sys.exit()

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
    
    
