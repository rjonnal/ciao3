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
init_ymin = 100
init_ymax = 256
init_exposure_time = 10000
init_spotsmin = 100
init_spotsmax = 2048

bins = np.arange(0,4100,16)
bin_centers = (bins[:-1]+bins[1:])/2.0

def f(cam):
    im = cam.get_image().astype(float)
    hprof = np.mean(im,axis=0)
    vprof = np.mean(im,axis=1)
    hist,_ = np.histogram(im[::4,::4].ravel(),bins)
    return hprof,vprof,hist,im

# Create the figure and the line that we will manipulate

fig, ((hax, vax), (histax, spotsax)) = plt.subplots(2,2)
fig.set_figwidth(12)
fig.set_figheight(8)

hprof,vprof,hist,spots = f(cam)

hline, = hax.plot(hpos,hprof,lw=2)
vline, = vax.plot(vpos,vprof,lw=2)
histline, = histax.semilogy(bin_centers,hist,lw=2)
spotsimage = spotsax.imshow(spots)

hax.set_xlabel('horizontal position')
vax.set_xlabel('vertical position')
hax.set_ylabel('amplitude')
hax.grid(True)
vax.grid(True)

histax.set_xlabel('amplitude (ADU)')
histax.set_ylabel('count')
histax.grid(True)

spotsax.set_xticks([])
spotsax.set_yticks([])

hax.set_ylim((0,init_ymax))
vax.set_ylim((0,init_ymax))
vax.set_yticklabels([])

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.35, bottom=0.25)


# Make a horizontal slider to control the frequency.
axexp = fig.add_axes([0.25, 0.1, 0.65, 0.03])
exp_slider = Slider(
    ax=axexp,
    label='Camera exposure time',
    valmin=100,
    valmax=100000,
    valinit=init_exposure_time,
)

axymin = fig.add_axes([0.05, 0.25, 0.01, 0.63])
ymin_slider = Slider(
    ax=axymin,
    label="ymin",
    valmin=0,
    valmax=4095,
    valinit=init_ymin,
    orientation="vertical"
)

axymax = fig.add_axes([0.12, 0.25, 0.01, 0.63])
ymax_slider = Slider(
    ax=axymax,
    label="ymax",
    valmin=0,
    valmax=4095,
    valinit=init_ymax,
    orientation="vertical"
)

axspotsmin = fig.add_axes([0.19, 0.25, 0.01, 0.63])
spotsmin_slider = Slider(
    ax=axspotsmin,
    label="spotsmin",
    valmin=0,
    valmax=4095,
    valinit=init_spotsmin,
    orientation="vertical"
)

axspotsmax = fig.add_axes([0.26, 0.25, 0.01, 0.63])
spotsmax_slider = Slider(
    ax=axspotsmax,
    label="spotsmax",
    valmin=0,
    valmax=4095,
    valinit=init_spotsmax,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    cam.set_exposure(int(exp_slider.val))
    hprof,vprof,hist,spots = f(cam)
    hax.set_ylim((ymin_slider.val,ymax_slider.val))
    vax.set_ylim((ymin_slider.val,ymax_slider.val))
    histax.set_xlim((spotsmin_slider.val,spotsmax_slider.val))
    
    hline.set_ydata(hprof)
    vline.set_ydata(vprof)
    histline.set_ydata(hist)
    spotsimage.set_clim((spotsmin_slider.val,spotsmax_slider.val))
    spotsimage.set_data(spots)
    fig.canvas.draw_idle()


# register the update function with each slider
exp_slider.on_changed(update)
ymax_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
quitax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(quitax, 'Quit', hovercolor='0.975')


def quit(event):
    plt.close('all')
    #sys.exit()
    
button.on_clicked(quit)

do = lambda: update(0)

timer = fig.canvas.new_timer(interval=100)
timer.add_callback(do)
timer.start()

plt.show()
