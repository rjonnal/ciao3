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
spots_x = sx//2
spots_y = sy//2
padding=10

def f(cam,minval,maxval):
    im = cam.get_image().astype(float)
    hprof = np.mean(im,axis=0)
    vprof = np.mean(im,axis=1)
    bins = np.linspace(minval,maxval,64)
    hist,_ = np.histogram(im[::4,::4].ravel(),bins)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    return hprof,vprof,hist,bin_centers,im

# Create the figure and the line that we will manipulate

fig, ((hax, vax, sshax), (histax, spotsax, ssvax)) = plt.subplots(2,3)
fig.set_figwidth(12)
fig.set_figheight(8)

hprof,vprof,hist,bin_centers,spots = f(cam,init_spotsmin,init_spotsmax)

hline, = hax.plot(hpos,hprof,lw=2)
vline, = vax.plot(vpos,vprof,lw=2)

sshline, = sshax.plot(np.arange(2*padding)-padding,np.arange(2*padding))
ssvline, = ssvax.plot(np.arange(2*padding)-padding,np.arange(2*padding))


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
fig.subplots_adjust(left=0.18, bottom=0.15)


# Make a horizontal slider to control the frequency.
axexp = fig.add_axes([0.25, 0.025, 0.5, 0.03])
exp_slider = Slider(
    ax=axexp,
    label='Camera exposure time',
    valmin=100,
    valmax=100000,
    valinit=init_exposure_time,
)

axymin = fig.add_axes([0.05, 0.5, 0.01, 0.23])
ymin_slider = Slider(
    ax=axymin,
    label="ymin",
    valmin=0,
    valmax=4095,
    valinit=init_ymin,
    orientation="vertical"
)

axymax = fig.add_axes([0.09, 0.5, 0.01, 0.23])
ymax_slider = Slider(
    ax=axymax,
    label="ymax",
    valmin=0,
    valmax=4095,
    valinit=init_ymax,
    orientation="vertical"
)

axspotsmin = fig.add_axes([0.05, 0.15, 0.01, 0.23])
spotsmin_slider = Slider(
    ax=axspotsmin,
    label="immin",
    valmin=0,
    valmax=4095,
    valinit=init_spotsmin,
    orientation="vertical"
)

axspotsmax = fig.add_axes([0.09, 0.15, 0.01, 0.23])
spotsmax_slider = Slider(
    ax=axspotsmax,
    label="immax",
    valmin=0,
    valmax=4095,
    valinit=init_spotsmax,
    orientation="vertical"
)

def profile_properties(x,profile):
    maxval = np.mean(sorted(profile)[-3:])
    profile = profile/np.max(profile)
    vals = x**2/(-2*np.log(profile))
    valid = np.where(vals>=0)
    vals = vals[valid]
    sigma = np.nanmean(np.sqrt(vals))
    return maxval,sigma

# The function to be called anytime a slider's value changes
def update(val):
    cam.set_exposure(int(exp_slider.val))
    hprof,vprof,hist,bin_centers,spots = f(cam,spotsmin_slider.val,spotsmax_slider.val)

    temp = np.zeros(spots.shape)
    temp[spots_y-5:spots_y+5,spots_x-5:spots_x+5] = spots[spots_y-5:spots_y+5,spots_x-5:spots_x+5]
    peak = np.argmax(temp)
    peaky,peakx = np.unravel_index(peak,temp.shape)

    hspotprof = spots[peaky,peakx-padding:peakx+padding]
    vspotprof = spots[peaky-padding:peaky+padding,peakx]

    x = np.arange(-padding,padding)
    sshax.set_title('max=%0.1f, $\sigma=%0.1f$'%profile_properties(x,hspotprof),fontsize=18)
    ssvax.set_title('max=%0.1f, $\sigma=%0.1f$'%profile_properties(x,vspotprof),fontsize=18)

    
    sshax.set_xlim((-padding,padding))
    ssvax.set_xlim((-padding,padding))
    sshax.set_ylim((spotsmin_slider.val,spotsmax_slider.val))
    ssvax.set_ylim((spotsmin_slider.val,spotsmax_slider.val))
    
    hax.set_ylim((ymin_slider.val,ymax_slider.val))
    vax.set_ylim((ymin_slider.val,ymax_slider.val))
    histax.set_xlim((spotsmin_slider.val,spotsmax_slider.val))
    sshline.set_ydata(hspotprof)
    ssvline.set_ydata(vspotprof)
    
    
    hline.set_ydata(hprof)
    vline.set_ydata(vprof)
    histline.set_xdata(bin_centers)
    histline.set_ydata(hist)
    
    spotsimage.set_clim((spotsmin_slider.val,spotsmax_slider.val))
    spotsimage.set_data(spots)
    
    fig.canvas.draw_idle()


# register the update function with each slider
exp_slider.on_changed(update)
ymax_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
quitax = fig.add_axes([0.8, 0.025, 0.1, 0.03])
button = Button(quitax, 'Quit', hovercolor='0.975')


def quit(event):
    plt.close('all')
    #sys.exit()
    
button.on_clicked(quit)


def onclick(event):
    global spots_x,spots_y
    if event.xdata != None and event.ydata != None:
        if event.inaxes==spotsax:
            spots_x = int(round(event.xdata))
            spots_y = int(round(event.ydata))

cid = fig.canvas.mpl_connect('button_press_event', onclick)


do = lambda: update(0)

timer = fig.canvas.new_timer(interval=100)
timer.add_callback(do)
timer.start()

plt.show()
