import sys,os,datetime
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

spots = cam.get_image()
sy,sx = spots.shape

# Define initial parameters
init_ymin = 100
init_ymax = 256
init_exposure_time = 10000
init_spotsmin = 100
init_spotsmax = 2048
spots_x = sx//2
spots_y = sy//2
padding=10
output_folder = 'script_spots_info_output'
font_size = 14
queue_size = 25 # number of image stats to average for display

def iminfo(im):
    sy,sx = im.shape
    imf = np.fft.fft2(im)
    ac = np.real(np.fft.fftshift(np.fft.ifft2(imf*np.conj(imf))))
    ac = (ac-np.min(ac))/(np.max(ac)-np.min(ac))
    ps = np.fft.fftshift(np.abs(imf)**2)
    ps = 20*np.log10(ps)
    return ac[sy//2,sx//2-padding:sx//2+padding:],ps[sy//2,sx//2:]

hpos = np.arange(sx)
vpos = np.arange(sy)
ac,ps = iminfo(spots)

acx = np.arange(-padding,padding)

freq = np.fft.fftfreq(sx)
freq = freq[:sx//2]


class Queue:

    def __init__(self,max_length=queue_size):
        self.queue = np.ones(max_length)*np.nan
        self.idx = 0
        self.max_length = max_length
        
    def push(self,item):
        self.queue[self.idx] = item
        self.idx = (self.idx+1)%self.max_length

    def mean(self):
        return np.nanmean(self.queue)
        
    

def f(cam,minval,maxval):
    im = cam.get_image().astype(float)
    hprof = np.mean(im,axis=0)
    vprof = np.mean(im,axis=1)
    bins = np.linspace(minval,maxval,64)
    hist,_ = np.histogram(im[::4,::4].ravel(),bins)
    bin_centers = (bins[:-1]+bins[1:])/2.0
    return hprof,vprof,hist,bin_centers,im

# Create the figure and the line that we will manipulate

fig, ((hax, vax, sshax, acax), (histax, spotsax, ssvax, psax)) = plt.subplots(2,4)
fig.set_figwidth(12)
fig.set_figheight(8)

hprof,vprof,hist,bin_centers,spots = f(cam,init_spotsmin,init_spotsmax)

hline, = hax.plot(hpos,hprof,lw=2)
vline, = vax.plot(vpos,vprof,lw=2)

sshline, = sshax.plot(np.arange(2*padding)-padding,np.arange(2*padding))
ssvline, = ssvax.plot(np.arange(2*padding)-padding,np.arange(2*padding))

acline, = acax.plot(acx,ac)
psline, = psax.plot(freq,ps)

histline, = histax.plot(bin_centers,hist,lw=2)
spotsimage = spotsax.imshow(spots)

hax.set_title('horizontal profile')
vax.set_title('vertical profile')
hax.set_ylabel('amplitude')
hax.grid(True)
vax.grid(True)

histax.set_xlabel('amplitude (ADU)')
histax.set_ylabel('count')
histax.set_ylim((0,100))
histax.grid(True)

acax.set_xlabel('pixel')
acax.set_xlabel('autocorrelation')
acax.set_ylim((0,1))
psax.set_xlabel('freq $px^{-1}$')
psax.set_ylabel('power spectrum (dB)')
acax.grid(True)
psax.grid(True)

spotsax.set_xticks([])
spotsax.set_yticks([])

hax.set_ylim((0,init_ymax))
vax.set_ylim((0,init_ymax))
vax.set_yticklabels([])

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.18, bottom=0.15)


# Make a horizontal slider to control the frequency.
axexp = fig.add_axes([0.25, 0.025, 0.4, 0.03])
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

hspotmaxq = Queue()
hspotsigmaq = Queue()
vspotmaxq = Queue()
vspotsigmaq = Queue()
immaxq = Queue()
imstdq = Queue()


# The function to be called anytime a slider's value changes
def update(val):
    global spots
    cam.set_exposure(int(exp_slider.val))
    hprof,vprof,hist,bin_centers,spots = f(cam,spotsmin_slider.val,spotsmax_slider.val)
    ac,ps = iminfo(spots)
    
    immaxq.push(np.max(spots))
    imstdq.push(np.std(spots))

    temp = np.zeros(spots.shape)
    temp[spots_y-5:spots_y+5,spots_x-5:spots_x+5] = spots[spots_y-5:spots_y+5,spots_x-5:spots_x+5]
    peak = np.argmax(temp)
    peaky,peakx = np.unravel_index(peak,temp.shape)

    hspotprof = spots[peaky,peakx-padding:peakx+padding]
    vspotprof = spots[peaky-padding:peaky+padding,peakx]

    x = np.arange(-padding,padding)
    hspotmax,hspotsigma = profile_properties(x,hspotprof)
    vspotmax,vspotsigma = profile_properties(x,vspotprof)
    hspotmaxq.push(hspotmax)
    hspotsigmaq.push(hspotsigma)
    vspotmaxq.push(vspotmax)
    vspotsigmaq.push(vspotsigma)
    
    sshax.set_title('max=%0.1f, $\sigma=%0.1f$'%(hspotmaxq.mean(),hspotsigmaq.mean()),fontsize=font_size)
    ssvax.set_title('max=%0.1f, $\sigma=%0.1f$'%(vspotmaxq.mean(),vspotsigmaq.mean()),fontsize=font_size)

    
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
    spotsax.set_title('max=%0.1f, $\sigma=%01.f$'%(immaxq.mean(),imstdq.mean()),fontsize=font_size)

    acline.set_ydata(ac)
    psline.set_ydata(ps)
    
    fig.canvas.draw_idle()


# register the update function with each slider
exp_slider.on_changed(update)
ymax_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
quitax = fig.add_axes([0.85, 0.025, 0.075, 0.03])
quitbutton = Button(quitax, 'Quit', hovercolor='0.975')
saveax = fig.add_axes([0.75, 0.025, 0.075, 0.03])
savebutton = Button(saveax, 'Save', hovercolor='0.975')

def quit(event):
    plt.close('all')
    #sys.exit()

def save(event):
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(output_folder,exist_ok=True)
    np.savetxt(os.path.join(output_folder,'spots_%s.txt'%now),spots)
    fig.savefig(os.path.join(output_folder,'info_%s.png'%now),dpi=100)
    
    
quitbutton.on_clicked(quit)
savebutton.on_clicked(save)

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
