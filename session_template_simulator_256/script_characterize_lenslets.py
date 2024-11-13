import sys,os
sys.path.append(os.path.split(__file__)[0])
import ciao_config as ccfg
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from ciao3.components.simulator import Simulator
from ciao3.components.sensors import Sensor
from ciao3.components import cameras

if ccfg.simulate:
    cam = Simulator()
    sensor = Sensor(cam)
else:
    cam = cameras.get_camera()
    sensor = Sensor(cam)


fig = plt.figure(figsize=(8,8))
ax_main = fig.add_axes([0,.4,.6,.6])
ax_h = fig.add_axes([0,.2,.6,.2])
ax_v = fig.add_axes([.6,.4,.2,.6])
ax_quit = fig.add_axes([.8,.0,.2,.2])
quit_button = Button(ax_quit,'Quit',hovercolor='0.975')



def quit(event):
    plt.close()
    sys.exit()
    
quit_button.on_clicked(quit)

while True:
    im = cam.get_image()
    sy,sx = im.shape
    x = np.arange(sx)
    y = np.arange(sy)
    
    hprof = np.mean(im,axis=0)
    vprof = np.mean(im,axis=1)
    ax_h.clear()
    ax_v.clear()
    ax_main.clear()

    ax_main.imshow(im,cmap='gray',aspect='equal')
    ax_h.plot(x,hprof)
    ax_v.plot(vprof,y)

    ax_h.text(10,np.min(hprof)-20,'%0.1f'%(np.max(hprof)),fontsize=24,va='top')
    ax_v.text(np.max(vprof)+20,10,'%0.1f'%(np.max(vprof)),fontsize=24,ha='left')
    
    plt.pause(.001)

