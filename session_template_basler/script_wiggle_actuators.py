import sys,os
sys.path.append(os.path.split(__file__)[0])
import ciao
import ciao_config as ccfg
from matplotlib import pyplot as plt
import numpy as np
import sys,os

print ccfg.ciao_session

RMS = .5
DELAY = 1.0

mirror = ciao.mirrors.Mirror()
mask = mirror.mirror_mask
command = np.zeros(mirror.n_actuators)
command_map = np.zeros(mask.shape)

###################################################################################################

# cam = ciao.cameras.PylonCamera()

# output_directory = 'C:/Users/VSRI/Desktop/DM_actuators_new_board/'
# if not os.path.exists(output_directory):
    # try:
        # os.mkdir(output_directory)
    # except Exception as e:
        # print e
        # sys.exit()
        
# plt.figure(figsize=(10,10))

# def write_image(filename):
    # plt.axes([0,0,1,1])
    # plt.cla()
    # plt.imshow(im,cmap='gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig(os.path.join(output_directory,filename))


# for k in range(97):

    # mirror.flatten()
    
    # filename = 'actuator_%02d_flat.png'%k
    # im = cam.get_image()
    # write_image(filename)
    
    # command[k] = 0.3
    # mirror.set_command(command)
    # mirror.update()
    # plt.pause(.01)
    # filename = 'actuator_%02d_pushed.png'%k
    # im = cam.get_image()
    # write_image(filename)
    
    # command[k] = 0.0
    # mirror.set_command(command)
    # mirror.update()
    # plt.pause(.01)
    
    # print k
    
    
# sys.exit()
    
# ###################################################################################################
fig, ax = plt.subplots()
def quit(ev):
    mirror.flatten()
    sys.exit()
    
fig.canvas.mpl_connect('button_press_event', quit)
while True:
    #command[:] = np.random.randn(len(command))*RMS
  
       
    #command[43:54]=np.ones(11)*0.5   #Vertical actuators
    #command[48]=1   #Central actuators
    #command[[2,8,16,26,37,59,70,80,88,94] ]=np.ones(10) *0.5  #Horizontal actuator

    command[43:54]=0.5 #Vertical actuators
    command[48]=0.5   #Central actuators
    command[[2,8,16,26,37,59,70,80,88,94]]=0.5  #Horizontal actuator


    
    np.clip(command,0,1)
    mirror.set_command(command)
    mirror.update()
    command_map[np.where(mask)] = mirror.controller.command
    
    #plt.cla()
    ax.clear()
    
    ax.imshow(command_map,clim=(-1,1))

    plt.pause(DELAY)

    
