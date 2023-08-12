import numpy as np
from matplotlib import pyplot as plt
import os

font_size = 5
plt.rcParams.update({'font.size': font_size})

def save_modes_chart(outfn,poke,currents,mask):
    plt.clf()
    cmin,cmax = np.min(currents),np.max(currents)
    
    U,s,V = np.linalg.svd(poke)
    sy,sx = V.shape

    for y in range(sy):
        row = y//10
        col = y-row*10

        bottom = float(9-row)/10.0
        left = (float(col)/10.0+.05)/2.0
        width = .08/2.0
        height = .08
        
        cond = np.max(s)/s[y]
        temp = np.zeros(mask.shape)
        temp[np.where(mask)] = V[y,:]
        
        plt.axes([left,bottom,width,height])
        plt.imshow(temp,interpolation='none',cmap='hot')
        plt.xticks([])
        plt.yticks([])
        if cond<1000:
            plt.title('%0.1f'%cond,fontsize=font_size)
        else:
            plt.title('%0.1e'%cond,fontsize=font_size)
            
    plt.axes([0.65,0.5,0.32,0.35])
    plt.bar(range(len(s)),s)
    plt.xlabel('mode',fontsize=font_size)
    plt.ylabel('singular value',fontsize=font_size)

    tag = os.path.splitext(os.path.split(outfn)[1])[0]
    
    plt.title('%s,cmin=%0.3f,cmax=%0.3f'%(tag,cmin,cmax),fontsize=font_size)
    plt.axes([0.65,0.05,0.32,0.35])
    plt.bar(range(len(s)),np.max(s)/s)
    plt.xlabel('mode',fontsize=font_size)
    plt.ylabel('condition number',fontsize=font_size)
    try:
        plt.savefig(outfn,dpi=300)
    except Exception as e:
        plt.savefig(outfn)
