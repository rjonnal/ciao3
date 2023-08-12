import numpy as np
import sys

if len(sys.argv)<3:
    print 'To run, call as follows:'
    print 'python script_make_mask.py N rad filename.txt'
    print 'where N is the width/height of the mask, rad is the radius in which to write ones'
    print 'e.g., for the SHWS use: python script_make_mask.py 20 9.6 reference_mask.txt'
    print 'and for the mirror mask use: python script_make_mask.py 11 5.5 mirror_mask.txt'
    sys.exit()
    
N = int(sys.argv[1])
rad = float(sys.argv[2])

try:
    outfn = sys.argv[3]
except Exception as e:
    outfn = None

xx,yy = np.meshgrid(np.arange(N),np.arange(N))

xx = xx - float(N-1)/2.0
yy = yy - float(N-1)/2.0

d = np.sqrt(xx**2+yy**2)

mask = np.zeros(xx.shape,dtype=np.uint8)
mask[np.where(d<=rad)] = 1

if not outfn is None:
    np.savetxt(outfn,mask,fmt='%d')
    print 'Mask with %d active elements written to %s.'%(np.sum(mask),outfn)
else:
    out = ''
    for row in mask:
        for col in row:
            out = out + '%d '%col
        out = out + '\n'
            
    print out

