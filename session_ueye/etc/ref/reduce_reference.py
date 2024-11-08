import numpy as np
from matplotlib import pyplot as plt

ref = np.loadtxt('reference.txt')
refx = ref[:,0]
refy = ref[:,1]

refx = refx/2.0
refy = refy/2.0
refx = refx + (1280-1024)/2.0

ref[:,0] = refx
ref[:,1] = refy

np.savetxt('reference.txt',ref)

plt.plot(refx,refy,'bs')
plt.show()
