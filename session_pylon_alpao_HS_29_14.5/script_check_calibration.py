import sys,os
sys.path.append(os.path.split(__file__)[0])
import ciao_config as ccfg
import numpy as np

print(dir(ccfg))

rcfn = ccfg.reference_coordinates_filename
rmfn = ccfg.reference_mask_filename
pfn = ccfg.poke_filename
mmfn = ccfg.mirror_mask_filename
mffn = ccfg.mirror_flat_filename


refmask = np.loadtxt(rmfn)
ref = np.loadtxt(rcfn)
poke = np.loadtxt(pfn)
mmask = np.loadtxt(mmfn)
mflat = np.loadtxt(mffn)

refmask_n_lenslets = int(np.sum(refmask))
ref_n_lenslets = ref.shape[0]
poke_n_lenslets = poke.shape[0]//2

poke_n_actuators = poke.shape[1]
mmask_n_actuators = int(np.sum(mmask))
mflat_n_actuators = len(mflat)

print('%s lenslet count: %d'%(rmfn,refmask_n_lenslets))
print('%s lenslet count: %d'%(rcfn,ref_n_lenslets))
print('%s lenslet count: %d'%(pfn,poke_n_lenslets))
print('%s actuator count: %d'%(pfn,poke_n_actuators))
print('%s actuator count: %d'%(mmfn,mmask_n_actuators))
print('%s actuator count: %d'%(mffn,mflat_n_actuators))

