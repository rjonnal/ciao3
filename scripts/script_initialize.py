import ciao_config as ccfg
import os,sys,shutil
import numpy as np

print('Running CIAO initialization script.')

# Check to make sure the required directories are present, and if not, make them.
print('Checking for required directories...')
required_directories = [ccfg.reference_directory,ccfg.dm_directory,ccfg.poke_directory,ccfg.logging_directory,ccfg.simulator_cache_directory]

for d in required_directories:
    try:
        os.makedirs(d)
        print('Creating directory %s.'%d)
    except OSError as ose:
        if os.path.exists(d):
            print('Directory %s exists. Okay.'%d)

# The application assumes that mirror mask and flat files exist, specified by mirror_mask_filename
# and mirror_flat_filename in ciao_config.py. Verify that these exist.
print('Checking for mirror mask file...')
if not os.path.exists(ccfg.mirror_mask_filename):
    sys.exit('Please create a mirror mask file %s, as described in documentation.'%ccfg.mirror_mask_filename)
else:
    mirror_mask = np.loadtxt(ccfg.mirror_mask_filename)
    print('Mirror mask file %s loaded.'%ccfg.mirror_mask_filename)

print('Checking for mirror flat file...')
if not os.path.exists(ccfg.mirror_flat_filename):
    answer = input('Mirror flat file %s not found. Create all-zero flat file? [Y/n] '%ccfg.mirror_flat_filename)
    if answer.lower() in ['y','']:
        flat = np.zeros(int(np.sum(mirror_mask)))
        np.savetxt(ccfg.mirror_flat_filename,flat,fmt='%0.6f')
        print('Writing all zeros to %s.'%ccfg.mirror_flat_filename)
    else:
        sys.exit('Please create a mirror flat file %s, as described in documentation.'%ccfg.mirror_flat_filename)
else:
    flat = np.loadtxt(ccfg.mirror_flat_filename)
    print('Mirror flat file %s loaded.'%ccfg.mirror_flat_filename)

# The application assumes that a reference mask file exists, specified by reference_mask_filename
# in ciao_config.py. Verify that this exists.
print('Checking for reference mask file...')
if not os.path.exists(ccfg.reference_mask_filename):
    sys.exit('Please create a reference mask file %s, as described in documentation.'%ccfg.reference_mask_filename)
else:
    reference_mask = np.loadtxt(ccfg.reference_mask_filename)
    
# The application assumes that one of two reference coordinate files exist, specified by
# reference_coordinate_filename and reference_coordinate_boostrap_filename; if the former
# exists, it is used, and if not, the latter is used, in principle to permit the measurement
# of real reference coordinates.
print('Checking for reference coordinates file...')
if not os.path.exists(ccfg.reference_coordinates_filename):
    if not os.path.exists(ccfg.reference_coordinates_bootstrap_filename):
        sys.exit('Please create a reference coordinates file %s, as described in documentation.'%ccfg.reference_coordinates_filename)
    else:
        ref = np.loadtxt(ccfg.reference_coordinates_bootstrap_filename)
        print('Bootstrap reference file present, but no true reference file. Please make sure to use the UI to record true reference coordinates.')
else:
    ref = np.loadtxt(ccfg.reference_coordinates_filename)

n_actuators = len(flat)
n_lenslets = int(np.sum(reference_mask))
print('Sensor has %d lenslets and mirror has %d actuators.'%(n_lenslets,n_actuators))

# Now load the poke matrix and verify that it exists and has the correct size.
print('Checking for poke matrix file %s...'%ccfg.poke_filename)
if not os.path.exists(ccfg.poke_filename):
    answer = input('Poke file %s not found. Create all-one poke file? [Y/n] '%ccfg.poke_filename)
    if answer.lower() in ['y','']:
        poke = np.zeros((n_lenslets*2,n_actuators))
        noise = np.random.randn(n_lenslets*2,n_actuators)*1e-6
        poke = poke+noise
        np.savetxt(ccfg.poke_filename,poke,fmt='%0.6f')
        print('Writing all ones to %s.'%ccfg.poke_filename)
    else:
        sys.exit('Exiting without writing poke file.')
else:
    poke = np.loadtxt(ccfg.poke_filename)
    print('Poke file loaded.')
    
print('Checking poke matrix size...')
poke_n_rows = n_lenslets*2
poke_n_cols = n_actuators
if (not poke.shape[0]==poke_n_rows) or (not poke.shape[1]==poke_n_cols):
    answer = input('Poke shape (%d,%d) disagrees with 2x number of lenslets (%d) and number of actuators (%d). Create all-one poke file? [Y/n] '%(poke.shape[0],poke.shape[1],n_lenslets*2,n_actuators))
    if answer.lower() in ['y','']:
        poke = np.zeros((n_lenslets*2,n_actuators))
        noise = np.random.randn(n_lenslets*2,n_actuators)*1e-6
        poke = poke+noise
        bak_poke_filename = ccfg.poke_filename.strip()+'.bak'
        shutil.copyfile(ccfg.poke_filename,bak_poke_filename)
        np.savetxt(ccfg.poke_filename,poke,fmt='%0.6f')
        print('Writing all ones to %s.'%ccfg.poke_filename)
    else:
        sys.exit('Exiting without writing poke file.')
    
else:
    poke = np.loadtxt(ccfg.poke_filename)
    print('Poke matrix size (%d,%d) is good.'%(poke.shape[0],poke.shape[1]))
