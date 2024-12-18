import ciao_config as ccfg
import numpy as np
import time,sys

class Poke:
    def __init__(self,poke_matrix):
        self.poke = poke_matrix
        poke_rows = poke_matrix.shape[1]
        self.n_modes = min(ccfg.loop_n_control_modes,poke_rows)
        
        self.n_ctrl_stored = 0
        self.ctrl_dict = {}
        self.ctrl_key_list = []
        self.invert()

    def mask_to_key(self,mask):
        if mask is None:
            return 'None'
        else:
            return '_'.join(['%d'%idx for idx in np.where(mask==0)[0]])+'_%d'%self.n_modes

    def get_stored_ctrl(self,mask):
        key = self.mask_to_key(mask)
        try:
            out = self.ctrl_dict[key]
        except KeyError as ke:
            out = None

    def store_ctrl(self,mask,ctrl):
        key = self.mask_to_key(mask)
        if key in list(self.ctrl_dict.keys()):
            return
        self.ctrl_dict[key] = ctrl
        self.ctrl_key_list.append(key)
        self.n_ctrl_stored+=1
        #self.print_dict_info()
        assert self.n_ctrl_stored==len(list(self.ctrl_dict.keys()))
        assert self.n_ctrl_stored==len(self.ctrl_key_list)

    def print_dict_info(self):
        print('N stored:',self.n_ctrl_stored)
        print('Current dictionary:')
        print(list(self.ctrl_dict.keys()))
        print('Current key list:')
        print(self.ctrl_key_list)
        print()

    def trim_ctrl_dict(self):
        if self.n_ctrl_stored<=ccfg.ctrl_dictionary_max_size:
            return
        else:
            n_to_remove = self.n_ctrl_stored-ccfg.ctrl_dictionary_max_size
            print('Removing %d'%n_to_remove)
            for k in range(n_to_remove):
                key = self.ctrl_key_list[k]
                self.print_dict_info()
                print('Key to remove: %s'%key)
                del self.ctrl_dict[key]
                self.n_ctrl_stored-=1
                self.ctrl_key_list.remove(key)
        
    def invert(self,subtract_mean=False,mask=None):
        self.ctrl = self.get_stored_ctrl(mask)
        if self.ctrl is not None:
            return
        
        t0 = time.time()

        poke = self.poke.copy()

        #mask = np.round(np.random.rand(poke.shape[0]//2)).astype(np.int)
        if mask is not None:
            double_mask = np.hstack((mask,mask))
            poke = poke[np.where(double_mask)[0],:]

        double_n_lenslets,n_actuators = poke.shape

        if subtract_mean:
            # subtract mean influence across actuators from
            # each actuator's influence
            # transpose, broadcast, transpose back:
            m_poke = np.mean(poke,axis=1)
            poke = (poke.T - m_poke).T

        U,s,V = np.linalg.svd(poke)
        self.full_cond = (s[0]/s).max()
        self.cutoff_cond = s[0]/s[self.n_modes-1]
        
        # zero upper modes
        if self.n_modes<n_actuators:
            s[self.n_modes:] = 0

        term1 = V.T
        term2 = np.zeros([n_actuators,double_n_lenslets])
        term2[:n_actuators,:n_actuators] = np.linalg.pinv(np.diag(s))
        term3 = U.T
        ctrlmat = np.dot(np.dot(term1,term2),term3)
        dt = time.time()-t0


        sanity_check = False
        if sanity_check:
            # double check the explicit Moore-Penrose pseudoinverse
            # above with LAPACK implementation (pinv)
            cutoff_cond = s[self.n_modes]/s[0]
            test = np.linalg.pinv(poke,cutoff_cond)
            if np.allclose(test,ctrlmat):
                print('Pseudoinverse is correct.')
                sys.exit()
            else:
                print('Pseudoinverse is incorrect.')
                sys.exit()
            
        self.ctrl = ctrlmat
        print('SVD %d modes %0.4e'%(self.n_modes,self.cutoff_cond))
        self.store_ctrl(mask,self.ctrl)
        self.trim_ctrl_dict()
        self.print_dict_info()
        print()
