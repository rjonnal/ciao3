import numpy as np
import pandas as pd
import time
from . import centroid
import sys
from PyQt5.QtCore import (QThread, QTimer, pyqtSignal, Qt, QPoint, QLine,
                          QMutex, QObject, pyqtSlot)

from PyQt5.QtWidgets import (QApplication, QPushButton, QWidget,
                             QHBoxLayout, QVBoxLayout, QGraphicsScene,
                             QLabel,QGridLayout, QCheckBox, QFrame, QGroupBox,
                             QSpinBox,QDoubleSpinBox,QSizePolicy,QFileDialog,
                             QErrorMessage, QSlider)
from PyQt5.QtGui import QColor, QImage, QPainter, QPixmap, qRgb, QPen, QBitmap, QPalette, QIcon
import os
from matplotlib import pyplot as plt
import datetime
from .tools import error_message, now_string, prepend, colortable, get_ram, get_process
import copy
from .zernike import Reconstructor
import cProfile
import scipy.io as sio
from .poke_analysis import save_modes_chart
from ctypes import CDLL,c_void_p
from .search_boxes import SearchBoxes
from .reference_generator import ReferenceGenerator
import ciao_config as ccfg
from .frame_timer import FrameTimer,BlockTimer
from .poke import Poke

class DataBuffer:

    def __init__(self,size_limit=1000,columns=None,tag='buffer'):
        self.buf = []
        self.size_limit = size_limit
        self.size = 0
        self.t0 = time.time()
        self.cols = columns
        self.tag = tag
        
    def add(self,new_list):
        assert type(new_list)==list
        t = time.time()-self.t0
        new_list = [t]+new_list
        self.buf.append(new_list)
        self.size+=1
        while len(self.buf)>self.size_limit:
            self.buf = self.buf[1:]
            self.size-=1

    def save(self):
        folder = self.tag
        os.makedirs(folder,exist_ok=True)
        ns = now_string()
        if self.cols is None:
            cols = ['time (s)']+['col %05d'%k for k in range(len(self.buf[-1])-1)]
        else:
            cols = ['time (s)']+self.cols
            
        df = pd.DataFrame(self.buf,columns=cols)
        df.to_csv(os.path.join(folder,'%s_%s.csv'%(self.tag,ns)))
        self.buf = []
        self.size = 0
        
    def full(self):
        return self.size==self.size_limit
    
class Loop(QObject):

    finished = pyqtSignal()
    started = pyqtSignal()
    
    def __init__(self,sensor,mirror,verbose=0):
        super(Loop,self).__init__()

        self.verbose = verbose
        
        self.sensor = sensor
        self.active_lenslets = np.ones(self.sensor.n_lenslets).astype(int)
        self.mirror = mirror

        self.update_rate = ccfg.loop_update_rate

        n_lenslets = self.sensor.n_lenslets
        n_actuators = self.mirror.n_actuators
        
        self.poke = None
        self.closed = False
        self.safe = True

        cols = ['x slope %03d'%k for k in range(n_lenslets)]+['y  slope %03d'%k for k in range(n_lenslets)]
        self.buf = DataBuffer(columns=cols,tag='slopes_buffer')

        # try to load the poke file specified in
        # ciao_config.py; if it doesn't exist, create
        # a dummy poke with all 1's; this will result
        # in an inverse control matrix with very low
        # gains, i.e. the mirror won't be driven
        if not os.path.exists(ccfg.poke_filename):
            dummy = np.ones((2*n_lenslets,n_actuators))
            np.savetxt(ccfg.poke_filename,dummy)
            
        self.load_poke(ccfg.poke_filename)
        self.gain = ccfg.loop_gain
        self.loss = ccfg.loop_loss
        self.paused = False
        self.n = 0
        try:
            self.started.connect(self.sensor.beeper.cache_tones)
        except TypeError:
            pass
        self.update_timer = BlockTimer('Loop update method')
        try:
            self.profile_update_method = ccfg.profile_loop_update_method
        except:
            self.profile_update_method = False


    def __del__(self):
        self.buf.save()
        
    def start(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(1000/self.update_rate))
        self.started.emit()
        
    def has_poke(self):
        return self.poke is not None

    def pause(self):
        self.mirror.pause()
        self.sensor.pause()
        self.paused = True

    def unpause(self):
        self.mirror.unpause()
        self.sensor.unpause()
        self.paused = False
        print('loop unpaused')

    def set_safe(self,val):
        self.safe = val

    def set_paused(self,val):
        if val:
            self.mirror.pause()
            self.sensor.pause()
            self.paused = True
        else:
            self.mirror.unpause()
            self.sensor.unpause()
            self.paused = False
        
    @pyqtSlot()
    def update(self):

        if self.profile_update_method:
            self.update_timer.tick('start')
        if not self.paused:
            if self.verbose>=5:
                print('Updating loop.')


            self.sensor.update()

            if self.profile_update_method:
                self.update_timer.tick('sensor.update')
            
            current_active_lenslets = np.ones(self.active_lenslets.shape)
            
            # if we're in safe mode, check the boxes:
            if self.safe:
                current_active_lenslets[np.where(self.sensor.box_maxes<ccfg.spots_threshold)] = 0
            
            if self.closed and self.has_poke():


                lenslets_changed = not all(self.active_lenslets==current_active_lenslets)
                all_lenslets_active = np.sum(current_active_lenslets)==self.sensor.n_lenslets
                # if the lenslets have changed, we have two options:
                # 1. if they're not all active, and ccfg.poke_invert_on_demand
                #    is set to True, then we do one of two things:
                #    a. if the active lenslets have changed, then we need to
                #       re-invert and set ready_to_correct True
                #    b. if the active lenslets haven't changed, then we just
                #       set ready_to_correct true
                # 2. if they're not all active, and ccfg.poke_invert_on_demand
                #    is set to False, then we need to set ready_to_correct to False
                # 3. if they're all active, set ready_to_correct True

                if not all_lenslets_active:
                    if ccfg.poke_invert_on_demand:
                        if lenslets_changed:
                            self.poke.invert(mask=current_active_lenslets)
                        self.ready_to_correct = True
                    else:
                        self.ready_to_correct = False
                else:
                    self.ready_to_correct = True


                xs = self.sensor.x_slopes[np.where(current_active_lenslets)[0]]
                ys = self.sensor.y_slopes[np.where(current_active_lenslets)[0]]

                if self.ready_to_correct:
                    assert 2*len(xs)==self.poke.ctrl.shape[1]
                
                if self.verbose>=1:
                    error = self.sensor.error
                    pcount = int(round(error*1e8))
                    print('rms'+'.'*pcount)

                if self.ready_to_correct:
                    slope_vec = np.hstack((xs,ys))
                    command = self.gain * np.dot(self.poke.ctrl,slope_vec)

                    command = self.mirror.get_command()*(1-self.loss) - command
                    self.mirror.set_command(command)
                    self.mirror.update()

                    if self.profile_update_method:
                        self.update_timer.tick('mirror.update')
                    
                    if self.verbose>=1:
                        if command.max()>ccfg.mirror_command_max*.99:
                            print('actuator saturated')
                        if command.min()<ccfg.mirror_command_min*.99:
                            print('actuator saturated')
                else:
                    print('not ready to correct')

            self.active_lenslets[:] = current_active_lenslets[:]

            
        self.n = self.n + 1
        if self.profile_update_method:
            self.update_timer.tick('end update')
            self.update_timer.tock()
            
        self.finished.emit()
        self.buf.add(list(np.hstack((self.sensor.x_slopes,self.sensor.y_slopes))))
        
        #if self.buf.full():
        #    #print(self.buf.buf)
        #    self.buf.save()
        #    sys.exit()
        
        
    def load_poke(self,poke_filename=None):
        try:
            poke = np.loadtxt(poke_filename)
        except Exception as e:
            error_message('Could not find %s.'%poke_filename)
            options = QFileDialog.Options()
            #options |= QFileDialog.DontUseNativeDialog
            poke_filename, _ = QFileDialog.getOpenFileName(
                            None,
                            "Please select a poke file.",
                            ccfg.poke_directory,
                            "Text Files (*.txt)",
                            options=options)
            poke = np.loadtxt(poke_filename)

        py,px = poke.shape
        expected_py = self.sensor.n_lenslets*2
        expected_px = self.mirror.n_actuators
        dummy = np.ones((expected_py,expected_px))
        
        try:
            assert (py==expected_py and px==expected_px)
        except AssertionError as ae:
            error_message('Poke matrix has shape (%d,%d), but (%d,%d) was expected. Using dummy matrix.'%(py,px,expected_py,expected_px))
            poke = dummy
            
        self.poke = Poke(poke)
        self.close_ok = ccfg.loop_condition_llim<self.get_condition_number()<ccfg.loop_condition_ulim

    def invert(self):
        if self.poke is not None:
            self.pause()
            time.sleep(1)
            self.poke.invert()
            time.sleep(1)
            QApplication.processEvents()
            self.unpause()
            time.sleep(.001)
            self.close_ok = ccfg.loop_condition_llim<self.get_condition_number()<ccfg.loop_condition_ulim

    def set_n_modes(self,n):
        try:
            self.poke.n_modes = n
        except Exception as e:
            print(e)

    def get_n_modes(self):
        out = -1
        try:
            out = self.poke.n_modes
        except Exception as e:
            print(e)
        return out

    def get_condition_number(self):
        out = -1
        try:
            out = self.poke.cutoff_cond
        except Exception as e:
            print(e)
        if out>2**32:
            out = np.inf
        return out
            
    def run_poke(self):
        cmin = ccfg.poke_command_min
        cmax = ccfg.poke_command_max
        n_commands = ccfg.poke_n_command_steps
        commands = np.linspace(cmin,cmax,n_commands)

        self.pause()
        time.sleep(1)
        
        n_lenslets = self.sensor.n_lenslets
        n_actuators = self.mirror.n_actuators
        
        x_mat = np.zeros((n_lenslets,n_actuators,n_commands))
        y_mat = np.zeros((n_lenslets,n_actuators,n_commands))
        ns = now_string()
        flat = self.mirror.flat
        
        for k_actuator in range(n_actuators):
            self.mirror.flatten()
            for k_command in range(n_commands):
                cur = commands[k_command]+flat[k_actuator]
                #print k_actuator,cur
                self.mirror.set_actuator(k_actuator,cur)
                #print k_actuator,k_command
                QApplication.processEvents()
                time.sleep(.01)
                self.sensor.sense()
                try:
                    if ccfg.save_poke_matrix_spots_images:
                        spots_folder = os.path.join(ccfg.poke_directory,'%s_spots_images'%ns)
                        if not os.path.exists(spots_folder):
                            os.makedirs(spots_folder)
                        filename = 'spots_%03d_%0.3f.npy'%(k_actuator,cur)
                        image = self.sensor.image
                        np.save(os.path.join(spots_folder,filename),image)
                except AttributeError as ae:
                    print(ae)
                    pass
                        
                x_mat[:,k_actuator,k_command] = self.sensor.x_slopes
                y_mat[:,k_actuator,k_command] = self.sensor.y_slopes
                self.finished.emit()
        # print 'done'
        self.mirror.flatten()
        
        d_commands = np.mean(np.diff(commands))
        d_x_mat = np.diff(x_mat,axis=2)
        d_y_mat = np.diff(y_mat,axis=2)

        x_response = np.mean(d_x_mat/d_commands,axis=2)
        y_response = np.mean(d_y_mat/d_commands,axis=2)
        poke = np.vstack((x_response,y_response))


        # After we make a new poke matrix, we will save it in
        # two files: an archive file that can be used to keep
        # track of old poke matrices, and the file specified
        # in the config file, e.g., 'poke.txt'.
        # The archive filename will use the time date string
        # generated above. This filename will also be used to
        # save the commands and the mirror mode chart PDF.
        
        poke_fn = ccfg.poke_filename
        archive_poke_fn = os.path.join(ccfg.poke_directory,'%s_poke.txt'%ns)
        archive_command_fn = os.path.join(ccfg.poke_directory,'%s_currents.txt'%ns)
        archive_chart_fn = os.path.join(ccfg.poke_directory,'%s_modes.pdf'%ns)
        
        np.savetxt(poke_fn,poke)
        np.savetxt(archive_poke_fn,poke)
        np.savetxt(archive_command_fn,commands)
        save_modes_chart(archive_chart_fn,poke,commands,self.mirror.mirror_mask)
        self.poke = Poke(poke)
        self.close_ok = ccfg.loop_condition_llim<self.get_condition_number()<ccfg.loop_condition_ulim
        time.sleep(1)
        self.unpause()

    def set_closed(self,val):
        self.closed = val


class SerialLoop(QObject):

    def __init__(self,sensor,mirror,verbose=0):
        super(SerialLoop,self).__init__()

        self.verbose = verbose
        
        self.sensor = sensor
        self.active_lenslets = np.ones(self.sensor.n_lenslets).astype(int)
        self.mirror = mirror

        n_lenslets = self.sensor.n_lenslets
        n_actuators = self.mirror.n_actuators
        
        self.poke = None
        self.closed = False

        # try to load the poke file specified in
        # ciao_config.py; if it doesn't exist, create
        # a dummy poke with all 1's; this will result
        # in an inverse control matrix with very low
        # gains, i.e. the mirror won't be driven
        if not os.path.exists(ccfg.poke_filename):
            dummy = np.ones((2*n_lenslets,n_actuators))
            np.savetxt(ccfg.poke_filename,dummy)
            
        self.load_poke(ccfg.poke_filename)
        self.gain = ccfg.loop_gain
        self.loss = ccfg.loop_loss
        self.paused = False

        self.n = 0
        
    def has_poke(self):
        return self.poke is not None

    def pause(self):
        self.paused = True

    def unpause(self):
        self.paused = False
        
    def start(self):
        if self.verbose>=5:
            print('Starting loop.')
        
            
    def update(self):
        if not self.paused:
            if self.verbose>=5:
                print('Updating loop.')
                
            if self.closed and self.has_poke():

                current_active_lenslets = np.zeros(self.active_lenslets.shape)
                current_active_lenslets[np.where(self.sensor.box_maxes>ccfg.spots_threshold)] = 1
                n_active_lenslets = int(np.sum(current_active_lenslets))
                
                if ccfg.poke_invert_on_demand:
                    if not all(self.active_lenslets==current_active_lenslets):
                        self.active_lenslets[:] = current_active_lenslets[:]
                        self.poke.invert(mask=self.active_lenslets)

                else:
                    if not self.sensor.n_lenslets==n_active_lenslets:
                        return

                xs = self.sensor.x_slopes[np.where(self.active_lenslets)[0]]
                ys = self.sensor.y_slopes[np.where(self.active_lenslets)[0]]
                if self.verbose>=1:
                    error = self.sensor.error
                    pcount = int(round(error*1e8))
                    print('rms'+'.'*pcount)
                
                slope_vec = np.hstack((xs,ys))
                command = self.gain * np.dot(self.poke.ctrl,slope_vec)
                command = self.mirror.get_command()*(1-self.loss) - command
                self.mirror.set_command(command)

                if self.verbose>=1:
                    if command.max()>ccfg.mirror_command_max*.95:
                        print('actuator saturated')
                    if command.min()<ccfg.mirror_command_min*.95:
                        print('actuator saturated')
                
            self.n = self.n + 1
                
    def load_poke(self,poke_filename=None):
        try:
            poke = np.loadtxt(poke_filename)
        except Exception as e:
            error_message('Could not find %s.'%poke_filename)
            options = QFileDialog.Options()
            #options |= QFileDialog.DontUseNativeDialog
            poke_filename, _ = QFileDialog.getOpenFileName(
                            None,
                            "Please select a poke file.",
                            ccfg.poke_directory,
                            "Text Files (*.txt)",
                            options=options)
            poke = np.loadtxt(poke_filename)

        py,px = poke.shape
        expected_py = self.sensor.n_lenslets*2
        expected_px = self.mirror.n_actuators
        dummy = np.ones((expected_py,expected_px))
        
        try:
            assert (py==expected_py and px==expected_px)
        except AssertionError as ae:
            error_message('Poke matrix has shape (%d,%d), but (%d,%d) was expected. Using dummy matrix.'%(py,px,expected_py,expected_px))
            poke = dummy
            
        self.poke = Poke(poke)

    def invert(self):
        if self.poke is not None:
            self.pause()
            time.sleep(1)
            self.poke.invert()
            time.sleep(1)
            QApplication.processEvents()
            self.unpause()
            time.sleep(1)

    def set_n_modes(self,n):
        try:
            self.poke.n_modes = n
        except Exception as e:
            print(e)

    def get_n_modes(self):
        out = -1
        try:
            out = self.poke.n_modes
        except Exception as e:
            print(e)
        return out

    def get_condition_number(self):
        out = -1
        try:
            out = self.poke.cutoff_cond
        except Exception as e:
            print(e)
        return out
            
    def run_poke(self):
        cmin = ccfg.poke_command_min
        cmax = ccfg.poke_command_max
        n_commands = ccfg.poke_n_command_steps
        commands = np.linspace(cmin,cmax,n_commands)

        self.pause()
        time.sleep(1)
        
        n_lenslets = self.sensor.n_lenslets
        n_actuators = self.mirror.n_actuators
        
        x_mat = np.zeros((n_lenslets,n_actuators,n_commands))
        y_mat = np.zeros((n_lenslets,n_actuators,n_commands))
        
        for k_actuator in range(n_actuators):
            self.mirror.flatten()
            for k_command in range(n_commands):
                cur = commands[k_command]
                #print k_actuator,cur
                self.mirror.set_actuator(k_actuator,cur)
                QApplication.processEvents()
                time.sleep(.01)
                self.sensor.sense()
                self.sensor_mutex.lock()
                x_mat[:,k_actuator,k_command] = self.sensor.x_slopes
                y_mat[:,k_actuator,k_command] = self.sensor.y_slopes
        # print 'done'
        self.mirror.flatten()
        
        d_commands = np.mean(np.diff(commands))
        d_x_mat = np.diff(x_mat,axis=2)
        d_y_mat = np.diff(y_mat,axis=2)

        x_response = np.mean(d_x_mat/d_commands,axis=2)
        y_response = np.mean(d_y_mat/d_commands,axis=2)
        poke = np.vstack((x_response,y_response))
        ns = now_string()


        # After we make a new poke matrix, we will save it in
        # two files: an archive file that can be used to keep
        # track of old poke matrices, and the file specified
        # in the config file, e.g., 'poke.txt'.
        # The archive filename will use the time date string
        # generated above. This filename will also be used to
        # save the commands and the mirror mode chart PDF.
        
        poke_fn = ccfg.poke_filename
        archive_poke_fn = os.path.join(ccfg.poke_directory,'%s_poke.txt'%ns)
        archive_command_fn = os.path.join(ccfg.poke_directory,'%s_currents.txt'%ns)
        archive_chart_fn = os.path.join(ccfg.poke_directory,'%s_modes.pdf'%ns)
        
        np.savetxt(poke_fn,poke)
        np.savetxt(archive_poke_fn,poke)
        np.savetxt(archive_command_fn,commands)
        save_modes_chart(archive_chart_fn,poke,commands,self.mirror.mirror_mask)
        self.poke = Poke(poke)
        
        time.sleep(1)
        self.unpause()

    def set_closed(self,val):
        self.closed = val
