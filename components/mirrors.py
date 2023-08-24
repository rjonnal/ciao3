import numpy as np
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
from .frame_timer import FrameTimer

class MirrorController(object):
    def __init__(self):
        self.cmax = ccfg.mirror_command_max
        self.cmin = ccfg.mirror_command_min
        self.command = np.zeros(ccfg.mirror_n_actuators,dtype=np.double)
        self.clipped = False
        
    def clip(self):
        self.clipped = (self.command.max()>=self.cmax or self.command.min()<=self.cmin)
        if self.clipped:
            self.command = np.clip(self.command,self.cmin,self.cmax)
                
    def set(self,vec):
        self.command[:] = vec[:]
        
    def send(self):
        return 1
        
class MirrorControllerCtypes(MirrorController):
    def __init__(self):
        super(MirrorControllerCtypes,self).__init__()
        self.acedev5 = CDLL("acedev5")
        self.mirror_id = self.acedev5.acedev5Init(0)
        self.command = np.zeros(ccfg.mirror_n_actuators,dtype=np.double)
        self.command_ptr = self.command.ctypes.data_as(c_void_p)
        self.send()
        
    def set(self,vec):
        assert len(vec)==len(self.command)
        self.command[:] = vec[:]

    def send(self):
        self.clip()
        return self.acedev5.acedev5Send(self.mirror_id,self.command_ptr)
        
        
class MirrorControllerPython(MirrorController):
    def __init__(self):
        print(sys.path)
        sys.path.append(os.path.dirname(__file__))
        print(sys.path)
        from .asdk import DM
        super(MirrorControllerPython,self).__init__()        
        self.mirror_id = ccfg.mirror_id
        #self.dm = DM(os.path.join(ccfg.dm_directory,self.mirror_id))
        self.dm = DM(self.mirror_id)
        
        n_actuators_queried = int( self.dm.Get('NBOfActuator') )
        try:
            assert n_actuators_queried==ccfg.mirror_n_actuators
        except AssertionError as ae:
            print('Number of actuator disagreement.')
        self.command[:] = 0.0 # this doesn't really matter
        
    def set(self,vec):
        self.command[:] = vec[:]
        
    def send(self):
        self.clip()
        return self.dm.Send(self.command)
        
        
class MirrorControllerPythonOld(MirrorController):

    def __init__(self):
        super(MirrorControllerPythonOld,self).__init__()
        sys.path.append(os.path.dirname(__file__))
        from .PyAcedev5 import PyAcedev5
        mirrorID = os.path.join(ccfg.dm_directory,ccfg.mirror_id)
        self.dm = PyAcedev5(ccfg.mirror_id)
        n_actuators_queried = int(self.dm.GetNbActuator())
        try:
            assert n_actuators_queried==ccfg.mirror_n_actuators
        except AssertionError as ae:
            print('Number of actuator disagreement.')
        self.command[:] = 0.0 # this doesn't really matter
        
    def set(self,vec):
        self.command[:] = vec[:]
        
    def send(self):
        self.clip()
        self.dm.values[:] = self.command[:]
        return self.dm.Send()

        
        
class Mirror:
    def __init__(self):
        
        try:
            self.controller = MirrorControllerPython()
            print('Mirror python initialization succeeded.')
        except Exception as e:
            print('Mirror python initialization failed:',e)
            try:
                self.controller = MirrorControllerPythonOld()
                print('Mirror python (old style) initialization succeeded.')
            except Exception as e:
                print('Mirror python (old style) initialization failed:',e)
                try:
                    self.controller = MirrorControllerCtypes()
                    print('Mirror c initialization succeeded.')
                except Exception as e:
                    print(e)
                    print('No mirror driver found. Using virtual mirror.')
                    self.controller = MirrorController()
            
        self.mirror_mask = np.loadtxt(ccfg.mirror_mask_filename)
        self.n_actuators = ccfg.mirror_n_actuators
        self.flat = np.loadtxt(ccfg.mirror_flat_filename)
        self.flat0 = np.loadtxt(ccfg.mirror_flat_filename)
        self.command_max = ccfg.mirror_command_max
        self.command_min = ccfg.mirror_command_min
        self.settling_time = ccfg.mirror_settling_time_s
        self.update_rate = ccfg.mirror_update_rate
        self.flatten()
        self.frame_timer = FrameTimer('Mirror',verbose=False)
        self.logging = False
        self.paused = False
        
    def update(self):
        if not self.paused:
            self.send()
        if self.logging:
            self.log()
        self.frame_timer.tick()

    def pause(self):
        print('mirror paused')
        self.paused = True

    def unpause(self):
        print('mirror unpaused')
        self.paused = False

    def send(self):
        self.controller.send()

    def flatten(self):
        self.controller.set(self.flat)
        self.send()

    def set_actuator(self,index,value):
        self.controller.command[index] = value
        self.send()
        
    def set_command(self,vec):
        self.controller.set(vec)
        
    def get_command(self):
        return self.controller.command
        
        
    def restore_flat(self):
        self.flat[:] = self.flat0[:]
        
    def set_flat(self):
        self.flat[:] = self.get_command()[:]
        np.savetxt(os.path.join(ccfg.dm_directory,'custom_flat_%s.txt'%now_string()),self.flat)
        
    def log(self):
        #outfn = os.path.join(ccfg.logging_directory,'mirror_%s.mat'%(now_string(True)))
        #d = {}
        #d['command'] = self.controller.command
        #sio.savemat(outfn,d)
        outfn = os.path.join(ccfg.logging_directory,'mirror_%s.txt'%(now_string(True)))
        np.savetxt(outfn,self.controller.command)
        

    def set_logging(self,val):
        self.logging = val
