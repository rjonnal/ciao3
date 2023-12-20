try:
    from PyQt5.QtMultimedia import QSound,QSoundEffect,QAudioDeviceInfo
    use_audio = True
except ImportError as ie:
    use_audio = False
    
from PyQt5.QtCore import QUrl,pyqtSlot
from PyQt5.Qt import QApplication
import ciao_config as ccfg
import sys
import numpy as np
import time
import os


class Beeper:

    def __init__(self,nskip=3):

        self.interval = nskip+1
        self.n = 0
        if use_audio:
            qadi = QAudioDeviceInfo()
            codecs = qadi.supportedCodecs()
            codec_exists = len(codecs)
            
        self.active = ('audio_directory' in dir(ccfg) and 'error_tones' in dir(ccfg) and use_audio)
        self.tone_dict = {}

    #@pyqtSlot()
    def cache_tones(self):
        if self.active:
            print('Caching beeper tones...')
            for minmax,tone_string in ccfg.error_tones:
                key = self.err_to_int(minmax[0])
                tonefn = os.path.join(ccfg.audio_directory,'%s.wav'%tone_string)
                if os.path.exists(tonefn):
                    val = QSound(tonefn)
                    self.tone_dict[key] = val
                #self.tonepg_dict[key] = pygame.mixer.Sound(tonefn)
            print('Done!')
        else:
            print('Not caching tones because qtmultimedia failed.')
            
    def err_to_int(self,err):
        return int(np.floor(err*1e8))

    def beep(self,error_in_nm):
        if self.active:
            #print(1)
            k = self.err_to_int(error_in_nm)
            if k in list(self.tone_dict.keys()) and self.n==0:
                se = self.tone_dict[k]
                se.play()
        else:
            pass
            #print 'play %0.1f'%(error_in_nm*1e9)
        self.n = (self.n+1)%self.interval

