import sys,os
sys.path.append(os.path.join(os.path.split(__file__)[0],'config'))

import numpy as np
import time
import ciao_config as ccfg
from matplotlib import pyplot as plt
import datetime
from tools import error_message, now_string, prepend, colortable, get_ram, get_process
from zernike import Zernike,Reconstructor
from search_boxes import SearchBoxes
from frame_timer import FrameTimer
from reference_generator import ReferenceGenerator


class Eye:

    def __init__(self):
        pass


if __name__=='__main__':
    
    e = Eye()
        
