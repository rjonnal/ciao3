import sys,os
import ciao
import ciao_config as ccfg
from matplotlib import pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication

if ccfg.simulate:
    sim = ciao.simulator.Simulator()
    cam = sim
    mirror = cam
else:
    cam = ciao.cameras.get_camera()
    mirror = ciao.mirrors.Mirror()
    
sensor = ciao.sensors.Sensor(cam)
    
app = QApplication(sys.argv)
loop = ciao.loops.Loop(sensor,mirror)
ui = ciao.ui.UI(loop)
loop.start()
sys.exit(app.exec_())


