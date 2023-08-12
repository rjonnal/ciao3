import sys,os
sys.path.append(os.path.split(__file__)[0])
import ciao3
import ciao_config as ccfg
from matplotlib import pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication




if ccfg.simulate:
    sim = ciao3.simulator.Simulator()
    sensor = ciao3.sensors.Sensor(sim)
    mirror = sim
else:
    cam = ciao3.cameras.get_camera()
    mirror = ciao3.mirrors.Mirror()
    sensor = ciao3.sensors.Sensor(cam)
    
app = QApplication(sys.argv)
loop = ciao3.loops.Loop(sensor,mirror)
ui = ciao3.ui.UI(loop)
loop.start()
sys.exit(app.exec_())


