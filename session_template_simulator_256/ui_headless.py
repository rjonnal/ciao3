import sys,os
sys.path.append(os.path.split(__file__)[0])
import ciao_config as ccfg
from matplotlib import pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication
from ciao3.components.simulator import Simulator
from ciao3.components.sensors import Sensor
from ciao3.components.loops import Loop
from ciao3.components import cameras
from ciao3.components.mirrors import Mirror
from ciao3.components.ui import UI

if ccfg.simulate:
    sim = Simulator()
    sensor = Sensor(sim)
    mirror = sim
else:
    cam = cameras.get_camera()
    mirror = Mirror()
    sensor = Sensor(cam)

loop = Loop(sensor,mirror)
sensor.pseudocalibrate()
loop.closed = True
while True:
    loop.update()
    print('RMS error: %0.1f nm RMS'%(sensor.error*1e9))

