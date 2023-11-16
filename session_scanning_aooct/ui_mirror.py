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
from ciao3.components.ui import MirrorUI

mirror = Mirror()
app = QApplication(sys.argv)
ui = MirrorUI(mirror)
sys.exit(app.exec_())
