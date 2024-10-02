from ciao3.components.mirrors import Mirror
from ciao3.components.ui import MirrorUI
from matplotlib import pyplot as plt
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication

mirror = Mirror()
app = QApplication(sys.argv)
ui = MirrorUI(mirror)
sys.exit(app.exec_())

