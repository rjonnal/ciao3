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
                             QErrorMessage, QSlider, QGraphicsView)
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
import os

# machine epsilon
eps = np.finfo(float).eps

#sensor_mutex = QMutex()
#mirror_mutex = QMutex()

try:
    os.mkdir('.gui_settings')
except Exception as e:
    print(e)


class StripChart(QWidget):
    """Permits display of a numerical quantitiy as well
    as a plot of its history."""

    def __init__(self,ylim=[-1,1],buffer_length=ccfg.plot_buffer_length,color=ccfg.plot_line_color,line_width=ccfg.plot_line_width,ytick_interval=None,print_function=lambda val: '%0.1e'%val,hlines=[0.0]):
        
        super(StripChart,self).__init__()
        self.layout = QVBoxLayout()
        self.lbl = QLabel()
        self.plot = QLabel()
        self.pixmap = QPixmap()
        self.buf = np.zeros(buffer_length)
        self.buffer_length = buffer_length
        self.buffer_current_index = 0
        self.layout.addWidget(self.lbl)
        self.layout.addWidget(self.plot)
        self.setLayout(self.layout)
        self.x = np.arange(buffer_length)
        self.hlines = hlines
        self.data_y_max = ylim[1]
        self.data_y_min = ylim[0]

        if ytick_interval is not None:
            t0 = np.fix(float(self.data_y_min)/float(ytick_interval))*ytick_interval
            t1 = np.fix(float(self.data_y_max)/float(ytick_interval))*ytick_interval
            self.yticks = np.arange(t0,t1,ytick_interval)
        else:
            self.yticks = []
        
        
        self.plot_width_px = ccfg.plot_width_px
        self.plot_height_px = ccfg.plot_height_px
        
        self.xtick0 = 0
        self.xtick1 = int(ccfg.plot_xtick_length)
        
        self.plot.setMinimumWidth(ccfg.plot_width_px)
        self.lbl.setMinimumWidth(ccfg.plot_width_px)
        self.xscale = int(float(ccfg.plot_width_px)/float(self.buffer_length-1))


        # there's a slight penalty for drawing a 32 bit pixmap instead of an 8
        # bit pixmap, but it dowsn't look like it really affects performance
        # so it's probably safe to make this false
        permit_only_gray_plots = False

        if permit_only_gray_plots:
            self.bmp = np.ones((ccfg.plot_height_px,ccfg.plot_width_px),dtype=np.uint8)*255
            bpl = int(self.bmp.nbytes/ccfg.plot_height_px)
            self.plot_background = QImage(self.bmp,ccfg.plot_width_px,ccfg.plot_height_px,
                                          bpl,
                                          QImage.Format_Indexed8)
        else:
            A = np.ones((ccfg.plot_height_px,ccfg.plot_width_px),dtype=np.uint32)*ccfg.plot_background_color[3]
            R = np.ones((ccfg.plot_height_px,ccfg.plot_width_px),dtype=np.uint32)*ccfg.plot_background_color[0]
            G = np.ones((ccfg.plot_height_px,ccfg.plot_width_px),dtype=np.uint32)*ccfg.plot_background_color[1]
            B = np.ones((ccfg.plot_height_px,ccfg.plot_width_px),dtype=np.uint32)*ccfg.plot_background_color[2]
            val = (A << 24 | R << 16 | G << 8 | B).flatten()
            bpl = int(val.nbytes/ccfg.plot_height_px)
            self.plot_background = QImage(val,ccfg.plot_width_px,ccfg.plot_height_px,
                                          bpl,
                                          QImage.Format_ARGB32)
            
        
        self.pixmap.convertFromImage(self.plot_background)
        
        self.lbl.setFixedHeight(ccfg.caption_height_px*2)
        
        self.setMinimumWidth(ccfg.plot_width_px)
        self.setMinimumHeight(ccfg.plot_height_px+ccfg.caption_height_px*2)
        self.print_function = print_function
        
        self.plot.setFrameShape(QFrame.Box)
        self.plot.setLineWidth(1)

        self.pen = QPen()
        self.pen.setColor(QColor(*color))
        line_width = int(np.ceil(line_width))

        self.pen.setWidth(line_width)

        self.ytick_pen = QPen()
        self.ytick_pen.setColor(QColor(0,0,0,255))
        self.ytick_pen.setWidth(1)
        self.ytick_pen.setStyle(Qt.DotLine)

        self.xtick_pen = QPen()
        self.xtick_pen.setColor(QColor(0,0,0,255))
        self.xtick_pen.setWidth(2)
        self.xtick_pen.setStyle(Qt.SolidLine)

        self.hline_pen = QPen()
        self.hline_pen.setColor(QColor(0,0,0,255))
        self.hline_pen.setWidth(1)
        self.hline_pen.setStyle(Qt.SolidLine)

        self.painter = QPainter()

        
    def setText(self,new_text):
        self.lbl.setText(new_text)
        

    def setValue(self,new_value):
        self.buf[self.buffer_current_index] = new_value
        self.buffer_current_index = (self.buffer_current_index+1)%self.buffer_length
        self.setText('%s\nsmoothed:%s'%(self.print_function(new_value),
                               self.print_function(self.buf.mean())))
        
    def scale_y(self,vec):
        h = self.plot.height()
        out = (h - (vec-self.data_y_min)/(self.data_y_max-self.data_y_min)*h)
        try:
            return out.astype(int)
        except AttributeError:
            return int(out)
        
    def setAlignment(self,new_alignment):
        self.lbl.setAlignment(new_alignment)

    def paintEvent(self,ev):

        pixmap = QPixmap()
        pixmap.convertFromImage(self.plot_background)
        
        self.painter.begin(pixmap)
        self.painter.setPen(self.ytick_pen)

        temp = self.scale_y(np.array(list(self.buf[self.buffer_current_index:])+list(self.buf[:self.buffer_current_index])))

        for yt in self.yticks:
            if True:#qlfix
                self.painter.drawLine(QLine(0,self.scale_y(yt),self.buffer_length*self.xscale,self.scale_y(yt)))


        self.painter.setPen(self.hline_pen)
        for hline in self.hlines:
            if True:#qlfix
                self.painter.drawLine(QLine(0,self.scale_y(hline),self.buffer_length*self.xscale,self.scale_y(hline)))


        self.painter.setPen(self.pen)
            
        for idx in range(self.buffer_length-1):
            x1 = (idx)*self.xscale
            x2 = (idx+1)*self.xscale
            y1 = temp[idx]
            y2 = temp[idx+1]

            if True:#qlfix:
                self.painter.drawLine(QLine(x1,y1,x2,y2))

            interval = self.buffer_length//10
            interval = ccfg.plot_xtick_interval
            if idx%interval==0:
                self.painter.setPen(self.xtick_pen)

                if True:#qlfix
                    self.painter.drawLine(QLine(
                        int(x1-self.buffer_current_index*self.xscale)%(self.buffer_length*self.xscale),
                        self.xtick0,
                        int(x1-self.buffer_current_index*self.xscale)%(self.buffer_length*self.xscale),
                        self.xtick1))
                self.painter.setPen(self.pen)
            
            #if True:#20<self.buffer_current_index<80:
            #    painter.drawEllipse(x1,y1,x2,y2)
        
        self.painter.end()
        self.plot.setPixmap(pixmap)


class Indicator(QLabel):
    """Permits time-averaged display and formatting of a numerical value."""

    def __init__(self,buffer_length=ccfg.plot_buffer_length,print_function=lambda val: '%0.1e'%val):
        
        super(Indicator,self).__init__()
        self.buf = np.zeros(buffer_length)
        self.buffer_length = buffer_length
        self.buffer_current_index = 0
        self.print_function = print_function
        
    def setValue(self,new_value):
        self.buf[self.buffer_current_index] = new_value
        self.buffer_current_index = (self.buffer_current_index+1)%self.buffer_length
        self.setText('%s'%self.print_function(self.buf.mean()))
        
        
class Overlay(QWidget):
    """Stores a list of 4-tuples (x1,x2,y1,y2), and can draw
    these over its pixmap as either lines between (x1,y1) and (x2,y2),
    or rects [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]."""
    
    def __init__(self,coords=[],color=(255,255,255,255),thickness=1,mode='rects',visible=True):
        self.coords = [[int(c) for c in x] for x in coords]
        self.pen = QPen()
        self.pen.setColor(QColor(*color))
        thickness = int(np.ceil(thickness))
        self.pen.setWidth(thickness)

        err_color = (color[2],color[1],color[0],color[3])
        self.err_pen = QPen()
        self.err_pen.setColor(QColor(*err_color))
        self.err_pen.setWidth(thickness*2)

        self.mode = mode
        self.visible = visible
        self.active = np.ones(len(coords))
        
    def draw(self,pixmap,downsample=1,active=None):
        if active is None:
            active=self.active
            
        d = int(downsample)
        if not self.visible:
            return
        if self.mode=='lines':
            painter = QPainter()
            painter.begin(pixmap)
            painter.setPen(self.pen)
            for index,(x1,x2,y1,y2) in enumerate(self.coords):
                if active[index]:
                    if True:#qlfix
                        painter.drawLine(QLine(int(x1)//d,int(y1)//d,int(x2)//d,int(y2)//d))
            painter.end()
        elif self.mode=='rects':
            painter = QPainter()
            painter.begin(pixmap)
            painter.setPen(self.pen)
            for index,(x1,x2,y1,y2) in enumerate(self.coords):
                if not active[index]:
                    painter.setPen(self.err_pen)
                else:
                    painter.setPen(self.pen)
                width = int(x2-x1)
                height = int(y2-y1)
                painter.drawRect(x1//d,y1//d,width//d,height//d)
            painter.end()
            
class ZoomDisplay(QWidget):
    def __init__(self,name,clim=(0,255),colormap='gray',zoom=1.0,overlays=[],downsample=1,n_contrast_steps=20):
        super(ZoomDisplay,self).__init__()
        self.name = name
        self.clim = clim
        self.zoom = zoom
        self.pixmap = QPixmap()
        self.label = QLabel()
        self.caption = QLabel()
        self.overlays = overlays
        self.colormap = colormap
        self.colortable = colortable(self.colormap)
        self.sized = False
        self.display_ratio = 1.0
        self.downsample = downsample
        
        self.mouse_coords = (ccfg.zoom_width/2.0,ccfg.zoom_height/2.0)
        self.sy,self.sx = 256,256 #initialize to something random

        layout = QHBoxLayout()
        subpanel = QWidget()
        sublayout = QVBoxLayout()
        sublayout.addWidget(self.caption)
        sublayout.addWidget(self.label)
        subpanel.setLayout(sublayout)
        layout.addWidget(subpanel)
        self.caption.setFixedHeight(ccfg.caption_height_px)
        self.caption.setText(name)
        self.label.setAlignment(Qt.AlignTop)

        
        # set up contrast sliders:
        slider_button_layout = QVBoxLayout()
        slider_layout = QHBoxLayout()
        
        self.n_steps = n_contrast_steps
        self.cmin_slider = QSlider(Qt.Vertical)
        self.cmax_slider = QSlider(Qt.Vertical)

        self.cmin_slider.setMinimum(0)
        self.cmax_slider.setMinimum(0)

        self.cmin_slider.setSingleStep(1)
        self.cmax_slider.setSingleStep(1)

        self.cmin_slider.setPageStep(5)
        self.cmax_slider.setPageStep(5)

        self.cmin_slider.setMaximum(self.n_steps)
        self.cmax_slider.setMaximum(self.n_steps)

        self.creset_button = QPushButton('Reset')
        self.cauto_button = QPushButton('Auto')
        self.creset_button.setFixedWidth(ccfg.contrast_button_width)
        self.cauto_button.setFixedWidth(ccfg.contrast_button_width)
        self.creset_button.clicked.connect(self.contrast_reset)
        self.cauto_button.clicked.connect(self.contrast_auto)

        self.set_display_clim()
        
        self.cmax_slider.setToolTip('%0.1e'%self.display_clim[1])
        self.cmin_slider.setToolTip('%0.1e'%self.display_clim[0])

        self.cmin_slider.setValue(self.real2slider(self.display_clim[0]))
        self.cmax_slider.setValue(self.real2slider(self.display_clim[1]))
        self.cmin_slider.valueChanged.connect(self.set_cmin)
        self.cmax_slider.valueChanged.connect(self.set_cmax)

        #layout.addWidget(self.cmin_slider)
        #layout.addWidget(self.cmax_slider)

        slider_layout.addWidget(self.cmin_slider)
        slider_layout.addWidget(self.cmax_slider)
        slider_button_layout.addLayout(slider_layout)
        slider_button_layout.addWidget(self.creset_button)
        slider_button_layout.addWidget(self.cauto_button)
        layout.addLayout(slider_button_layout)
        
        self.setLayout(layout)
        
    def mousePressEvent(self,e):
        self.mouse_coords = (e.x()*self.display_ratio,e.y()*self.display_ratio)

    def contrast_auto(self):
        try:
            self.display_clim = (self.data.min(),self.data.max())
        except Exception as e:
            print(e)
        self.set_sliders()

    def set_sliders(self):
        self.cmin_slider.setValue(self.real2slider(self.display_clim[0]))
        self.cmax_slider.setValue(self.real2slider(self.display_clim[1]))
            
    def contrast_reset(self):
        self.display_clim = [limit for limit in self.clim]
        #self.set_display_clim()
        self.set_sliders()
        
    def set_display_clim(self):
        try:
            self.display_clim = np.loadtxt(os.path.join('.gui_settings','clim_%s.txt'%self.name))
        except Exception as e:
            print(e)
            self.display_clim = self.clim
        
    def zoomed(self):
        x1 = int(round(self.mouse_coords[0]-ccfg.zoom_width//2))
        x2 = int(round(self.mouse_coords[0]+ccfg.zoom_width//2))
        if x1<0:
            x2 = x2 - x1
            x1 = 0
        if x2>=self.sx:
            x1 = x1 - (x2-self.sx) - 1
            x2 = self.sx - 1
            
        y1 = int(round(self.mouse_coords[1]-ccfg.zoom_height//2))
        y2 = int(round(self.mouse_coords[1]+ccfg.zoom_height//2))
        if y1<0:
            y2 = y2 - y1
            y1 = 0
        
        if y2>=self.sy:
            y1 = y1 - (y2-self.sy) - 1
            y2 = self.sy - 1

        return self.data[y1:y2,x1:x2]
        
    def real2slider(self,val):
        # convert a real value into a slider value
        return round(int((val-float(self.clim[0]))/float(self.clim[1]-self.clim[0])*self.n_steps))

    def slider2real(self,val):
        # convert a slider integer into a real value
        return float(val)/float(self.n_steps)*(self.clim[1]-self.clim[0])+self.clim[0]
    
    def set_cmax(self,slider_value):
        self.display_clim = (self.display_clim[0],self.slider2real(slider_value))
        np.savetxt('.gui_settings/clim_%s.txt'%self.name,self.display_clim)
        self.cmax_slider.setToolTip('%0.1e'%self.display_clim[1])
        
    def set_cmin(self,slider_value):
        self.display_clim = (self.slider2real(slider_value),self.display_clim[1])
        np.savetxt('.gui_settings/clim_%s.txt'%self.name,self.display_clim)
        self.cmin_slider.setToolTip('%0.1e'%self.display_clim[0])

    def show(self,data,active=None):
        data = data[::self.downsample,::self.downsample]
        self.data = data
        if not self.sized:
            self.label.setMinimumHeight(int(self.zoom*data.shape[0]))
            self.label.setMinimumWidth(int(self.zoom*data.shape[1]))
            self.sized = True
        try:
            cmin,cmax = self.display_clim
            bmp = np.round(np.clip((data.astype(float)-cmin)/(cmax-cmin+eps),0,1)*255).astype(np.uint8)
            self.sy,self.sx = bmp.shape
            n_bytes = bmp.nbytes
            bytes_per_line = int(n_bytes/self.sy)
            image = QImage(bmp,self.sx,self.sy,bytes_per_line,QImage.Format_Indexed8)
            image.setColorTable(self.colortable)
            self.pixmap.convertFromImage(image)
            for o in self.overlays:
                o.draw(self.pixmap,self.downsample,active)
            #self.label.setPixmap(self.pixmap)
            self.label.setPixmap(self.pixmap.scaled(self.label.width(),self.label.height(),Qt.KeepAspectRatio))
            self.display_ratio = float(self.sx)/float(self.label.width())
        except Exception as e:
            print(e)

class UI(QWidget):

    def __init__(self,loop):
        super(UI,self).__init__()
        self.sensor_mutex = QMutex()#loop.sensor_mutex
        self.mirror_mutex = QMutex()#loop.mirror_mutex
        self.loop = loop
        try:
            pass
            self.loop.finished.connect(self.update)
        except Exception as e:
            pass
        self.draw_boxes = ccfg.show_search_boxes
        self.draw_lines = ccfg.show_slope_lines
        self.init_UI()
        self.frame_timer = FrameTimer('UI',verbose=False)
        self.update_timer = BlockTimer('UI update method')
        try:
            self.profile_update_method = ccfg.profile_ui_update_method
        except:
            self.profile_update_method = False
            
        self.show()

    #def get_draw_boxes(self):
    #    return self.draw_boxes


    def __del__(self):
        print("hello?")
        self.update_timer.tock()
        
    def keyPressEvent0(self,event):
        if event.key()==Qt.Key_W:
            self.loop.sensor.search_boxes.up()
        if event.key()==Qt.Key_Z:
            self.loop.sensor.search_boxes.down()
        if event.key()==Qt.Key_A:
            self.loop.sensor.search_boxes.left()
        if event.key()==Qt.Key_S:
            self.loop.sensor.search_boxes.right()
        self.update_box_coords()

    def keyPressEvent(self,event):
        if event.key()==Qt.Key_W:
            self.loop.sensor.up()
        if event.key()==Qt.Key_Z:
            self.loop.sensor.down()
        if event.key()==Qt.Key_A:
            self.loop.sensor.left()
        if event.key()==Qt.Key_S:
            self.loop.sensor.right()
        self.update_box_coords()

        
    def update_box_coords(self):
        self.boxes_coords = []
        for x1,x2,y1,y2 in zip(self.loop.sensor.search_boxes.x1,
                               self.loop.sensor.search_boxes.x2,
                               self.loop.sensor.search_boxes.y1,
                               self.loop.sensor.search_boxes.y2):
            self.boxes_coords.append((x1,x2,y1,y2))
            self.overlay_boxes.coords = self.boxes_coords
    

    def update_focus(self,val):
        self.loop.sensor.set_defocus(val)
        self.update_box_coords()
            
    def set_draw_boxes(self,val):
        self.draw_boxes = val
        self.overlay_boxes.visible = val
    #def get_draw_lines(self):
    #    return self.draw_lines

    def set_draw_lines(self,val):
        self.draw_lines = val
        self.overlay_slopes.visible = val

    def show_mirror_ui(self):
        mui = MirrorUI(self.loop.mirror)
        
    def init_UI(self):
        self.setWindowIcon(QIcon('./icons/ciao.png'))
        self.setWindowTitle('CIAO')

        self.setMinimumWidth(ccfg.ui_width_px)
        self.setMinimumHeight(ccfg.ui_height_px)
        
        layout = QGridLayout()
        imax = 2**ccfg.bit_depth-1
        imin = 0

        self.boxes_coords = []
        for x1,x2,y1,y2 in zip(self.loop.sensor.search_boxes.x1,
                               self.loop.sensor.search_boxes.x2,
                               self.loop.sensor.search_boxes.y1,
                               self.loop.sensor.search_boxes.y2):
            self.boxes_coords.append((x1,x2,y1,y2))

        self.overlay_boxes = Overlay(coords=self.boxes_coords,mode='rects',color=ccfg.search_box_color,thickness=ccfg.search_box_thickness)

        self.overlay_slopes = Overlay(coords=[],mode='lines',color=ccfg.slope_line_color,thickness=ccfg.slope_line_thickness)
        
        self.id_spots = ZoomDisplay('Spots',clim=ccfg.spots_contrast_limits,colormap=ccfg.spots_colormap,zoom=0.25,overlays=[self.overlay_boxes,self.overlay_slopes],downsample=ccfg.spots_image_downsampling)
        
        layout.addWidget(self.id_spots,0,0,3,3)

        self.id_mirror = ZoomDisplay('Mirror',clim=ccfg.mirror_contrast_limits,colormap=ccfg.mirror_colormap,zoom=1.0)
        self.id_wavefront = ZoomDisplay('Wavefront',clim=ccfg.wavefront_contrast_limits,colormap=ccfg.wavefront_colormap,zoom=1.0)

        self.id_zoomed_spots = ZoomDisplay('Zoomed spots',clim=ccfg.spots_contrast_limits,colormap=ccfg.spots_colormap,zoom=5.0)
        
        layout.addWidget(self.id_mirror,0,3,1,1)
        layout.addWidget(self.id_wavefront,1,3,1,1)
        layout.addWidget(self.id_zoomed_spots,2,3,1,1)
        
        column_2 = QVBoxLayout()
        column_2.setAlignment(Qt.AlignTop)
        self.cb_closed = QCheckBox('Loop &closed')
        self.cb_closed.setChecked(self.loop.closed)
        self.cb_closed.stateChanged.connect(self.loop.set_closed)

        self.cb_paused = QCheckBox('Loop &paused')
        self.cb_paused.setChecked(self.loop.paused)
        self.cb_paused.stateChanged.connect(self.loop.set_paused)
        
        self.cb_safe = QCheckBox('Loop safe')
        self.cb_safe.setChecked(self.loop.safe)
        self.cb_safe.stateChanged.connect(self.loop.set_safe)
        
        loop_control_layout = QHBoxLayout()
        loop_control_layout.addWidget(self.cb_closed)
        loop_control_layout.addWidget(self.cb_paused)
        loop_control_layout.addWidget(self.cb_safe)
        
        
        self.cb_draw_boxes = QCheckBox('Draw boxes')
        self.cb_draw_boxes.setChecked(self.draw_boxes)
        self.cb_draw_boxes.stateChanged.connect(self.set_draw_boxes)

        self.cb_draw_lines = QCheckBox('Draw lines')
        self.cb_draw_lines.setChecked(self.draw_lines)
        self.cb_draw_lines.stateChanged.connect(self.set_draw_lines)

        self.cb_logging = QCheckBox('Logging')
        self.cb_logging.setChecked(False)
        self.cb_logging.stateChanged.connect(self.loop.sensor.set_logging)
        self.cb_logging.stateChanged.connect(self.loop.mirror.set_logging)

        self.pb_save_buffer = QPushButton('Save slopes')
        self.pb_save_buffer.clicked.connect(self.loop.buf.save)
        
        self.pb_poke = QPushButton('Measure poke matrix')
        self.pb_poke.clicked.connect(self.loop.run_poke)

        self.pb_record_reference = QPushButton('Record reference')
        self.pb_record_reference.clicked.connect(self.loop.sensor.record_reference)

        self.pb_pseudocalibrate = QPushButton('Pseudocalibrate')
        self.pb_pseudocalibrate.clicked.connect(self.loop.sensor.pseudocalibrate)
        
        self.pb_reload_reference = QPushButton('Reload reference')
        self.pb_reload_reference.clicked.connect(self.loop.sensor.reload_reference)
        
        
        self.pb_flatten = QPushButton('&Flatten')
        self.pb_flatten.clicked.connect(self.loop.mirror.flatten)

        self.pb_restore_flat = QPushButton('Restore flat')
        self.pb_restore_flat.clicked.connect(self.restore_flat)
        self.pb_restore_flat.setEnabled(False)
        

        self.pb_set_flat = QPushButton('Set flat')
        self.pb_set_flat.clicked.connect(self.set_flat)
        self.pb_set_flat.setCheckable(True)
        #print dir(self.pb_set_flat)
        #sys.exit()
        
        self.pb_quit = QPushButton('&Quit')
        self.pb_quit.clicked.connect(self.quit)

        poke_layout = QHBoxLayout()
        poke_layout.addWidget(QLabel('Modes:'))
        self.modes_spinbox = QSpinBox()
        n_actuators = int(np.sum(self.loop.mirror.mirror_mask))
        self.modes_spinbox.setMaximum(n_actuators)
        self.modes_spinbox.setMinimum(0)
        self.modes_spinbox.valueChanged.connect(self.loop.set_n_modes)
        self.modes_spinbox.setValue(self.loop.get_n_modes())
        poke_layout.addWidget(self.modes_spinbox)
        self.pb_invert = QPushButton('Invert')
        self.pb_invert.clicked.connect(self.loop.invert)
        poke_layout.addWidget(self.pb_invert)


        modal_layout = QHBoxLayout()
        modal_layout.addWidget(QLabel('Corrected Zernike orders:'))
        self.corrected_order_spinbox = QSpinBox()
        max_order = self.loop.sensor.reconstructor.N_orders
        self.corrected_order_spinbox.setMaximum(max_order)
        self.corrected_order_spinbox.setMinimum(0)
        self.corrected_order_spinbox.valueChanged.connect(self.loop.sensor.set_n_zernike_orders_corrected)
        self.corrected_order_spinbox.setValue(self.loop.sensor.get_n_zernike_orders_corrected())
        modal_layout.addWidget(self.corrected_order_spinbox)
        modal_layout.addWidget(QLabel('(%d -> no filtering)'%max_order))
        
        dark_layout = QHBoxLayout()
        self.cb_dark_subtraction = QCheckBox('Subtract dark')
        self.cb_dark_subtraction.setChecked(self.loop.sensor.dark_subtract)
        self.cb_dark_subtraction.stateChanged.connect(self.loop.sensor.set_dark_subtraction)
        dark_layout.addWidget(self.cb_dark_subtraction)
        
        dark_layout.addWidget(QLabel('Dark subtract N:'))
        self.n_dark_spinbox = QSpinBox()
        self.n_dark_spinbox.setMinimum(1)
        self.n_dark_spinbox.setMaximum(9999)
        self.n_dark_spinbox.valueChanged.connect(self.loop.sensor.set_n_dark)
        self.n_dark_spinbox.setValue(self.loop.sensor.n_dark)
        dark_layout.addWidget(self.n_dark_spinbox)
        
        self.pb_set_dark = QPushButton('Set dark')
        self.pb_set_dark.clicked.connect(self.loop.sensor.set_dark)
        dark_layout.addWidget(self.pb_set_dark)

        
        

        centroiding_layout = QHBoxLayout()
        centroiding_layout.addWidget(QLabel('Centroiding:'))
        self.cb_fast_centroiding = QCheckBox('Fast centroiding')
        self.cb_fast_centroiding.setChecked(self.loop.sensor.fast_centroiding)
        self.cb_fast_centroiding.stateChanged.connect(self.loop.sensor.set_fast_centroiding)
        centroiding_layout.addWidget(self.cb_fast_centroiding)
        self.centroiding_width_spinbox = QSpinBox()
        self.centroiding_width_spinbox.setMaximum(self.loop.sensor.search_boxes.half_width)
        self.centroiding_width_spinbox.setMinimum(0)
        self.centroiding_width_spinbox.valueChanged.connect(self.loop.sensor.set_centroiding_half_width)
        self.centroiding_width_spinbox.setValue(self.loop.sensor.centroiding_half_width)
        centroiding_layout.addWidget(QLabel('Fast centroiding half width'))
        centroiding_layout.addWidget(self.centroiding_width_spinbox)
        
        
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel('Background adjustment:'))
        self.bg_spinbox = QSpinBox()
        self.bg_spinbox.setValue(self.loop.sensor.background_correction)
        self.bg_spinbox.setMaximum(500)
        self.bg_spinbox.setMinimum(-500)
        self.bg_spinbox.valueChanged.connect(self.loop.sensor.set_background_correction)
        bg_layout.addWidget(self.bg_spinbox)


        aberration_layout = QHBoxLayout()

        aberration_layout.addWidget(QLabel('Defocus:'))
        self.f_spinbox = QDoubleSpinBox()
        self.f_spinbox.setValue(0.0)
        self.f_spinbox.setSingleStep(0.01)
        self.f_spinbox.setMaximum(10.0)
        self.f_spinbox.setMinimum(-10.0)
        #self.f_spinbox.valueChanged.connect(self.loop.sensor.set_defocus)
        self.f_spinbox.valueChanged.connect(self.update_focus)
        aberration_layout.addWidget(self.f_spinbox)




        
        aberration_layout.addWidget(QLabel('Astig 0:'))
        self.a0_spinbox = QDoubleSpinBox()
        self.a0_spinbox.setValue(0.0)
        self.a0_spinbox.setSingleStep(0.01)
        self.a0_spinbox.setMaximum(10.0)
        self.a0_spinbox.setMinimum(-10.0)
        self.a0_spinbox.valueChanged.connect(self.loop.sensor.set_astig0)
        aberration_layout.addWidget(self.a0_spinbox)

        aberration_layout.addWidget(QLabel('Astig 45:'))
        self.a1_spinbox = QDoubleSpinBox()
        self.a1_spinbox.setValue(0.0)
        self.a1_spinbox.setSingleStep(0.01)
        self.a1_spinbox.setMaximum(10.0)
        self.a1_spinbox.setMinimum(-10.0)
        self.a1_spinbox.valueChanged.connect(self.loop.sensor.set_astig1)
        aberration_layout.addWidget(self.a1_spinbox)



        aberration_layout.addWidget(QLabel('Tip:'))
        self.tip_spinbox = QDoubleSpinBox()
        self.tip_spinbox.setValue(0.0)
        self.tip_spinbox.setSingleStep(0.01)
        self.tip_spinbox.setMaximum(10.0)
        self.tip_spinbox.setMinimum(-10.0)
        self.tip_spinbox.valueChanged.connect(self.loop.sensor.set_tip)
        aberration_layout.addWidget(self.tip_spinbox)

        aberration_layout.addWidget(QLabel('Tilt:'))
        self.tilt_spinbox = QDoubleSpinBox()
        self.tilt_spinbox.setValue(0.0)
        self.tilt_spinbox.setSingleStep(0.01)
        self.tilt_spinbox.setMaximum(10.0)
        self.tilt_spinbox.setMinimum(-10.0)
        self.tilt_spinbox.valueChanged.connect(self.loop.sensor.set_tilt)
        aberration_layout.addWidget(self.tilt_spinbox)


        
        self.pb_aberration_reset = QPushButton('Reset')
        def reset():
            self.f_spinbox.setValue(0.0)
            self.a0_spinbox.setValue(0.0)
            self.a1_spinbox.setValue(0.0)
            self.tip_spinbox.setValue(0.0)
            self.tilt_spinbox.setValue(0.0)
            self.loop.sensor.aberration_reset()
        self.pb_aberration_reset.clicked.connect(reset)
        aberration_layout.addWidget(self.pb_aberration_reset)

        self.pb_mirror_ui = QPushButton('MirrorUI')
        self.pb_mirror_ui.clicked.connect(self.show_mirror_ui)
        aberration_layout.addWidget(self.pb_mirror_ui)
        
        
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel('Exposure (us):'))
        self.exp_spinbox = QSpinBox()
        exp = self.loop.sensor.cam.get_exposure()
        self.exp_spinbox.setSingleStep(100)
        self.exp_spinbox.setMaximum(1000000)
        self.exp_spinbox.setMinimum(100)
        self.exp_spinbox.setValue(exp)
        self.exp_spinbox.valueChanged.connect(self.loop.sensor.cam.set_exposure)
        exp_layout.addWidget(self.exp_spinbox)
        
        self.stripchart_error = StripChart(ylim=ccfg.error_plot_ylim,ytick_interval=ccfg.error_plot_ytick_interval,print_function=ccfg.error_plot_print_func,hlines=[0.0,ccfg.wavelength_m/14.0],buffer_length=ccfg.error_plot_buffer_length)
        self.stripchart_error.setAlignment(Qt.AlignRight)
        
        #self.stripchart_defocus = StripChart(ylim=ccfg.zernike_plot_ylim,ytick_interval=ccfg.zernike_plot_ytick_interval,print_function=ccfg.zernike_plot_print_func,buffer_length=ccfg.zernike_plot_buffer_length)
        #self.stripchart_defocus.setAlignment(Qt.AlignRight)

        self.lbl_tip = QLabel()
        self.lbl_tip.setAlignment(Qt.AlignRight)
        
        self.lbl_tilt = QLabel()
        self.lbl_tilt.setAlignment(Qt.AlignRight)
        
        self.lbl_cond = QLabel()
        self.lbl_cond.setAlignment(Qt.AlignRight)

        self.lbl_sensor_fps = QLabel()
        self.lbl_sensor_fps.setAlignment(Qt.AlignRight)

        self.ind_centroiding_time = Indicator(buffer_length=50,print_function=lambda x: '%0.0f us (centroiding)'%(x*1e6))
        self.ind_centroiding_time.setAlignment(Qt.AlignRight)

        self.ind_image_max = Indicator(buffer_length=10,print_function=lambda x: '%d ADU (max)'%x)
        self.ind_image_mean = Indicator(buffer_length=10,print_function=lambda x: '%d ADU (mean)'%x)
        self.ind_image_min = Indicator(buffer_length=10,print_function=lambda x: '%d ADU (min)'%x)
        self.ind_mean_box_background = Indicator(buffer_length=10,print_function=lambda x: '%d ADU (background)'%x)
        self.ind_buffer_size = Indicator(buffer_length=10,print_function=lambda x: '%d (buf length)'%x)
        
        
        self.ind_image_max.setAlignment(Qt.AlignRight)
        self.ind_image_mean.setAlignment(Qt.AlignRight)
        self.ind_image_min.setAlignment(Qt.AlignRight)
        self.ind_mean_box_background.setAlignment(Qt.AlignRight)
        self.ind_buffer_size.setAlignment(Qt.AlignRight)

        self.lbl_mirror_fps = QLabel()
        self.lbl_mirror_fps.setAlignment(Qt.AlignRight)
        
        self.lbl_ui_fps = QLabel()
        self.lbl_ui_fps.setAlignment(Qt.AlignRight)
        
        flatten_layout = QHBoxLayout()
        flatten_layout.addWidget(self.pb_flatten)
        flatten_layout.addWidget(self.pb_restore_flat)
        flatten_layout.addWidget(self.pb_set_flat)
        
        column_2.addLayout(flatten_layout)
        
        column_2.addLayout(loop_control_layout)
        #column_2.addWidget(self.cb_fast_centroiding)
        column_2.addLayout(aberration_layout)
        column_2.addLayout(exp_layout)
        column_2.addLayout(centroiding_layout)
        
        if ccfg.estimate_background:
            column_2.addLayout(bg_layout)
        column_2.addLayout(poke_layout)
        #column_2.addLayout(modal_layout)
        if ccfg.use_dark_subtraction:
            column_2.addLayout(dark_layout)


        annotations_layout = QHBoxLayout()
        annotations_layout.addWidget(self.cb_draw_boxes)
        annotations_layout.addWidget(self.cb_draw_lines)
        column_2.addLayout(annotations_layout)

        other_layout = QHBoxLayout()
        
        other_layout.addWidget(self.pb_poke)
        other_layout.addWidget(self.pb_pseudocalibrate)
        other_layout.addWidget(self.pb_quit)

        column_2.addLayout(other_layout)

        
        column_2.addWidget(self.stripchart_error)
        #column_2.addWidget(self.stripchart_defocus)
        column_2.addWidget(self.lbl_tip)
        column_2.addWidget(self.lbl_tilt)
        column_2.addWidget(self.lbl_cond)
        column_2.addWidget(self.ind_image_max)
        column_2.addWidget(self.ind_image_mean)
        column_2.addWidget(self.ind_image_min)
        column_2.addWidget(self.ind_mean_box_background)
        
        column_2.addWidget(self.ind_centroiding_time)
        column_2.addWidget(self.ind_buffer_size)
        
        column_2.addWidget(self.lbl_sensor_fps)
        column_2.addWidget(self.lbl_mirror_fps)
        column_2.addWidget(self.lbl_ui_fps)
        
        
        
        #column_2.addWidget(self.pb_reload_reference)
        
        column_2.addWidget(self.cb_logging)
        column_2.addWidget(self.pb_save_buffer)
        
        layout.addLayout(column_2,0,6,3,1)
        
        self.setLayout(layout)
        
    def quit(self):
        self.loop.sensor.cam.close()
        sys.exit()
        
    def flatten(self):
        self.mirror_mutex.lock()
        self.loop.mirror.flatten()
        self.mirror_mutex.unlock()
        
    def restore_flat(self):
        self.loop.mirror.restore_flat()
        self.pb_set_flat.setChecked(False)
        self.pb_set_flat.setFocus(False)
        self.pb_restore_flat.setEnabled(False)
        
    def set_flat(self):
        self.loop.mirror.set_flat()
        self.pb_set_flat.setChecked(True)
        self.pb_restore_flat.setEnabled(True)
        
    @pyqtSlot()
    def update(self):

        if self.profile_update_method:
            self.update_timer.tick('start')
            
        #self.mirror_mutex.lock()
        #self.sensor_mutex.lock()
        sensor = self.loop.sensor
        mirror = self.loop.mirror

        temp = [(x,xerr,y,yerr) for x,xerr,y,yerr in
                zip(sensor.search_boxes.x,sensor.x_centroids,
                    sensor.search_boxes.y,sensor.y_centroids)]

        self.overlay_slopes.coords = []
        for x,xerr,y,yerr in temp:
            dx = (xerr-x)*ccfg.slope_line_magnification
            dy = (yerr-y)*ccfg.slope_line_magnification
            x2 = x+dx
            y2 = y+dy
            self.overlay_slopes.coords.append((x,x2,y,y2))


        self.boxes_coords = []
        for x1,x2,y1,y2 in zip(self.loop.sensor.search_boxes.x1,
                               self.loop.sensor.search_boxes.x2,
                               self.loop.sensor.search_boxes.y1,
                               self.loop.sensor.search_boxes.y2):
            self.boxes_coords.append((x1,x2,y1,y2))

        #self.overlay_boxes = Overlay(coords=self.boxes_coords,mode='rects',color=ccfg.search_box_color,thickness=ccfg.search_box_thickness)




            
        if self.profile_update_method:
            self.update_timer.tick('create slope lines overlay')

        self.id_spots.show(sensor.image,self.loop.active_lenslets)
        
        if self.profile_update_method:
            self.update_timer.tick('show spots')


        mirror_map = np.zeros(mirror.mirror_mask.shape)
        mirror_map[np.where(mirror.mirror_mask)] = mirror.get_command()[:]
        self.id_mirror.show(mirror_map)
        self.id_wavefront.show(sensor.wavefront)

        self.id_zoomed_spots.show(self.id_spots.zoomed())
        
        #self.lbl_error.setText(ccfg.wavefront_error_fmt%(sensor.error*1e9))
        self.stripchart_error.setValue(sensor.error)
        #self.stripchart_defocus.setValue(sensor.zernikes[4]*ccfg.beam_diameter_m)
        
        self.lbl_tip.setText(ccfg.tip_fmt%(sensor.tip*1000000))
        self.lbl_tilt.setText(ccfg.tilt_fmt%(sensor.tilt*1000000))
        self.lbl_cond.setText(ccfg.cond_fmt%(self.loop.get_condition_number()))

        self.ind_image_max.setValue(sensor.image_max)
        self.ind_image_mean.setValue(sensor.image_mean)
        self.ind_image_min.setValue(sensor.image_min)
        self.ind_mean_box_background.setValue(sensor.get_average_background())
        self.ind_buffer_size.setValue(self.loop.buf.size)
        self.ind_centroiding_time.setValue(sensor.centroiding_time)
        
        self.lbl_sensor_fps.setText(ccfg.sensor_fps_fmt%sensor.frame_timer.fps)
        self.lbl_mirror_fps.setText(ccfg.mirror_fps_fmt%mirror.frame_timer.fps)
        self.lbl_ui_fps.setText(ccfg.ui_fps_fmt%self.frame_timer.fps)

        if self.loop.close_ok:
            self.cb_closed.setEnabled(True)
        else:
            self.cb_closed.setEnabled(False)
            self.cb_closed.setChecked(False)
            self.loop.closed = False

        if self.profile_update_method:
            self.update_timer.tick('end update')
            self.update_timer.tock()
            
        #self.mirror_mutex.unlock()
        #self.sensor_mutex.unlock()
            
            
    def select_single_spot(self,click):
        x = click.x()*self.downsample
        y = click.y()*self.downsample
        self.single_spot_index = self.loop.sensor.search_boxes.get_lenslet_index(x,y)

    def paintEvent(self,event):
        self.frame_timer.tick()

class MirrorUI(QWidget):

    def __init__(self,mirror):
        super(MirrorUI,self).__init__()
        self.mirror = mirror
        self.indices = np.ones(self.mirror.mirror_mask.shape,dtype=int)*-1
        self.n_actuators = int(np.sum(self.mirror.mirror_mask))
        self.n_actuators_x = self.mirror.mirror_mask.shape[1]
        self.n_actuators_y = self.mirror.mirror_mask.shape[0]
        
        self.indices[np.where(self.mirror.mirror_mask)] = np.arange(self.n_actuators,dtype=int)
        self.init_UI()
        self.show()

    def actuate(self,idx,val):
        self.mirror.set_actuator(idx,val)
        
    def init_UI(self):

        
        self.setWindowIcon(QIcon('./icons/ciao.png'))
        self.setWindowTitle('CIAO Mirror')
        
        layout = QGridLayout()

        ny,nx = self.indices.shape
        controls = []

        actuator_funcs = []

        self.sb_vec = []
        for y in range(ny):
            for x in range(nx):
                idx = self.indices[y,x]
                if idx==-1:
                    continue
                sb = QDoubleSpinBox()
                sb.setDecimals(3)
                sb.setSingleStep(0.001)
                sb.setMaximum(1.0)
                sb.setMinimum(-1.0)
                sb.setValue(self.mirror.flat[idx])
                sb.valueChanged.connect(lambda val,idx=idx: self.actuate(idx,val))
                hbox = QVBoxLayout()
                hbox.addWidget(QLabel('A%03d'%idx))
                hbox.addWidget(sb)
                w = QWidget()
                w.setLayout(hbox)
                layout.addWidget(w,x,y,1,1)
                self.sb_vec.append(sb)


        qb = QPushButton('&Quit')
        qb.clicked.connect(sys.exit)
        layout.addWidget(qb,self.n_actuators_y,0,1,1)
        
        qb = QPushButton('&Flatten')
        qb.clicked.connect(self.flatten)
        layout.addWidget(qb,self.n_actuators_y,1,1,1)

        qb = QPushButton('&Save')
        qb.clicked.connect(self.save)
        layout.addWidget(qb,self.n_actuators_y,2,1,1)

        #ta = QTextArea('temp.txt')
        #layout.addWidget(ta,self.n_actuators_y,3,1,1)

        self.mirror_zd = ZoomDisplay('mirror_ui',clim=ccfg.mirror_contrast_limits,colormap=ccfg.mirror_colormap,zoom=30.0)
        layout.addWidget(self.mirror_zd,0,self.n_actuators_x,*self.indices.shape)
        
        self.setLayout(layout)
        
    @pyqtSlot()
    def update(self):
        pass

    def paintEvent(self,event):
        mirror_map = np.zeros(self.mirror.mirror_mask.shape)
        mirror_map[np.where(self.mirror.mirror_mask)] = self.mirror.get_command()[:]
        self.mirror_zd.show(mirror_map)

    def flatten(self):
        ny,nx = self.indices.shape
        for y in range(ny):
            for x in range(nx):
                idx = self.indices[y,x]
                if idx==-1:
                    continue
                self.sb_vec[idx].setValue(self.mirror.flat[idx])
        self.mirror.flatten()


    def save(self):
        out = self.mirror.get_command()
        print("Saving current commands to 'manual_flat.txt'")
        np.savetxt('manual_flat.txt',out)
