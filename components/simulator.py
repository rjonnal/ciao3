import numpy as np
import time
import ciao_config as ccfg
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
from .zernike import Zernike
from .search_boxes import SearchBoxes
from .frame_timer import FrameTimer
from .reference_generator import ReferenceGenerator

class Simulator:

    def __init__(self):
        """The Simulator object simulates the camera, mirror, and dynamic aberrations
        of the system. CIAO can be run in simulation mode by instantiating a simulator
        object, then a sensor object using the simulator as its camera, and then a
        loop object using that sensor and the simulator in place of the mirror."""

        self.frame_timer = FrameTimer('simulator')
        
        # We need to define a meshes on which to build the simulated spots images
        # and the simulated wavefront:
        self.sy = ccfg.image_height_px
        self.sx = ccfg.image_width_px
        self.wavefront = np.zeros((self.sy,self.sx))

        # Some parameters for the spots image:
        self.dc = 100
        self.spots_range = 2000
        self.spots = np.ones((self.sy,self.sx))*self.dc
        self.image = self.spots.astype(np.uint16)
        self.spots = self.noise(self.spots)
        self.pixel_size_m = ccfg.pixel_size_m

        # compute single spot
        self.lenslet_pitch_m = ccfg.lenslet_pitch_m
        self.f = ccfg.lenslet_focal_length_m
        self.L = ccfg.wavelength_m
        fwhm_px = (1.22*self.L*self.f/self.lenslet_pitch_m)/self.pixel_size_m
        
        xvec = np.arange(self.sx)
        yvec = np.arange(self.sy)
        xvec = xvec-xvec.mean()
        yvec = yvec-yvec.mean()
        XX,YY = np.meshgrid(xvec,yvec)
        d = np.sqrt(XX**2+YY**2)
        
        self.beam_diameter_m = ccfg.beam_diameter_m
        self.beam_radius_m = self.beam_diameter_m/2.0
        
        self.disc_diameter = 170
        #self.disc_diameter = ccfg.beam_diameter_m/self.pixel_size_m # was just set to 110
        
        self.disc = np.zeros((self.sy,self.sx))
        self.disc[np.where(d<=self.disc_diameter)] = 1.0
        
        self.X = np.arange(self.sx,dtype=float)*self.pixel_size_m
        self.Y = np.arange(self.sy,dtype=float)*self.pixel_size_m
        self.X = self.X-self.X.mean()
        self.Y = self.Y-self.Y.mean()
        
        self.XX,self.YY = np.meshgrid(self.X,self.Y)


        self.RR = np.sqrt(self.XX**2+self.YY**2)
        self.mask = np.zeros(self.RR.shape)
        self.mask[np.where(self.RR<=self.beam_radius_m)] = 1.0


        use_partially_illuminated_lenslets = True
        if use_partially_illuminated_lenslets:
            d_lenslets = int(np.ceil(self.beam_diameter_m/self.lenslet_pitch_m))
        else:
            d_lenslets = int(np.floor(self.beam_diameter_m/self.lenslet_pitch_m))

        rad = float(d_lenslets)/2.0

        xx,yy = np.meshgrid(np.arange(d_lenslets),np.arange(d_lenslets))

        xx = xx - float(d_lenslets-1)/2.0
        yy = yy - float(d_lenslets-1)/2.0

        d = np.sqrt(xx**2+yy**2)

        self.lenslet_mask = np.zeros(xx.shape,dtype=np.uint8)
        self.lenslet_mask[np.where(d<=rad)] = 1
        self.n_lenslets = int(np.sum(self.lenslet_mask))

        self.x_lenslet_coords = xx*self.lenslet_pitch_m/self.pixel_size_m+self.sx/2.0
        self.y_lenslet_coords = yy*self.lenslet_pitch_m/self.pixel_size_m+self.sy/2.0
        in_pupil = np.where(self.lenslet_mask)
        self.x_lenslet_coords = self.x_lenslet_coords[in_pupil]
        self.y_lenslet_coords = self.y_lenslet_coords[in_pupil]

        
        self.lenslet_boxes = SearchBoxes(self.x_lenslet_coords,self.y_lenslet_coords,ccfg.search_box_half_width)

        #plt.plot(self.x_lenslet_coords,self.y_lenslet_coords,'ks')
        #plt.show()

        self.mirror_mask = np.loadtxt(ccfg.mirror_mask_filename)
        self.n_actuators = int(np.sum(self.mirror_mask))

        self.command = np.zeros(self.n_actuators)
        
        # virtual actuator spacing in magnified or demagnified
        # plane of camera
        actuator_spacing = ccfg.beam_diameter_m/float(self.mirror_mask.shape[0])
        ay,ax = np.where(self.mirror_mask)
        ay = ay*actuator_spacing
        ax = ax*actuator_spacing
        ay = ay-ay.mean()
        ax = ax-ax.mean()

        self.flat = np.loadtxt(ccfg.mirror_flat_filename)
        self.flat0 = np.loadtxt(ccfg.mirror_flat_filename)
        
        self.n_zernike_terms = ccfg.n_zernike_terms
        #actuator_sigma = actuator_spacing*0.75
        actuator_sigma = actuator_spacing*1.5

        
        key = '%d'%hash((tuple(ax),tuple(ay),actuator_sigma,tuple(self.X),tuple(self.Y),self.n_zernike_terms))
        key = key.replace('-','m')

        try:
            os.mkdir(ccfg.simulator_cache_directory)
        except OSError as e:
            pass

        cfn = os.path.join(ccfg.simulator_cache_directory,'%s_actuator_basis.npy'%key)
        
        try:
            self.actuator_basis = np.load(cfn)
            print('Loading cached actuator basis set...')
        except Exception as e:
            actuator_basis = []
            print('Building actuator basis set...')
            for x,y in zip(ax,ay):
                xx = self.XX - x
                yy = self.YY - y
                surf = np.exp((-(xx**2+yy**2)/(2*actuator_sigma**2)))
                surf = (surf - surf.min())/(surf.max()-surf.min())
                actuator_basis.append(surf.ravel())
                plt.clf()
                plt.imshow(surf)
                plt.title('generating actuator basis\n%0.2e,%0.2e'%(x,y))
                plt.pause(.1)

            self.actuator_basis = np.array(actuator_basis)
            np.save(cfn,self.actuator_basis)


        zfn = os.path.join(ccfg.simulator_cache_directory,'%s_zernike_basis.npy'%key)
        self.zernike = Zernike()
        try:
            self.zernike_basis = np.load(zfn)
            print('Loading cached zernike basis set...')
        except Exception as e:
            zernike_basis = []
            print('Building zernike basis set...')
            #zernike = Zernike()
            for z in range(self.n_zernike_terms):
                surf = self.zernike.get_j_surface(z,self.XX,self.YY)
                zernike_basis.append(surf.ravel())

            self.zernike_basis = np.array(zernike_basis)
            np.save(zfn,self.zernike_basis)

        #self.new_error_sigma = np.ones(self.n_zernike_terms)*100.0

        self.zernike_orders = np.zeros(self.n_zernike_terms)
        for j in range(self.n_zernike_terms):
            self.zernike_orders[j] = self.zernike.j2nm(j)[0]

        self.baseline_error_sigma = 1.0/(1.0+self.zernike_orders)*10.0
        self.baseline_error_sigma[3:6] = 150.0
        
        self.new_error_sigma = self.zernike_orders/10.0/100.0
        
        # don't generate any piston, tip, or tilt:
        self.baseline_error_sigma[:3] = 0.0
        self.new_error_sigma[:3] = 0.0
        
        self.error = self.get_error(self.baseline_error_sigma)

        self.exposure_us = 10000
        
        self.paused = False
        self.logging = False

    def log(self):
        outfn = os.path.join(ccfg.logging_directory,'mirror_%s.txt'%(now_string(True)))
        np.savetxt(outfn,self.command)

    def pause(self):
        self.paused = True

    def unpause(self):
        self.paused = False

    def set_logging(self,val):
        self.logging = val

    def flatten(self):
        self.command[:] = self.flat[:]
        #self.update()

    def restore_flat(self):
        self.flat[:] = self.flat0[:]
        
    def set_flat(self):
        self.flat[:] = self.get_command()[:]
        
    def get_command(self):
        return self.command

    def set_command(self,vec):
        self.command[:] = vec[:]
        #self.update()
        
    def set_actuator(self,index,value):
        self.command[index]=value
        self.update()
        
    def set_exposure(self,exposure_us):
        self.exposure_us = int(exposure_us)
        return
        
    def get_exposure(self):
        return self.exposure_us

    def noise(self,im):
        noiserms = np.random.randn(im.shape[0],im.shape[1])*np.sqrt(im)
        return im+noiserms

    def get_error(self,sigma):
        coefs = np.random.randn(self.n_zernike_terms)*sigma
        return np.reshape(np.dot(coefs,self.zernike_basis),(self.sy,self.sx))

    def defocus_animation(self):
        err = np.zeros(self.n_zernike_terms)
        for k in np.arange(0.0,100.0):
            err[4] = np.random.randn()
            im = np.reshape(np.dot(err,self.zernike_basis),(self.sy,self.sx))
            plt.clf()
            plt.imshow(im-im.min())
            plt.colorbar()
            plt.pause(.1)
    
    def plot_actuators(self):
        edge = self.XX.min()
        wid = self.XX.max()-edge
        plt.imshow(self.mask,extent=[edge,edge+wid,edge,edge+wid])
        plt.autoscale(False)
        plt.plot(ax,ay,'ks')
        plt.show()

    def update(self):
        mirror = np.reshape(np.dot(self.command,self.actuator_basis),(self.sy,self.sx))
        
        err = self.error + self.get_error(self.new_error_sigma)

        dx = np.diff(err,axis=1)
        dy = np.diff(err,axis=0)
        sy,sx = err.shape
        col = np.zeros((sy,1))
        row = np.zeros((1,sx))
        dx = np.hstack((col,dx))
        dy = np.vstack((row,dy))
        #err = err - dx - dy
        self.wavefront = mirror+err
        y_slope_vec = []
        x_slope_vec = []
        self.spots[:] = 0.0
        for idx,(x,y,x1,x2,y1,y2) in enumerate(zip(self.lenslet_boxes.x,
                                                   self.lenslet_boxes.y,
                                                   self.lenslet_boxes.x1,
                                                   self.lenslet_boxes.x2,
                                                   self.lenslet_boxes.y1,
                                                   self.lenslet_boxes.y2)):
            subwf = self.wavefront[y1:y2+1,x1:x2+1]
            yslope = np.mean(np.diff(subwf.mean(1)))
            dy = yslope*self.f/self.pixel_size_m
            ycentroid = y+dy
            ypx = int(round(y+dy))
            xslope = np.mean(np.diff(subwf.mean(0)))
            dx = xslope*self.f/self.pixel_size_m
            xcentroid = x+dx
            
            #if idx==20:
            #    continue
            
            self.spots = self.interpolate_dirac(xcentroid,ycentroid,self.spots)
            x_slope_vec.append(xslope)
            y_slope_vec.append(yslope)
            #QApplication.processEvents()
        self.spots = np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(self.spots))*self.disc))
        self.x_slopes = np.array(x_slope_vec)
        self.y_slopes = np.array(y_slope_vec)
        if self.logging:
            self.log()
        
        
    def get_image(self):
        self.update()
        self.frame_timer.tick()
        spots = (self.spots-self.spots.min())/(self.spots.max()-self.spots.min())*self.spots_range*float(self.exposure_us)/20000.+self.dc
        nspots = self.noise(spots)
        nspots = np.clip(nspots,0,4095)
        nspots = np.round(nspots).astype(np.int16)
        self.image = nspots
        return nspots
        
    def interpolate_dirac(self,x,y,frame):
        # take subpixel precision locations x and y and insert an interpolated
        # delta w/ amplitude 1 there
        #no interpolation case:
        #frame[int(round(y)),int(round(x))] = 1.0
        #return frame
        x1 = int(np.floor(x))
        x2 = x1+1
        y1 = int(np.floor(y))
        y2 = y1+1
        
        for yi in [y1,y2]:
            for xi in [x1,x2]:
                yweight = 1.0-(abs(yi-y))
                xweight = 1.0-(abs(xi-x))
                try:
                    frame[yi,xi] = yweight*xweight
                except Exception as e:
                    pass
        return frame
            
    def wavefront_to_spots(self):
        
        pass

    def show_zernikes(self):
        for k in range(self.n_zernike_terms):
            b = np.reshape(self.zernike_basis[k,:],(self.sy,self.sx))
            plt.clf()
            plt.imshow(b)
            plt.colorbar()
            plt.pause(.5)

    def close(self):
        print('Closing simulator.')
        
if __name__=='__main__':

    sim = Simulator()
    for k in range(100):
        sim.update()
        im = sim.get_image()
        print(k,im.mean())
        plt.cla()
        plt.imshow(im)
        plt.pause(.1)
