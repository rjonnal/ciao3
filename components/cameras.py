import numpy as np
import glob
import ciao_config as ccfg
import os,sys
try:
    from pypylon import pylon
except Exception as e:
    print(e)

try:
    from ximea import xiapi
except Exception as e:
    print(e)

try:
    from pyueye import ueye
except Exception as e:
    print(e)
    
from ctypes import *
from ctypes.util import find_library
from time import time

def get_camera():
    if ccfg.camera_id.lower()=='pylon':
        return PylonCamera()
    elif ccfg.camera_id.lower()=='ace':
        return AOCameraAce()
    elif ccfg.camera_id.lower()=='ximea':
        return XimeaCamera()
    elif ccfg.camera_id.lower()=='ueye':
        return UeyeCamera()
    else:
        return SimulatedCamera()



    
class PylonCamera:

    def __init__(self,timeout=500):
        self.camera = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice())

        self.camera.Open()
        self.camera.PixelFormat.Value = 'Mono12'
        # enable all chunks
        self.camera.ChunkModeActive = True
        #self.camera.PixelFormat = "Mono12"
        
        for cf in self.camera.ChunkSelector.Symbolics:
            self.camera.ChunkSelector = cf
            self.camera.ChunkEnable = True

        self.timeout = timeout
        self.image = None

    def get_image(self):
        self.image = self.camera.GrabOne(self.timeout).Array.astype(np.int16)
        return self.image
    
    def close(self):
        return

    def set_exposure(self,exposure_us):
        self.camera.ExposureTime.Value = float(exposure_us)
        return
        
    def get_exposure(self):
        return int(self.camera.ExposureTime.Value)
    

class UeyeCamera:

    def __init__(self,timeout=500):
        hCam = ueye.HIDS(0)
        nRet = ueye.is_InitCamera(hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("UEYE error: is_InitCamera ERROR")

        cInfo = ueye.CAMINFO()
        sInfo = ueye.SENSORINFO()
        self.pcImageMemory = ueye.c_mem_p()
        MemID = ueye.int()
        rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()

        
        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
        nRet = ueye.is_GetCameraInfo(hCam, cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")

        # You can query additional information about the sensor type used in the camera
        nRet = ueye.is_GetSensorInfo(hCam, sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

        nRet = ueye.is_ResetToDefault( hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")

        m_nColorMode = ueye.IS_CM_MONO8
        self.nBitsPerPixel = ueye.INT(8)
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)

        # Can be used to set the size and position of an "area of interest"(AOI) within an
        nRet = ueye.is_AOI(hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        self.width = rectAOI.s32Width
        self.height = rectAOI.s32Height

        # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(hCam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory, MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(hCam, self.pcImageMemory, MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(hCam, m_nColorMode)

        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

    def get_image(self):
        
        array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
        self.image = np.reshape(array,(self.height.value, self.width.value, self.bytes_per_pixel))
        return self.image
    
    def close(self):
        return

    def set_exposure(self,exposure_us):
        #self.camera.ExposureTime.Value = float(exposure_us)
        return
        
    def get_exposure(self):
        return 1
        #return int(self.camera.ExposureTime.Value)

    
class XimeaCamera:

    def __init__(self,timeout=500):
        self.camera = xiapi.Camera()
        self.camera.open_device()
        try:
            self.set_exposure(ccfg.camera_exposure_us)
        except AttributeError as ae:
            print(ae)
            print("ciao_config.py is missing an entry for exposure time; please put 'camera_exposure_us = 1000' or similar into the ciao_config.py file for your session")
            sys.exit()
        self.camera.start_acquisition()
        self.img = xiapi.Image()
        self.image = None

    def get_image(self):
        self.camera.get_image(self.img)
        self.image = np.reshape(np.frombuffer(self.img.get_image_data_raw(),dtype=np.uint8),
                          (self.img.height,self.img.width)).astype(np.int16)
        return self.image
    
    def close(self):
        self.camera.stop_acquisition()
        self.camera.close_device()
        
    def set_exposure(self,exposure_us):
        print(exposure_us)
        self.camera.set_exposure(exposure_us)
        
    def get_exposure(self):
        return self.camera.get_exposure()

class SimulatedCamera:

    def __init__(self):
        self.image_list = sorted(glob.glob(os.path.join(ccfg.simulated_camera_image_directory,'*.npy')))
        self.n_images = len(self.image_list)
        self.index = 0
        #self.images = [np.load(fn) for fn in self.image_list]
        self.opacity = False
        self.sy,self.sx = np.load(self.image_list[0]).shape
        self.oy = int(round(np.random.rand()*self.sy//2+self.sy//4))
        self.ox = int(round(np.random.rand()*self.sx//2+self.sx//4))
        self.XX,self.YY = np.meshgrid(np.arange(self.sx),np.arange(self.sy))
        self.image = None

    def set_opacity(self,val):
        self.opacity = val

    def get_opacity(self):
        return self.opacity
            
    def get_image(self):
        im = np.load(self.image_list[self.index])
        #im = self.images[self.index]

        if self.opacity:
            im = self.opacify(im)
            self.oy = self.oy+np.random.randn()*.5
            self.ox = self.ox+np.random.randn()*.5

        self.index = (self.index + 1)%self.n_images
        self.image = im
        return self.image
        
    
    def opacify(self,im,sigma=50):
        xx,yy = self.XX-self.ox,self.YY-self.oy
        d = np.sqrt(xx**2+yy**2)
        #mask = np.exp((-d)/(2*sigma**2))
        #mask = mask/mask.max()
        #mask = 1-mask
        mask = np.ones(d.shape)
        mask[np.where(d<=sigma)] = 0.2
        out = np.round(im*mask).astype(np.int16)
        return out
        
    def close(self):
        return

    def set_exposure(self,exposure_us):
        return
        
    def get_exposure(self):
        return 10000
        
