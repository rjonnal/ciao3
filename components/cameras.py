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
    try:
        _is_SetExposureTime = ueye._bind("is_SetExposureTime",
                                         [ueye.ctypes.c_uint, ueye.ctypes.c_double,
                                          ueye.ctypes.POINTER(ueye.ctypes.c_double)], ueye.ctypes.c_int)
        IS_GET_EXPOSURE_TIME = 0x8000

        @staticmethod
        def is_SetExposureTime(hCam, EXP, newEXP):
            """
            Description

            The function is_SetExposureTime() sets the with EXP indicated exposure time in ms. Since this
            is adjustable only in multiples of the time, a line needs, the actually used time can deviate from
            the desired value.

            The actual duration adjusted after the call of this function is readout with the parameter newEXP.
            By changing the window size or the readout timing (pixel clock) the exposure time set before is changed also.
            Therefore is_SetExposureTime() must be called again thereafter.

            Exposure-time interacting functions:
                - is_SetImageSize()
                - is_SetPixelClock()
                - is_SetFrameRate() (only if the new image time will be shorter than the exposure time)

            Which minimum and maximum values are possible and the dependence of the individual
            sensors is explained in detail in the description to the uEye timing.

            Depending on the time of the change of the exposure time this affects only with the recording of
            the next image.

            :param hCam: c_uint (aka c-type: HIDS)
            :param EXP: c_double (aka c-type: DOUBLE) - New desired exposure-time.
            :param newEXP: c_double (aka c-type: double *) - Actual exposure time.
            :returns: IS_SUCCESS, IS_NO_SUCCESS

            Notes for EXP values:

            - IS_GET_EXPOSURE_TIME Returns the actual exposure-time through parameter newEXP.
            - If EXP = 0.0 is passed, an exposure time of (1/frame rate) is used.
            - IS_GET_DEFAULT_EXPOSURE Returns the default exposure time newEXP Actual exposure time
            - IS_SET_ENABLE_AUTO_SHUTTER : activates the AutoExposure functionality.
              Setting a value will deactivate the functionality.
              (see also 4.86 is_SetAutoParameter).
            """
            _hCam = ueye._value_cast(hCam, ueye.ctypes.c_uint)
            _EXP = ueye._value_cast(EXP, ueye.ctypes.c_double)
            ret = UeyeCamera._is_SetExposureTime(_hCam, _EXP, ueye.ctypes.byref(newEXP) if newEXP is not None else None)
            return ret
    except NameError as ne:
        print(ne)
        
    def __init__(self, selector=''):
        self.hCam = ueye.HIDS(0)  # 0: first available camera;  1-254: The camera with the specified camera ID
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.INT()
        self.rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.nBitsPerPixel = ueye.INT()
        self.m_nColorMode = ueye.INT()
        self.bytes_per_pixel = 0
        self.width = ueye.INT()
        self.height = ueye.INT()
        self.size = (-1, -1)
        self.ok = False
        self.error_str = ''
        self.last_frame = None
        self.connect()

    def _error(self, err_str):
        self.error_str = err_str
        return

    def connect(self):
        self.error_str = ''

        # Starts the driver and establishes the connection to the camera
        rc = ueye.is_InitCamera(self.hCam, None)
        if rc != ueye.IS_SUCCESS:
            return self._error("is_InitCamera ERROR")

        # Reads out the data hard-coded in the non-volatile camera memory
        # and writes it to the data structure that cInfo points to
        rc = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if rc != ueye.IS_SUCCESS:
            return self._error("is_GetCameraInfo ERROR")

        # You can query additional information about the sensor type used in the camera
        rc = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if rc != ueye.IS_SUCCESS:
            return self._error("is_GetSensorInfo ERROR")

        rc = ueye.is_ResetToDefault(self.hCam)
        if rc != ueye.IS_SUCCESS:
            return self._error("is_ResetToDefault ERROR")

        # Set display mode to DIB
        rc = ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB)

        # Set the right color mode
        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(self.hCam, self.nBitsPerPixel, self.m_nColorMode)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_BAYER: ", )

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            self.nBitsPerPixel = ueye.INT(32)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_CBYCRY: ", )

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )

        else:
            # for monochrome camera models use Y8 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("Color mode: not identified")

        print("\tm_nColorMode: \t\t", self.m_nColorMode)
        print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
        print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
        print()

        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        rc = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        if rc != ueye.IS_SUCCESS:
            return self._error("is_AOI ERROR")

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height
        self.size = (self.width.value, self.height.value)

        # Prints out some information about the camera and the sensor
        print("Camera model:\t\t", self.sInfo.strSensorName.decode('utf-8'))
        print("Camera serial no.:\t", self.cInfo.SerNo.decode('utf-8'))
        print("Camera image size:\t", str(self.size))
        print()

        # Allocates an image memory for an image having its dimensions defined by width and height
        # and its color depth defined by nBitsPerPixel
        rc = ueye.is_AllocImageMem(self.hCam, self.width, self.height,
                                   self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if rc != ueye.IS_SUCCESS:
            return self._error("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            rc = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if rc != ueye.IS_SUCCESS:
                return self._error("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                rc = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)

        # Activates the camera's live video mode (free run mode)
        rc = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
        if rc != ueye.IS_SUCCESS:
            return self._error("is_CaptureVideo ERROR")

        # Enables the queue mode for existing image memory sequences
        rc = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID,
                                     self.width, self.height, self.nBitsPerPixel, self.pitch)
        if rc != ueye.IS_SUCCESS:
            return self._error("is_InquireImageMem ERROR")
        else:
            print("IDS camera: connection ok")
            self.ok = True

    def get_image(self):
        if not self.ok:
            return None
        # In order to display the image in an OpenCV window we need to...
        # ...extract the data of our image memory
        array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
        print(array.shape)
        print(self.height.value)
        print(self.width.value)
        print(self.bytes_per_pixel)
        print(float(len(array)/float(self.height.value)/float(self.width.value)))
        # ...reshape it in an numpy array...
        frame = np.reshape(array, (self.height.value, self.width.value))
        frame = frame.astype(int)
        # ...resize the image by a half
        # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        self.last_frame = frame
        return frame

    def close(self):
        # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
        ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)

        # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
        ueye.is_ExitCamera(self.hCam)

    def set_exposure(self, level_us):
        """
        :param level_us: exposure level in micro-seconds, or zero for auto exposure
        
        note that you can never exceed 1000000/fps, but it is possible to change the fps
        """
        p1 = ueye.DOUBLE()
        if level_us == 0:
            rc = UeyeCamera._is_SetExposureTime(self.hCam, ueye.IS_SET_ENABLE_AUTO_SHUTTER, p1)
            print(f'set_camera_exposure: set to auto')
        else:
            ms = ueye.DOUBLE(level_us / 1000)
            rc = UeyeCamera._is_SetExposureTime(self.hCam, ms, p1)
            print(f'set_camera_exposure: requested {ms.value}, got {p1.value}')

    def get_exposure(self, force_val=False):
        """
        returns the current exposure time in micro-seconds, or zero if auto exposure is on

        :param force_val: if True, will return level of exposure even if auto exposure is on
        """
        p1 = ueye.DOUBLE()
        p2 = ueye.DOUBLE()
        # we dump both auto-gain and auto exposure states:
        rc = ueye.is_SetAutoParameter(self.hCam, ueye.IS_GET_ENABLE_AUTO_GAIN, p1, p2)
        print(f'IS_GET_ENABLE_AUTO_GAIN={p1.value == 1}')
        rc = ueye.is_SetAutoParameter(self.hCam, ueye.IS_GET_ENABLE_AUTO_SHUTTER, p1, p2)
        print(f'IS_GET_ENABLE_AUTO_SHUTTER={p1.value == 1}')
        if (not force_val) and p1.value == 1:
            return 0  # auto exposure
        rc = UeyeCamera._is_SetExposureTime(self.hCam, UeyeCamera.IS_GET_EXPOSURE_TIME, p1)
        print(f'IS_GET_EXPOSURE_TIME={p1.value}')
        return int(round(p1.value * 1000))


    
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
        
