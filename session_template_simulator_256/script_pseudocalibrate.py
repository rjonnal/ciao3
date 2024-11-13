import numpy as np
import time
from ciao3.components import centroid
import sys
import os
from matplotlib import pyplot as plt
import datetime
import cProfile
from ctypes import CDLL,c_void_p
from ciao3.components.search_boxes import SearchBoxes
import ciao_config as ccfg
import json
from ciao3.components.simulator import Simulator
from ciao3.components import cameras
from ciao3.components.mirrors import Mirror
from ciao3.components.ui import UI
from ciao3.components.frame_timer import FrameTimer
from ciao3.components.zernike import Reconstructor
from ciao3.components.tools import error_message, now_string, prepend, colortable, get_ram, get_process

class Sensor:

    def __init__(self,camera):
        self.image_width_px = ccfg.image_width_px
        self.image_height_px = ccfg.image_height_px
        self.dark_image = np.zeros((ccfg.image_height_px,ccfg.image_width_px),dtype=np.int16)
        self.dark_subtract = False
        self.n_dark = 10
        self.lenslet_pitch_m = ccfg.lenslet_pitch_m
        self.lenslet_focal_length_m = ccfg.lenslet_focal_length_m
        self.pixel_size_m = ccfg.pixel_size_m
        self.beam_diameter_m = ccfg.beam_diameter_m
        self.wavelength_m = ccfg.wavelength_m
        self.background_correction = ccfg.background_correction
        self.centroiding_iterations = ccfg.centroiding_iterations
        self.iterative_centroiding_step = ccfg.iterative_centroiding_step
        self.filter_lenslets = ccfg.sensor_filter_lenslets
        self.estimate_background = ccfg.estimate_background
        self.reconstruct_wavefront = ccfg.sensor_reconstruct_wavefront
        self.remove_tip_tilt = ccfg.sensor_remove_tip_tilt

        # calculate diffraction limited spot size on sensor, to determine
        # the centroiding window size
        lenslet_dlss = 1.22*self.wavelength_m*self.lenslet_focal_length_m/self.lenslet_pitch_m
        lenslet_dlss_px = lenslet_dlss/self.pixel_size_m
        # now we need to account for smearing of spots due to axial displacement of retinal layers
        extent = 500e-6 # retinal thickness
        smear = extent*6.75/16.67 # pupil diameter and focal length; use diameter in case beacon is at edge of field
        try:
            magnification = ccfg.retina_sensor_magnification
        except Exception as e:
            magnification = 1.0
        total_size = lenslet_dlss+smear*magnification
        total_size_px = total_size/self.pixel_size_m
        self.centroiding_half_width = int(np.floor(total_size_px/2.0))*2
        
        try:
            xy = np.loadtxt(ccfg.reference_coordinates_filename)
        except Exception as e:
            xy = np.loadtxt(ccfg.reference_coordinates_bootstrap_filename)
            print('Bootstrapping with %s'%ccfg.reference_coordinates_bootstrap_filename)
            
        self.search_boxes = SearchBoxes(xy[:,0],xy[:,1],ccfg.search_box_half_width)
        self.sensor_mask = np.loadtxt(ccfg.reference_mask_filename)
        
        self.x0 = np.zeros(self.search_boxes.x.shape)
        self.y0 = np.zeros(self.search_boxes.y.shape)
        
        self.x0[:] = self.search_boxes.x[:]
        self.y0[:] = self.search_boxes.y[:]
        
        self.n_lenslets = self.search_boxes.n
        n_lenslets = self.n_lenslets
        self.image = np.zeros((ccfg.image_height_px,ccfg.image_width_px))
        self.x_slopes = np.zeros(n_lenslets)
        self.y_slopes = np.zeros(n_lenslets)
        self.manual_x_offsets = np.zeros(n_lenslets)
        self.manual_y_offsets = np.zeros(n_lenslets)
        
        self.x_centroids = np.zeros(n_lenslets)
        self.y_centroids = np.zeros(n_lenslets)
        self.box_maxes = np.zeros(n_lenslets)
        self.box_mins = np.zeros(n_lenslets)
        self.box_means = np.zeros(n_lenslets)
        self.valid_centroids = np.zeros(n_lenslets,dtype=np.int16)

        try:
            self.fast_centroiding = ccfg.fast_centroiding
        except Exception as e:
            self.fast_centroiding = False
        self.box_backgrounds = np.zeros(n_lenslets)
        self.error = 0.0
        self.tip = 0.0
        self.tilt = 0.0
        self.zernikes = np.zeros(ccfg.n_zernike_terms)
        self.zernike_ui_offsets = np.zeros(ccfg.n_zernike_terms) # units are artibrary, from UI spinboxes
        self.n_zernike_terms = ccfg.n_zernike_terms
        
        self.wavefront = np.zeros(self.sensor_mask.shape)

        self.image_max = -1
        self.image_min = -1
        self.image_mean = -1
        
        self.cam = camera
        self.frame_timer = FrameTimer('Sensor',verbose=False)
        self.reconstructor = Reconstructor(self.search_boxes.x,
                                           self.search_boxes.y,self.sensor_mask)

        self.n_zernike_orders_corrected=self.reconstructor.N_orders
        self.centroiding_time = -1.0


        self.logging = False
        self.paused = False
        
        try:
            self.profile_update_method = ccfg.profile_sensor_update_method
        except:
            self.profile_update_method = False

        os.makedirs(ccfg.logging_directory,exist_ok=True)
        
    def update(self):
        if not self.paused:
            try:
                self.sense()
            except Exception as e:
                print('Sensor update exception:',e)
            if self.logging:
                self.log()
                
        self.frame_timer.tick()

    def reload_reference(self):
        xy = np.loadtxt(ccfg.reference_coordinates_filename)
        self.pause()
        self.search_boxes.move(xy[:,0],xy[:,1])
        self.unpause()

        
    def left(self):
        self.manual_x_offsets-=1
        self.set_reference_offsets()
        
    def right(self):
        self.manual_x_offsets+=1
        self.set_reference_offsets()
        
    def up(self):
        self.manual_y_offsets-=1
        self.set_reference_offsets()
        
    def down(self):
        self.manual_y_offsets+=1
        self.set_reference_offsets()
    
    def pause(self):
        print('sensor paused')
        self.paused = True

    def unpause(self):
        print('sensor unpaused')
        self.paused = False

    def get_average_background(self):
        return np.mean(self.box_backgrounds)

    def get_n_zernike_orders_corrected(self):
        return self.n_zernike_orders_corrected

    def set_n_zernike_orders_corrected(self,n):
        self.n_zernike_orders_corrected = n
        
    def set_dark_subtraction(self,val):
        self.dark_subtract = val

    def set_n_dark(self,val):
        self.n_dark = val

    def set_dark(self):
        self.pause()
        temp = np.zeros(self.dark_image.shape)
        for k in range(self.n_dark):
            temp = temp + self.cam.get_image()
        temp = np.round(temp/float(self.n_dark)).astype(np.int16)
        self.dark_image[...] = temp[...]
        self.unpause()

    def log(self):
        t_string = now_string(True)

        d = {}
        now = now_string()
        d['time'] = now
        d['x_slopes'] = list(self.x_slopes)
        d['y_slopes'] = list(self.y_slopes)
        d['x_centroids'] = list(self.x_centroids)
        d['y_centroids'] = list(self.y_centroids)

        d['search_box_x1'] = [int(k) for k in self.search_boxes.x1]
        d['search_box_x2'] = [int(k) for k in self.search_boxes.x2]
        d['search_box_y1'] = [int(k) for k in self.search_boxes.y1]
        d['search_box_y2'] = [int(k) for k in self.search_boxes.y2]
        
        d['ref_x'] = list(self.search_boxes.x)
        d['ref_y'] = list(self.search_boxes.y)
        d['error'] = self.error
        d['tip'] = self.tip
        d['tilt'] = self.tilt
        d['wavefront'] = [[x for x in row] for row in self.wavefront]
        d['zernikes'] = list(np.squeeze(self.zernikes))

        with open(os.path.join(ccfg.logging_directory,'sensor_%s.json'%now),'w') as fid:
            outstr = json.dumps(d)
            fid.write(outstr)

        np.savetxt(os.path.join(ccfg.logging_directory,'spots_%s.txt'%now),self.image)

        
    def set_background_correction(self,val):
        #sensor_mutex.lock()
        self.background_correction = val
        #sensor_mutex.unlock()

    def set_fast_centroiding(self,val):
        self.fast_centroiding = val

    def set_logging(self,val):
        self.logging = val


    def set_reference_offsets(self):
        newx = self.search_boxes.x0 + self.manual_x_offsets
        newy = self.search_boxes.y0 + self.manual_y_offsets

        for z in range(self.n_zernike_terms):
            offset = self.zernike_ui_offsets[z]
            if offset:
                newx = newx + self.reconstructor.slope_matrix[:self.n_lenslets,z]*offset*ccfg.zernike_dioptric_equivalent
                newy = newy + self.reconstructor.slope_matrix[self.n_lenslets:,z]*offset*ccfg.zernike_dioptric_equivalent
        self.pause()
        self.search_boxes.move(newx,newy)
        self.unpause()
            
    def set_defocus(self,val):
        self.zernike_ui_offsets[4] = val
        self.set_reference_offsets()
        
    def set_astig0(self,val):
        self.zernike_ui_offsets[3] = val
        self.set_reference_offsets()
        
    def set_astig1(self,val):
        self.zernike_ui_offsets[5] = val
        self.set_reference_offsets()

    def set_tip(self,val):
        self.zernike_ui_offsets[1] = val
        self.set_reference_offsets()
        
    def set_tilt(self,val):
        self.zernike_ui_offsets[2] = val
        self.set_reference_offsets()

        
    def set_defocus_old(self,val):
        self.pause()
        newx = self.search_boxes.x0 + self.reconstructor.defocus_dx*val*ccfg.zernike_dioptric_equivalent
        newy = self.search_boxes.y0 + self.reconstructor.defocus_dy*val*ccfg.zernike_dioptric_equivalent
        self.search_boxes.move(newx,newy)
        
        self.unpause()

    def set_astig_old(self,val):
        self.pause()
        
        newx = self.search_boxes.x + self.reconstructor.astig0_dx*val*ccfg.zernike_dioptric_equivalent
        
        newy = self.search_boxes.y + self.reconstructor.astig0_dy*val*ccfg.zernike_dioptric_equivalent
        self.search_boxes.move(newx,newy)
        
        self.unpause()
        
    def set_astig_old(self,val):
        self.pause()
        
        newx = self.search_boxes.x + self.reconstructor.astig1_dx*val*ccfg.zernike_dioptric_equivalent
        
        newy = self.search_boxes.y + self.reconstructor.astig1_dy*val*ccfg.zernike_dioptric_equivalent

        self.search_boxes.move(newx,newy)
        
        self.unpause()


    def aberration_reset(self):
        self.pause()
        self.search_boxes.move(self.x0,self.y0)
        self.unpause()
        
    def set_centroiding_half_width(self,val):
        self.centroiding_half_width = val
        
    def sense(self,debug=False):

            
        self.image = self.cam.get_image()
        
        if self.dark_subtract:
            self.image = self.image - self.dark_image

        self.image_min = self.image.min()
        self.image_mean = self.image.mean()
        self.image_max = self.image.max()
            
        
        t0 = time.time()
        if not self.fast_centroiding:
            if self.estimate_background:
                centroid.estimate_backgrounds(spots_image=self.image,
                                              sb_x_vec = self.search_boxes.x,
                                              sb_y_vec = self.search_boxes.y,
                                              sb_bg_vec = self.box_backgrounds,
                                              sb_half_width_p = self.search_boxes.half_width)
                self.box_backgrounds = self.box_backgrounds + self.background_correction
                
            centroid.compute_centroids(spots_image=self.image,
                                       sb_x_vec = self.search_boxes.x,
                                       sb_y_vec = self.search_boxes.y,
                                       sb_bg_vec = self.box_backgrounds,
                                       sb_half_width_p = self.search_boxes.half_width,
                                       iterations_p = self.centroiding_iterations,
                                       iteration_step_px_p = self.iterative_centroiding_step,
                                       x_out = self.x_centroids,
                                       y_out = self.y_centroids,
                                       mean_intensity = self.box_means,
                                       maximum_intensity = self.box_maxes,
                                       minimum_intensity = self.box_mins,
                                       num_threads_p = 1)
        else:
            centroid.fast_centroids(spots_image=self.image,
                                    sb_x_vec = self.search_boxes.x,
                                    sb_y_vec = self.search_boxes.y,
                                    sb_half_width_p = self.search_boxes.half_width,
                                    centroiding_half_width_p = self.centroiding_half_width,
                                    x_out = self.x_centroids,
                                    y_out = self.y_centroids,
                                    sb_max_vec = self.box_maxes,
                                    valid_vec = self.valid_centroids,
                                    verbose_p = 0,
                                    num_threads_p = 1)
        self.centroiding_time = time.time()-t0
        self.x_slopes = (self.x_centroids-self.search_boxes.x)*self.pixel_size_m/self.lenslet_focal_length_m
        self.y_slopes = (self.y_centroids-self.search_boxes.y)*self.pixel_size_m/self.lenslet_focal_length_m
        
        self.tilt = np.mean(self.x_slopes)
        self.tip = np.mean(self.y_slopes)
        
        if self.remove_tip_tilt:
            self.x_slopes-=self.tilt
            self.y_slopes-=self.tip

            
        if self.reconstruct_wavefront:
            self.zernikes,self.wavefront,self.error = self.reconstructor.get_wavefront(self.x_slopes,self.y_slopes)
            
            
            self.filter_slopes = self.n_zernike_orders_corrected<self.reconstructor.N_orders
            
            if self.filter_slopes:

                # Outline of approach: the basic idea is to filter the residual error
                # slopes by Zernike mode before multiplying by the mirror command
                # matrix.
                # 1. multiply the slopes by a wavefront reconstructor matrix
                #    to get Zernike coefficients; these are already output by
                #    the call to self.reconstructor.get_wavefront above
                # 2. zero the desired modes
                # 3. multiply the modes by the inverse of that matrix, which is stored
                #    in the Reconstructor object as reconstructor.slope_matrix

                # convert the order into a number of terms:
                n_terms = self.reconstructor.Z.nm2j(self.n_zernike_orders_corrected,self.n_zernike_orders_corrected)

                # get the slope matrix (inverse of zernike matrix, which maps slopes onto zernikes)
                slope_matrix = self.reconstructor.slope_matrix

                # create a filtered set of zernike terms
                # not sure if we should zero piston here
                z_filt = np.zeros(len(self.zernikes))
                z_filt[:n_terms+1] = self.zernikes[:n_terms+1]
                
                zero_piston = True
                if zero_piston:
                    z_filt[0] = 0.0
                
                # filter the slopes, and assign them to this sensor object:
                filtered_slopes = np.dot(slope_matrix,z_filt)
                self.x_slopes = filtered_slopes[:self.n_lenslets]
                self.y_slopes = filtered_slopes[self.n_lenslets:]
            
            


    def record_reference(self):

        print('recording reference')
        self.pause()
        xcent = []
        ycent = []
        for k in range(ccfg.reference_n_measurements):
            print('measurement %d of %d'%(k+1,ccfg.reference_n_measurements), end=' ')
            self.sense()
            print('...done')
            xcent.append(self.x_centroids)
            ycent.append(self.y_centroids)

        xcent = np.array(xcent)
        ycent = np.array(ycent)
        
        x_ref = xcent.mean(0)
        y_ref = ycent.mean(0)

        x_var = xcent.var(0)
        y_var = ycent.var(0)
        err = np.sqrt(np.mean([x_var,y_var]))
        print('reference coordinate error %0.3e px RMS'%err)
        
        self.search_boxes = SearchBoxes(x_ref,y_ref,self.search_boxes.half_width)
        refxy = np.array((x_ref,y_ref)).T

        # Record the new reference set to two locations, the
        # filename specified by reference_coordinates_filename
        # in ciao config, and also an archival filename to keep
        # track of the history.
        archive_fn = os.path.join(ccfg.reference_directory,prepend('reference.txt',now_string()))
        
        np.savetxt(archive_fn,refxy,fmt='%0.3f')
        np.savetxt(ccfg.reference_coordinates_filename,refxy,fmt='%0.3f')
        
        self.unpause()
        time.sleep(1)


    def pseudocalibrate(self):
        print('Pseudocalibrating...')
        self.pause()
        self.aberration_reset()
        
        self.sense()

        spots = self.image
        spots_model = np.zeros(spots.shape)
        sx,sy = spots.shape
        
        
        mask = self.sensor_mask
        mask_sy,mask_sx = mask.shape
        
        sigma = 5
        border = int(round(sigma*3))
        XX,YY = np.meshgrid(np.arange(sx),np.arange(sy))

        new_xref = []
        new_yref = []
        spots_model_fn = os.path.join(ccfg.reference_directory,'spots_model.npy')

        for y in range(mask_sy):
            for x in range(mask_sx):
                if mask[y,x]:
                    print(y,x)
                    y_m = y*self.lenslet_pitch_m
                    y_px = y_m/self.pixel_size_m+border
                    x_m = x*self.lenslet_pitch_m
                    x_px = x_m/self.pixel_size_m+border
                    new_xref.append(x_px)
                    new_yref.append(y_px)


        
        try:
            spots_model = np.load(spots_model_fn)
        except:
            for y in range(mask_sy):
                for x in range(mask_sx):
                    if mask[y,x]:
                        y_m = y*self.lenslet_pitch_m
                        y_px = y_m/self.pixel_size_m+border
                        x_m = x*self.lenslet_pitch_m
                        x_px = x_m/self.pixel_size_m+border
                        
                        xx = XX-x_px
                        yy = YY-y_px
                        temp = np.exp(-(xx**2+yy**2)/(2*sigma**2))
                        spots_model = spots_model+temp
            np.save(spots_model_fn,spots_model)

        new_xref = np.array(new_xref)
        new_yref = np.array(new_yref)


        sigma = 10.0
        xx = XX-sx/2.0
        yy = YY-sy/2.0
        g = np.exp(-(xx**2+yy**2)/(2*sigma**2))+1
        spots_spectrum = np.fft.fft2(spots)
        
        filt = 1.0/g
        plt.figure()
        plt.imshow(np.log10(np.abs(spots_spectrum)))
        plt.colorbar()
        plt.figure()
        plt.imshow(g)
        plt.colorbar()
        plt.figure()
        plt.imshow(filt)
        plt.colorbar()
        #spots = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.fft2(spots)*np.fft.fft2(filt))))
        plt.figure()
        plt.imshow(spots)
        plt.colorbar()
        spots = np.real(np.fft.ifft2(spots_spectrum*np.fft.fft2(filt)))
        plt.figure()
        plt.imshow(spots)
        plt.colorbar()
        plt.show()
        sys.exit()
        nxc = np.real(np.fft.ifft2(np.fft.fft2(spots)*np.conj(np.fft.fft2(spots_model))))
        peaky,peakx = np.unravel_index(np.argmax(nxc),nxc.shape)
        
        shifty = peaky
        shiftx = peakx
        if shifty>sy//2:
            shifty = shifty - sy
        if shiftx>sx//2:
            shiftx = shiftx - sx
            
        new_xref = new_xref + shiftx
        new_yref = new_yref + shifty


        self.search_boxes = SearchBoxes(new_xref,new_yref,self.search_boxes.half_width)

        # not sure if cross-correlation automatically minimizes tip and tilt; let's do it explicitly
        xerr = []
        yerr = []
        for k in range(ccfg.reference_n_measurements):
            print('measurement %d of %d'%(k+1,ccfg.reference_n_measurements), end=' ')
            self.sense()
            print('...done')
            xerr.append(self.x_centroids-self.search_boxes.x)
            yerr.append(self.y_centroids-self.search_boxes.y)


        xerr = np.array(xerr)
        yerr = np.array(yerr)
        xerr = np.mean(xerr,0)
        yerr = np.mean(yerr,0)


        # calculate scalar tip/tilt values to add these to new_xref and new_yref
        mxerr = np.mean(xerr)
        myerr = np.mean(yerr)

        # # as a sanity check, let's move through a range of offsets to see what the minimum tip and tilt are
        sbx = self.search_boxes.x
        sby = self.search_boxes.y
        step = 0.1
        grid = np.arange(-2,2,step)
        tip = np.zeros(len(grid))
        tilt = np.zeros(len(grid))

        for xidx,dx in enumerate(grid):
            self.search_boxes.move(sbx+dx,sby)
            self.sense()
            tilt[xidx] = self.zernikes[2]
                        
        for yidx,dy in enumerate(grid):
            self.search_boxes.move(sbx,sby+dy)
            self.sense()
            tip[yidx] = self.zernikes[1]


        mxerr2 = grid[np.argmin(np.abs(tilt))]
        myerr2 = grid[np.argmin(np.abs(tip))]

        try:
            # assert that the difference between the brute force zernike minimization and the bulk x and y errors
            # are smaller than the step size; this assures us that the bulk estimate is a good way to center the search
            # boxes
            assert np.abs(mxerr-mxerr2)<step
            assert np.abs(myerr-myerr2)<step
        except AssertionError:
            print('Warning: problem with tip tilt estimation in pseudocalibration. Bulk tip and tilt, calculated from average x and y dimension centroid error, does not match zernike minimization values. mxerr=%0.3f,mxerr2=%0.3f; myerr=%0.3f,myerr2=%0.3f.'%(mxerr,mxerr2,myerr,myerr2))
            
        new_xref = new_xref + mxerr
        new_yref = new_yref + myerr

        self.search_boxes = SearchBoxes(new_xref,new_yref,self.search_boxes.half_width)

        # Record the new reference set to two locations, the
        # filename specified by reference_coordinates_filename
        # in ciao config, and also an archival filename to keep
        # track of the history.
            
        ns = now_string()

        outfn = os.path.join(ccfg.reference_directory,prepend('pseudocalibration_report.pdf',ns))
        archive_fn = os.path.join(ccfg.reference_directory,prepend('reference.txt',ns))
        
        refxy = np.array((new_xref,new_yref)).T
        
        np.savetxt(archive_fn,refxy,fmt='%0.3f')
        np.savetxt(ccfg.reference_coordinates_filename,refxy,fmt='%0.3f')
        self.unpause()

        if True:
            plt.figure()
            plt.subplot(2,4,1)
            plt.imshow(spots,cmap='gray')
            plt.title('spots')
            plt.subplot(2,4,2)
            plt.imshow(spots_model,cmap='gray')
            plt.title('spots model')
            
            plt.subplot(2,4,3)
            plt.imshow(nxc,cmap='gray')
            plt.plot(peakx,peaky,'ys',markersize=3,markerfacecolor='none')
            plt.title('x corr')
            
            plt.subplot(2,4,4)
            plt.imshow(spots,cmap='gray')
            plt.plot(new_xref,new_yref,'rs',markersize=3,alpha=0.5,markerfacecolor='none')
            plt.title('references rough')


            plt.subplot(2,4,5)
            plt.bar(np.arange(len(xerr)),xerr)
            plt.subplot(2,4,6)
            plt.bar(np.arange(len(yerr)),yerr)
            plt.subplot(2,4,7)
            plt.plot(grid,np.abs(tilt))
            plt.xlabel('sb offset (x)')
            plt.ylabel('$Z_1^{1}$')
            plt.text(plt.gca().get_xlim()[0],plt.gca().get_ylim()[1],'remove_tip_tilt=%s'%ccfg.sensor_remove_tip_tilt,ha='left',va='top')
            plt.axvline(mxerr)
            plt.subplot(2,4,8)
            plt.plot(grid,np.abs(tip))
            plt.xlabel('sb offset (y)')
            plt.ylabel('$Z_1^{-1}$')
            plt.text(plt.gca().get_xlim()[0],plt.gca().get_ylim()[1],'remove_tip_tilt=%s'%ccfg.sensor_remove_tip_tilt,ha='left',va='top')
            plt.axvline(myerr)
            plt.pause(.1)
            plt.show()
            plt.savefig(outfn)
            plt.close()
            
            
        
        
if __name__=='__main__':
    if ccfg.simulate:
        sim = Simulator()
        sensor = Sensor(sim)
        mirror = sim
    else:
        cam = cameras.get_camera()
        mirror = Mirror()
        sensor = Sensor(cam)

sensor.pseudocalibrate()
