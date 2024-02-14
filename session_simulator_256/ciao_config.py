# A set of name-value pairs specifying local configuration of
# ciao installation. Where appropriate, each parameter's final
# characters represent units. 

##############################################################
# Use simulated camera and mirror (True) or simulate these (False):
simulate = True
# A unique, permanent identifier for the optical system
# associated with this installation of ciao:
system_id = 'simulator'

# An identifier for the deformable mirror, used to load
# the correct configuration files:
mirror_id = 'HSDM97-15-010'

# An identifier for the camera; can be 'ace', 'pylon', or
# 'simulator'
camera_id = 'pylon'

import os # 
image_width_px = 300
image_height_px = 300
bit_depth = 12

ciao_session = '.'
ciao_session = os.path.split(__file__)[0]

# define some directories for configuration files
reference_directory = ciao_session + '/etc/ref/'
dm_directory = ciao_session + '/etc/dm/'
poke_directory = ciao_session + '/etc/ctrl/'
logging_directory = ciao_session + '/log'
simulator_cache_directory = ciao_session + '/.simulator_cache'

# The reference coordinates need a starting point (see README.md), and these
# were recorded using calibration/record_reference_coordinates.py:
reference_coordinates_bootstrap_filename = reference_directory + 'reference_initial.txt'
reference_coordinates_filename = reference_directory + 'reference.txt'
reference_mask_filename = reference_directory + 'reference_mask.txt'
poke_filename = poke_directory + 'poke.txt'

# sensor settings:
reference_n_measurements = 10
lenslet_pitch_m = 500e-6
lenslet_focal_length_m = 20.0e-3
pixel_size_m = 40e-6

beam_diameter_m = 10e-3
interface_scale_factor = 0.75
wavelength_m = 840e-9
estimate_background = True
background_correction = 0
search_box_half_width = 5
spots_threshold = 100.0
sensor_update_rate = 1.0 # deprecated in current version
sensor_filter_lenslets = False
sensor_reconstruct_wavefront = True
sensor_remove_tip_tilt = False
centroiding_num_threads = 1
iterative_centroiding_step = 2
centroiding_iterations = 5
use_dark_subtraction = False

mirror_update_rate = 1.0 # deprecated in current version
mirror_flat_filename = dm_directory + 'flat.txt'
mirror_mask_filename = dm_directory + 'mirror_mask.txt'
mirror_command_max = 1.0
mirror_command_min = -1.0
mirror_settling_time_s = 0.001
mirror_n_actuators = 97

poke_command_max = 0.1
poke_command_min = -0.1
poke_n_command_steps = 5
poke_invert_on_demand = False

ctrl_dictionary_max_size = 10

loop_n_control_modes = 94
loop_gain = 0.3
loop_loss = 0.01
loop_update_rate = 500.0
loop_condition_ulim = 1000.
loop_condition_llim = 1.0

n_zernike_terms = 66
zernike_dioptric_equivalent = 1.5


# UI settings:
ui_width_px = 1200
ui_height_px = 800

plot_width_px = 400
plot_height_px = 100

caption_height_px = 15

error_plot_ylim = (-10e-9,1000e-9)
error_plot_ytick_interval = 100e-9
error_plot_print_func = lambda val: '%0.1f nm RMS'%(val*1e9)
error_plot_buffer_length = 100

zernike_plot_ylim = (-5e-6,5e-6)
zernike_plot_ytick_interval = 1e-6
zernike_plot_print_func = lambda val: '%0.1f um defocus'%(val*1e6)
zernike_plot_buffer_length = 200

plot_background_color = (255,255,255,255)

plot_line_color = (0,128,0,128)
plot_line_width = 2.0

plot_hline_color = (0,128,0,255)
plot_hline_width = 1.0

plot_xtick_color = (0,0,0,255)
plot_xtick_width = 1.0
plot_xtick_length = plot_height_px*.1
plot_xtick_interval = 50

plot_ytick_color = (0,0,0,255)
plot_ytick_width = 1.0

plot_buffer_length = 100


spots_image_downsampling = 1
search_box_color = (0,63,127,200)
search_box_thickness = 0.2
show_search_boxes = True
show_slope_lines = True

slope_line_thickness = 0.5
slope_line_color = (200,100,100,200)
slope_line_magnification = 100
spots_colormap = 'bone'
spots_contrast_limits = (0,2**bit_depth-1)
wavefront_colormap = 'jet'
wavefront_contrast_limits = (-1e-6,1e-6)
mirror_colormap = 'mirror'
mirror_contrast_limits = (-1,1)
zoom_width = 32
zoom_height = 32
single_spot_color = (255,63,63,255)
single_spot_thickness = 2.0

contrast_button_width = 40

ui_fps_fmt = '%0.2f Hz (UI)'
sensor_fps_fmt = '%0.2f Hz (Sensor)'
mirror_fps_fmt = '%0.2f Hz (Mirror)'
wavefront_error_fmt = '%0.1f nm RMS (Error)'
tip_fmt = '%0.4f mrad (Tip)'
tilt_fmt = '%0.4f mrad (Tilt)'
cond_fmt = '%0.2f (Condition)'

search_box_half_width_max = int(lenslet_pitch_m/pixel_size_m)//2

# Audio settings
audio_directory = ciao_session + '/etc/audio'
error_tones = [((0.0, 1e-08), 'A_sharp_5'),
               ((1e-08, 2e-08), 'G_sharp_5'),
               ((2e-08, 3e-08), 'G_5'),
               ((3e-08, 4e-08), 'F_5'),
               ((4e-08, 5e-08), 'D_sharp_5'),
               ((5e-08, 6e-08), 'D_5'),
               ((6e-08, 7e-08), 'C_5'),
               ((7e-08, 8e-08), 'A_sharp_4'),
               ((8e-08, 9e-08), 'G_sharp_4'),
               ((9e-08, 1e-07), 'G_4'),
               ((1e-07, 1.1e-07), 'F_4'),
               ((1.1e-07, 1.2e-07), 'D_sharp_4'),
               ((1.2e-07, 1.3e-07), 'D_4'),
               ((1.3e-07, 1.4e-07), 'C_4')]


rigorous_iteration = False
if rigorous_iteration:
    # First, calculate the PSF FWHM for the lenslets:
    import math
    lenslet_psf_fwhm_m = 1.22*wavelength_m*lenslet_focal_length_m/lenslet_pitch_m
    # Now see how many pixels this is:
    lenslet_psf_fwhm_px = lenslet_psf_fwhm_m/pixel_size_m 

    diffraction_limited_width_px = round(math.ceil(lenslet_psf_fwhm_px))
    if diffraction_limited_width_px%2==0:
        diffraction_limited_width_px+=1
    diffraction_limited_half_width_px = (diffraction_limited_width_px-1)//2

    iterative_centroiding_step = 1
    centroiding_iterations = int(round((search_box_half_width-diffraction_limited_half_width_px)//iterative_centroiding_step))

