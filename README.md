# CIAO: community-inspired adaptive optics
Python tools for controlling, simulating, and characterizing adaptive optics (AO) systems

# Setup and installation

## Prerequisites for minimal installation

These are the prerequisites for installing a version of the software which allows it to be run in simulation mode.

1. Install [emacs](https://www.gnu.org/software/emacs/), [Notepad++](https://notepad-plus-plus.org/download), or another editor.
2. Install [Git](https://git-scm.com/download/win)
3. Install [Anaconda for Python 2.7](https://www.anaconda.com/distribution/#download-section)
4. If you're using Windows, install the [Visual C++ compiler for Python 2.7](https://www.microsoft.com/en-us/download/details.aspx?id=44266). In Linux, gcc will be invoked instead; it's probably already installed on your system, but you can verify that it is with ```gcc --version```.
5. Clone this repository.

These prerequisites assume you are using the default hardware (Alpao mirror and a SHWS based on a Basler Ace USB3 camera).

## Additional prerequisites for UC Davis hardware loop implementation

1. Install Alpao drivers

## Anaconda prompt

Anaconda comes with a terminal emulator called "Anaconda prompt", which permits you to call any system commands (i.e. commands availalbe using the built-in Windows prompt) as well as Anaconda-specific commands (e.g., ```conda```) without having to modify your system path. In Windows it is the simplest way to interact with CIAO. It is assumed that you have basic familiarity with the command-line interface (CLI) of your operating system. Thus, specific syntax for copying files or folders or executing commands is omitted from this guide, in favor of platform-neutral descriptive instructions.

## Compiling C extensions for centroiding

Almost everything in CIAO could be written in Python using the Numpy library, with adequate performance for typical AO loops (20-30 Hz). However, in the interest of speeding things up, the costliest operation--centroiding the spots in the Shack-Hartmann image--has been moved down into a C-extension written using Cython. Depending on your OS and hardware, the precompiled binaries (```centroid.so``` and ```centroid.pyd```) may work out of the box. However, the safest thing to do is recompile the Cython code. To do this, navigate to the ```ciao/components/centroid``` folder and issue the following command:

    python setup.py build_ext --inplace
    
You may see some warnings (e.g. about deprecation of Numpy features), but shouldn't see any errors.

# Quick start

If you have succesfully completed the "Setup and installation" steps above, following this recipe should allow you to get a simulator up and running quickly.

1. Navigate into the ```ciao``` directory and make a copy of ```session_template``` and name it ```local_session_simulator_256```.
2. Navigate into ```local_session_simulator_256``` and issue ```python script_initialize.py```.
3. Create a mirror mask by issuing ```python script_make_mask.py 11 5.5 ./etc/dm/mirror_mask.txt```.
4. Issue ```python script_initialize.py``` again, and type 'Y' and press enter, to create an all-zero flat file.
5. Create a SHWS mask by issuing ```python script_make_mask.py 20 9.6 ./etc/ref/reference_mask.txt```.
6. Optional: issue ```python script_make_beeps.py``` to generate the WAV files for audio feedback.
7. Edit ```local_session_simulator_256/ciao_config.py```. Ensure that each of the following parameters are set as described below:

        simulate = True
        system_id = 'simulator'
        mirror_id = 'simulator'
        camera_id = 'simulator'
        image_width_px = 256
        image_height_px = 256
        lenslet_pitch_m = 500e-6
        lenslet_focal_length_m = 20.0e-3
        pixel_size_m = 40e-6
        beam_diameter_m = 10e-3
        search_box_half_width = 5
        iterative_centroiding_step = 2
        centroiding_iterations = 2
        mirror_n_actuators = 97

8. Issue ```python script_record_initial_reference_coordinates.py etc/ref/reference_initial.txt``` to create bootstrapping reference coordinates. Follow the instructions in the terminal and use the resulting plots to refine these coordinates.
9. Issue ```python ui_ciao.py```. The UI should appear.
10. Click **Record reference** a few times.
11. Click **Measure poke matrix** and wait for the poke matrix to be measured.
12. Click **Loop closed**.

# Slow start

## CIAO sessions and ```ciao_config.py```

CIAO depends on a notion of a *session*, which allows multiple configurations to be installed on the same computer. For instance, it may be useful to have a closed-loop session, a wavefront sensing session (e.g., for system alignment with a separate, calibrated sensor), and a simulation session, all on one computer. Each session requires a dedicated folder/directory, and a dedicated ```ciao_config.py``` file which specifies all of the relevant parameters of the session. CIAO has many tools, and there is a broad variety of use cases, some covered below; however, the two main ways to use CIAO are 1) as part of a GUI-based program for wavefront sensing or correction, where real-time feedback is critical; and 2) as part of a script to calibrate the system or make measurements. By convention, these programs are prefaced with ```ui_``` and ```script_```, respectively. These scripts must all be located in the session folder, alongside ```ciao_config.py```.

The advantage of this approach is that once things are configured correctly, the sessions can be run without modifications, even simultaneously (notwithstanding device driver conflicts). A disadvantage of this approach is that the top level programs scripts must add their filesystem locations to the Python path at runtime, because the rest of CIAO will all need access to the same ```ciao_config.py``` file, and attempt to import it. A related disadvantage is that users should be careful to avoid putting copies of ```ciao_config.py``` elsewhere, e.g. in the ```components``` directory or in any folder in the Python path, where it could in principle be loaded instead of the correct file for the session. Session directories should also never be added to the Python path, as this could result in the loading of incorrect versions of ```ciao_config.py```. (See **Design considerations** below for alternative approaches which were not pursued but may be preferable).

In short, every top level script must begin with the following two lines:

    import sys,os
    sys.path.append(os.path.split(__file__)[0])


## ```session_template``` folder

The default installation contains a folder called ```session_template``` which should can be copied to create a new session. This contains at least the following files:

    ciao_config.py
    script_initialize.py
    script_make_mask.py
    script_record_reference_coordinates.py
    ui_ciao.py
    
1. ```ciao_config.py``` is the session's configuration file, containing all of the parameters of the session's system, either real or simulated.
2. ```script_initialize.py``` is used to initialize a session--to create the required folders and assist in creating other calibration files.
3. ```script_make_mask.py``` is used to generate mask files for the SHWS and DM (see "Creating mask files" below).
4. ```script_record_initial_reference_coordinates.py``` is used to generate ballpark reference coordinates which can then be used to bootstrap precise reference coordinates. The problem is that we need search boxes before we can compute the centroids of reference beam spots, but we need centroids of the reference beam spots in order to position search boxes. This script attempts to guess the initial locations of search boxes based on the spots image, and permits a bit of interactive adjustment of those coordinates.
5. ```ui_ciao.py``` launches the closed-loop GUI.

## Local sessions

By default, any session whose name begins with ```local_session_``` will not be pushed into or pulled from the Git repository. This is a convenient way to guard against accidental collisions between local sessions and those stored in the repo.

## Creating a session

The quickest way to create a session is to use ```script_initialize.py```. After creating a copy of ```session_template```, rename it something descriptive starting with ```local_session_```, enter the directory and issue the following command:

    python script_initialize.py
    
This will create the following directory structure in your session directory.

    .
    |-- ciao_config.py
    |-- icons
    |   `-- ciao.png
    |-- etc
    |   |-- audio
    |   |-- ctrl
    |   |-- dm
    |   `-- ref
    |-- log
    |-- script_initialize.py
    |-- script_make_mask.py
    |-- script_record_initial_reference_coordinates.py
    `-- ui_ciao.py
    
    
It will also prompt you to create mask files, reference coordinates, etc.

## Creating mask files

The reference mask is a two-dimensional arrays of zeros and ones which specifies which of the Shack-Hartmann lenslets to use. The mirror mask specifies, similarly, the logical locations of active mirror actuators. For most deformable mirrors, there is no ambiguity about which locations should be ones and which should be zeros. For the ALPAO 97-actuator mirrors, for instance, the correct mask is generated by the command: 

    python script_make_mask.py 11 5.5 mirror_mask.txt
    
This creates the following mask, with 97 1's and 24 0's, which accurately describes the logical positions of the DM's actuators.

    0 0 0 1 1 1 1 1 0 0 0 
    0 0 1 1 1 1 1 1 1 0 0 
    0 1 1 1 1 1 1 1 1 1 0 
    1 1 1 1 1 1 1 1 1 1 1 
    1 1 1 1 1 1 1 1 1 1 1 
    1 1 1 1 1 1 1 1 1 1 1 
    1 1 1 1 1 1 1 1 1 1 1 
    1 1 1 1 1 1 1 1 1 1 1 
    0 1 1 1 1 1 1 1 1 1 0 
    0 0 1 1 1 1 1 1 1 0 0 
    0 0 0 1 1 1 1 1 0 0 0
    
However, without knowing that the DM has 97 actuators, this is an ill-defined problem. The following command:

    python script_make_mask.py 11 5.0 mirror_mask.txt
    
generates the following mask, which has a diameter of 11 actuators but the incorrect number of total actuators:

    0 0 0 0 0 1 0 0 0 0 0 
    0 0 1 1 1 1 1 1 1 0 0 
    0 1 1 1 1 1 1 1 1 1 0 
    0 1 1 1 1 1 1 1 1 1 0 
    0 1 1 1 1 1 1 1 1 1 0 
    1 1 1 1 1 1 1 1 1 1 1 
    0 1 1 1 1 1 1 1 1 1 0 
    0 1 1 1 1 1 1 1 1 1 0 
    0 1 1 1 1 1 1 1 1 1 0 
    0 0 1 1 1 1 1 1 1 0 0 
    0 0 0 0 0 1 0 0 0 0 0
    
The latter problem arises when defining a mask for the locations of active SHWS lenslets. In most cases, there will be some ambiguities about which lenslets should be used and which shouldn't--a problem without an obvious *a priori* solution. In this case, some experimentation may be called for.

Masks for the mirror and Shack-Hartmann lenslet array must be created, and copied into the ```etc/dm``` and ```etc/ref``` directories. This step is covered above in **Creating a session**.

## Creating a reference coordinate file

The reference coordinates are the (x,y) locations on the sensor, in pixel units, where spots are expected to fall when a planar wavefront is incident on the sensor. These are the coordinates with which the boundaries of the search boxes are defined, and the coordinates with which the local slope of the wavefront is computed, used to drive the mirror in closed loop or to reconstruct the wavefront and measure wavefront error.

The coordinates are stored in a file named, e.g. ```reference_coordinates.txt``` in the ```etc/ref/``` directory. This should be a comma-delimited plain text file, with N rows and two items per row, where N is the number of active lenslets (see **masks** above) and the two items are x and y coordinates, respectively.

Several approaches have been used to generate these coordinates, but a common approach is to shine a collimated beam on the sensor and record the positions (centers of mass) of the resulting spots. There is a bit of a catch-22 in this approach, however, since the definition of search boxes and calculation of centers of mass require coordinates to get started. A script, ```calibration/record_reference_coordinates.py``` is included to generate in initial set of coordinates, which can be used to bootstrap a more accurate set. It works by using the geometric centers of the lenslets to generate a fake spots image, and then cross-correlating it with a real image from the sensor. The process proceeds as follows:

1. Run ```python record_reference_coordinates.py N temp.txt```, where ```N``` is the number of sensor images to average.

2. Move ```temp.txt``` into the ```etc/ref``` directory and define ```reference_coordinates_filename``` in ```config.py``` accordingly.

3. Run CIAO and verify that the spots are roughly centered in the search boxes.

4. Click **```Record reference```**. This may need to be done more than once, because background noise in a search box that's not centered at the spot's center of mass causes a bias toward the reference coordinates, i.e. an underestimate of error. The residual wavefront error RMS may be used to verify that the coordinates have been recorded correctly, since apparent error due to shot noise will eventually reach a stable minimum. Typically this value should be $\leq 10\;nm$.

## Design principles

0. **Balance exploratory/educational goals with real-time goals**. This software is intended to be neither the highest-performance AO software nor the most [literate](https://en.wikipedia.org/wiki/Literate_programming), but to balance both goals. This is hard to achieve, and requires judgement calls, but some examples:

    a. We want the mirror object to be useful in real-time operation, running in its own thread, but also to be instantiable in [REPL](https://en.wikipedia.org/wiki/Read-eval-print-loop). Therefore, even though it may be faster and simpler to subclass a ```threading.Thread``` or ```QThread```, instead create threads and move the objects to threads as needed only in those classes used in the real-time case, such as ```ciao.Loop```. The same goes for mutexes; if employed, lock and unlock them from the real-time class, and don't employ them in classes meant to be used in both contexts.
    
    b. Qt signals and slots are used for event-driven programming, necessary for real-time operation. As with threads, above, instead of subclassing Qt signals and slots, use Qt decorators. This way, the functions can be called in REPL without trouble, but can also call each other via events in the real-time case.

1. **Use open source tools whenever possible**. The only exceptions should be closed-source drivers for AO components such as cameras and deformable mirrors.

2. **Use human-readable configuration, initialization, and log files.** Whenever possible, use plain text files, along with delimiters. Configuration and initialization files should be written in Python whenever possible.

3. **Avoid overspecification.** Specify parameters of the system in only one place. For example, since the number of lenslets is specified by the SHWS mask, don't specify it elsewhere. The size of the poke matrix, for instance, is implied by the SHWS and mirror masks, and needn't be specified anywhere else.

4. **Variable naming.** Class names should be one word with a capitalized first letter; if multiple words are necessary, use CamelCase. Variable names should be descriptive, with underscores between words. In order to maximize the exploratory and educational value of the code, when a variable has a unit, please append it to the end of the variable name, e.g. ```wavelength_m = 800e-9```.

## Camera-specific installation instructions

### Ximea cameras

1. Install the [latest Ximea API](https://www.ximea.com/support/wiki/apis/XIMEA_Windows_Software_Package). As of January, 2020, this was "V4.19.14 Beta", and this is the earliest version to contain the Python API.

2. During installation, make sure to check or select the Python API ("xiApiPython"), as it may be unchecked by default.

3. After installation, copy the ```API/Python/v2/ximea``` directory into your Python [site-packages](https://stackoverflow.com/questions/122327/how-do-i-find-the-location-of-my-python-site-packages-directory) directory. (This does not seem like the best way to accomplish this; I think ```c:/XIMEA/API/Python/v2/ximea``` should be added to the system's ```PYTHONPATH``` environment variable, but I haven't tested this approach.)

### Basler cameras (GigE and USB only)

1. Install [Basler Pylon 5.2](https://www.baslerweb.com/en/sales-support/downloads/software-downloads/pylon-5-2-0-windows/)

2. Install [pypylon](https://github.com/basler/pypylon/releases/download/1.4.0/pypylon-1.4.0-cp27-cp27m-win_amd64.whl). First download, then use 'pip install pypylon...amd64.whl'.


# Design considerations

## Installing mulitple instances of CIAO

One probably common use case for CIAO is having multiple "installations" or "versions" of the software to, for instance, be able to run a simulator with a low resolution, computationally fast beam, and then a real hardware-based loop or wavefront sensor. There are many ways to do this, some of which are listed below.

1. **Virtual environments**. These can be created using popular scientific Python distributions such as Anaconda or Enthought Python Distribution, or through the use of the ```virtualenv``` package. Advantages: the most *Pythonic* solution; enables multiple CIAOs; prevents other problems such as conflicts with pre-existing Python installations used for other purposes on the same computer. Disadvantages: requires moderate expertise on the topic of virtual environments; could cause major issues for novice developers.

2. **Configure once**. In this approach, the config file would be loaded just once and, because imported modules are objects, passed as a required parameter for instantiating any subsequent CIAO objects. Advantages: allows easy reconfiguration at runtime and, thus, a simple way to programatically explore parameter space; this is the approach [advocated by Python Software Foundation Fellow, Alex Martelli](https://stackoverflow.com/questions/2348927/python-single-configuration-file/2348941); avoids potential collisions of conflicting ```ciao_config.py``` files--a problem that could be extremely difficult to debug. Disadvantages: requires the same object to be propagated through the entire instantiation hierarchy, leading to ugly/mystifying signatures for every class's constructor.

3. **Multiple copies of config file**. Here we would have one file called ```ciao_config.py``` in the Python path, and others called, for instance ```ciao_config_simulation_version.py``` and ```ciao_config_closed_loop_version.py```, and one of the latter would be copied over the former whenever necessary. Advantages: the code base is cleanest, as it doesn't have to know anything other than to load ```ciao_config.py``` at runtime. Disadvantages: it's easy to make a mistake and delete a config file; requires a lot of bookkeeping.

4. **Global variable**. Like **Configure once** but instead of passing the config object around, access it as a global variable. Advantages: avoids ugly constructor signatures; simple. Disadvantages: confusing code because of the config variable's broad scope.

5. **Sessions**. Create folders with all top-level CIAO scripts and a single ```ciao_config.py``` in each. Advantages: once it's set up correctly you can forget about it; obviates confusing code in most of CIAO code base. Disadvantages: requires all scripts to live in the same directory as the config file; requires every script to modify ```sys.path``` at the very top, adding the session directory so that the only ```ciao_config.py``` file visible to any CIAO object is the session's. This is the approach I've selected, because it prioritizes ease of use, at the slight expense of transparency and flexibility.



# Topics for conversation

1. Other than condition number, what algorithmic or numerical tests can be employed to predict the performance of a given poke/control matrix?

2. What rules of thumb should be employed when deciding whether a spot should be used? Put differently, how should one choose ```rad``` when generating a SHWS mask, as described above?

