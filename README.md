# CIAO: community-inspired adaptive optics
Python tools for controlling, simulating, and characterizing adaptive optics (AO) systems

## Quick start--running the simulator session

### Prerequisites for minimal installation

These are the prerequisites for installing a version of the software which allows it to be run in simulation mode.

1. Install [emacs](https://www.gnu.org/software/emacs/), [Notepad++](https://notepad-plus-plus.org/download), or another editor.
2. Install [Git](https://git-scm.com/download/)
3. Install [Anaconda for Python 3.X+](https://www.anaconda.com/download#downloads)
4. If you're using Windows, install the [Visual C++ compiler for Python](https://wiki.python.org/moin/WindowsCompilers). Make sure you get the right version for your Anaconda Python install. In Linux/Mac, gcc will be invoked instead; it's probably already installed on your system, but you can verify that it is with ```gcc --version```.
5. If necessary, create a directory where your Python libraries will reside, and add that directory to the environment variable `PYTHONPATH`. 
6. Clone this repository into the directory specified by `PYTHONPATH` by navigating to that directory in a terminal and typing: `git clone https://github.com/rjonnal/ciao3`. In our lab we use the locations `C:\code` (Windows) and `/home/USER/code` (Linux) as a rule.
7. If `git clone` doesn't work, you can download CIAO as a [zip file](https://github.com/rjonnal/ciao3/archive/refs/heads/main.zip) instead, but you should really try to get `git` to work.

### Anaconda prompt

Anaconda comes with a terminal emulator called "Anaconda prompt", which permits you to call any system commands (i.e. commands availalbe using the built-in Windows prompt) as well as Anaconda-specific commands (e.g., ```conda```) without having to modify your system path. In Windows it is the simplest way to interact with CIAO. It is assumed that you have basic familiarity with the command-line interface (CLI) of your operating system. Thus, specific syntax for copying files or folders or executing commands is omitted from this guide, in favor of platform-neutral descriptive instructions.

If you are using Windows, it is highly recommended that you install common Linux/Mac CLI tools using `conda install m2-base`. The Windows versions of these CLI tools are bizarre and inconvenient.

### Installation of required Python libraries

The instructions below contain minimal instructions for getting started with a CIAO simulation session. If you are new to Anaconda or Miniconda, please read [this introduction](https://docs.conda.io/projects/conda/en/23.3.x/user-guide/getting-started.html).

1. Create a `ciao` virtual environment in Anaconda. At the conda terminal, type: `conda create --name ciao3 python=3.12`. We are installing version 3.12 because it's the last version of Python tested against. CIAO will probably run on newer versions of Python too.

2. Activate the `ciao3` environment: `conda activate ciao3`. You will have to activate this environment any time you want to run CIAO or its components.

3. Verify Python version in the `ciao3` environment: `python --version`.

4. CIAO depends on the following conda packages:
    ```
	matplotlib
	numpy
	scipy
	psutil
	cython
	pandas
    ```
5. These can be installed with separate calls to `conda install PACKAGE`, or using the supplied environment YAML file: from `ciao3/` issue: `conda env create -f ciao3_environment.yml`. The latter will read the YAML file at from the `ciao3/` folder and install the required packages.

### Compiling the centroiding algorithm

As of now, the only real-time algorithm that is substantially slower in pure Python than compiled C is the background-estimation and center-of-mass calculation. Almost everything in CIAO could be written in Python using the Numpy library, with adequate performance for typical AO loops (20-30 Hz). However, in the interest of speeding things up, the costliest operation--centroiding the spots in the Shack-Hartmann image--has been moved down into a C-extension written using Cython[Cython](https://cython.readthedocs.io/en/latest/). 

Cython is a superset of Python with some additional features including, most importantly, static typing. After Cython programs are written, they are compiled into machine code. The compiled machine code programs have extensions `.so` (for 'shared object', in Linux) and `.pyd` (in Windows, similar to a DLL file). Both are dynamically linked libraries, i.e., libraries whose functions are called by other programs, as opposed to libraries whose functions are compiled into other programs. To the Python programmer, an `.so` or `.pyd` file behaves just like a `.py` file; you can import it, import from it, etc.

In a perfect world, installation of a C compiler such as gcc or Visual C++ compiler combined with installation of cython via conda (as described above) will just make Cython work correctly.

To compile the Cython code, navigate to the ```ciao3/components/centroid_cython``` folder and issue the following command: `python setup.py build_ext --inplace`.

You may see some warnings (e.g. about deprecation of Numpy features), but shouldn't see any errors. After that, copy the new `.so` or `.pyd` file into the `ciao3/components/` folder and rename it `centroid.so` or `centroid.pyd`.

The `bash` programs `rebuild.sh` and `cleanup.sh` can be used on Linux systems. Windows users can look at these simple scripts to see what they do. Just remember that on Windows the compiled linked library has the extension `.pyd`, and this must be renamed/moved to `ciao3/components/centroid.pyd`.

### Copying the template session folder and running the UI

Next, navigate to `ciao3/` and copy/rename the `session_template_simulator_256`, creating a local version `local_session_simulator_256`. In Linux: `cp -rvf session_template_simulator_256 local_session_simulator_256`.

Then, navigate to the local session folder: `cd local_session_simulator_256`.

Then, run the simulator UI: `python ui_ciao.py`.


## Sessions

### CIAO sessions and ```ciao_config.py```

CIAO depends on a notion of a *session*, which allows multiple configurations to be installed on the same computer. For instance, it may be useful to have a closed-loop session, a wavefront sensing session (e.g., for system alignment with a separate, calibrated sensor), and a simulation session, all on one computer. Each session requires a dedicated folder/directory, and a dedicated ```ciao_config.py``` file which specifies all of the relevant parameters of the session. CIAO has many tools, and there is a broad variety of use cases, some covered below; however, the three main ways to use CIAO are 1) as part of a GUI-based program for wavefront sensing or correction, where real-time feedback is critical; 2) as part of a script to calibrate the system or make measurements; 3) in a [Jupyter](https://jupyter.org/) notebook designed for instruction or documentation. By convention, these programs are prefaced with ```ui_```, ```script_```, and ```nb_``` respectively. These scripts must all be located in the session folder, alongside ```ciao_config.py```.

The advantage of this approach is that once things are configured correctly, the sessions can be run without modifications, even simultaneously (notwithstanding device driver conflicts). A disadvantage of this approach is that the top level programs scripts must add their filesystem locations to the Python path at runtime, because the rest of CIAO will all need access to the same ```ciao_config.py``` file, and attempt to import it. A related disadvantage is that users should be careful to avoid putting copies of ```ciao_config.py``` elsewhere, e.g. in the ```components``` directory or in any folder in the Python path, where they could in principle be loaded instead of the correct file for the session. Session directories should also never be added to the Python path environment variable `PYTHONPATH`, as this could result in the loading of incorrect versions of ```ciao_config.py```. (See **Design considerations** below for alternative approaches which were not pursued but may be preferable).

In short, every top level script must begin with the following two lines:

    import sys,os
    sys.path.append(os.path.split(__file__)[0])

### ```session_template``` folder

The default installation contains several folders called ```session_template_XXXX```, which can be copied to create new sessions. This template contains at least the following files:

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

### Local sessions

By default, any session whose name begins with ```local_session_``` will not be pushed into or pulled from the Git repository. This is a convenient way to guard against accidental collisions between local sessions and those stored in the repo.

## Quick start--making a session from scratch

If you have succesfully completed the **Quick start--running the simulator session** steps above, following this recipe should allow you to get a custom simulator up and running quickly.

1. Navigate into the ```ciao``` directory and make a copy of ```session_template``` and name it ```local_session_XXXX```.
2. Navigate into ```local_session_XXXX``` and issue ```python script_initialize.py```.
3. Create a mirror mask by issuing ```python script_make_mask.py 11 5.5 ./etc/dm/mirror_mask.txt```. These numbers represent the logical width of the mirror mask file and logical radius of the mirror. You may alter them as you wish.
4. Issue ```python script_initialize.py``` again, and type 'Y' and press enter, to create an all-zero flat file.
5. Create a SHWS mask by issuing ```python script_make_mask.py 20 9.6 ./etc/ref/reference_mask.txt```. These numbers represent the logical width of the reference mask file and the logical radius of the reference mask.
6. Optional: issue ```python script_make_beeps.py``` to generate the WAV files for audio feedback.
7. Edit ```local_session_XXXX/ciao_config.py```. Ensure that each of the following parameters are set correctly. Some example values are shown below:

	```
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
	```
8. Issue ```python script_record_initial_reference_coordinates.py etc/ref/reference_initial.txt``` to create bootstrapping reference coordinates. Follow the instructions in the terminal and use the resulting plots to refine these coordinates.
9. Issue ```python ui_ciao.py```. The UI should appear.
10. Click **Pseudocalibrate** a few times. The first time you do this it will take a few minutes. The slowest step is generating a simulated/theoretical spots image that is required for determining the ideal location of the reference coordinates.
11. Click **Measure poke matrix** and wait for the poke matrix to be measured.
12. Click **Loop closed**.

## Calibration using a planar wavefront [UNDER REVIEW]

If you have a planar reference wavefront incident on the SHWS and would like to use it to calibrate the reference coordinates, do the following:

1. Edit ```SESSION/ciao_config.py``` and check the value of ```reference_n_measurements```. The default value should be 10, but this may be increased if the SHWS SNR is low. This constant determines the number of spots images that will be centroided. The resulting centers of mass will be averaged together to generate the reference coordinates.
2. Issue ```python ui_ciao.py```. The UI should appear.
3. Place a model eye in the system such that the SHWS spots appear where they will when measuring an eye.
4. Adjust the exposure time in the UI such that at least half of the camera's dynamic range is used (e.g., 2048 for a 12-bit camera).
5. Click **Pseudocalibrate**. This may take a few minutes. This will generate reference coordinates and search boxes using the lenslet array's geometry, as specified in the session's ```ciao_config.py```. The relative coordinates are determined by the lenslet geometry, while the absolute coordinates are determined by cross-correlation of a simulated spots image with what is currently seen on the camera.
6. Remove or block the model eye, and direct the planar reference beam to the SHWS.
7. If necessary, adjust the exposure time again such that such that at least half of the camera's dynamic range is used (e.g., 2048 for a 12-bit camera).
8. If necessary, adjust tip and tilt of the reference beam to center them in the existing search boxes as well as possible. This is a good thing to do because the reference coordinates are most accurately recorded when the spots are centered, since the impact of background estimation error is minimized. **The planar beam's tip and tilt should not be adjusted using any optical elements shared by the planar reference beam and the model eye beam**, since it's critical at this stage not to affect the model eye (or real eye) spot locations.
9. Click **Record Reference**. This will alter the real time reference coordinates as well as the ```etc/ref/reference.txt``` file, such that subsequent runs of ```ui_ciao.py``` will use the newly recorded reference.

## Slow start

### Creating a session

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

### Creating mask files

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

### Creating a reference coordinate file

The reference coordinates are the (x,y) locations on the sensor, in pixel units, where spots are expected to fall when a planar wavefront is incident on the sensor. These are the coordinates with which the boundaries of the search boxes are defined, and the coordinates with which the local slope of the wavefront is computed, used to drive the mirror in closed loop or to reconstruct the wavefront and measure wavefront error.

The coordinates are stored in a file named, e.g. ```reference_coordinates.txt``` in the ```etc/ref/``` directory. This should be a comma-delimited plain text file, with N rows and two items per row, where N is the number of active lenslets (see **masks** above) and the two items are x and y coordinates, respectively.

Several approaches have been used to generate these coordinates, but a common approach is to shine a collimated beam on the sensor and record the positions (centers of mass) of the resulting spots. There is a bit of a catch-22 in this approach, however, since the definition of search boxes and calculation of centers of mass require coordinates to get started. A script, ```calibration/record_reference_coordinates.py``` is included to generate in initial set of coordinates, which can be used to bootstrap a more accurate set. It works by using the geometric centers of the lenslets to generate a fake spots image, and then cross-correlating it with a real image from the sensor. The process proceeds as follows:

1. Run ```python record_reference_coordinates.py N temp.txt```, where ```N``` is the number of sensor images to average.

2. Move ```temp.txt``` into the ```etc/ref``` directory and define ```reference_coordinates_filename``` in ```config.py``` accordingly.

3. Run CIAO and verify that the spots are roughly centered in the search boxes.

4. Click **```Record reference```**. This may need to be done more than once, because background noise in a search box that's not centered at the spot's center of mass causes a bias toward the reference coordinates, i.e. an underestimate of error. The residual wavefront error RMS may be used to verify that the coordinates have been recorded correctly, since apparent error due to shot noise will eventually reach a stable minimum. Typically this value should be $\leq 10\;nm$.

### Design principles

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

### IDS UEYE cameras

1. Install [UEYE](https://www.ids-imaging.us/downloads.html?gad_source=1&gclid=Cj0KCQiAlsy5BhDeARIsABRc6ZtTxhOhyqbUeWjh0gVq_kP4N5pw2W3b3Ahuyl-SKVLPt5IYdxAFoVkaAi_7EALw_wcB) for your operating system.
2. Install [pyueye](https://pypi.org/project/pyueye/) via pip: `pip install pyueye`. Make sure you're in the `ciao3` environment first, with `conda activate ciao3`.

## Design considerations

### Installing mulitple instances of CIAO

One probably common use case for CIAO is having multiple "installations" or "versions" of the software to, for instance, be able to run a simulator with a low resolution, computationally fast beam, and then a real hardware-based loop or wavefront sensor. There are many ways to do this, some of which are listed below.

1. **Virtual environments**. These can be created using popular scientific Python distributions such as Anaconda or Enthought Python Distribution, or through the use of the ```virtualenv``` package. Advantages: the most *Pythonic* solution; enables multiple CIAOs; prevents other problems such as conflicts with pre-existing Python installations used for other purposes on the same computer. Disadvantages: requires management of virtual environments; requires multiple installations of Python scientific stack (numpy, scipy, etc.).

2. **Configure once**. In this approach, the config file would be loaded just once and, because imported modules are objects, passed as a required parameter for instantiating any subsequent CIAO objects. Advantages: allows easy reconfiguration at runtime and, thus, a simple way to programatically explore parameter space; this is the approach [advocated by Python Software Foundation Fellow, Alex Martelli](https://stackoverflow.com/questions/2348927/python-single-configuration-file/2348941); avoids potential collisions of conflicting ```ciao_config.py``` files--a problem that could be extremely difficult to debug. Disadvantages: requires the same object to be propagated through the entire instantiation hierarchy, leading to ugly/mystifying signatures for every class's constructor.

3. **Multiple copies of config file**. Here we would have one file called ```ciao_config.py``` in the Python path, and others called, for instance ```ciao_config_simulation_version.py``` and ```ciao_config_closed_loop_version.py```, and one of the latter would be copied over the former whenever necessary. Advantages: the code base is cleanest, as it doesn't have to know anything other than to load ```ciao_config.py``` at runtime. Disadvantages: it's easy to make a mistake and delete a config file; requires a lot of bookkeeping.

4. **Global variable**. Like **Configure once** but instead of passing the config object around, access it as a global variable. Advantages: avoids ugly constructor signatures; simple. Disadvantages: confusing code because of the config variable's broad scope.

5. **Sessions**. Create folders with all top-level CIAO scripts and a single ```ciao_config.py``` in each. Advantages: once it's set up correctly you can forget about it; obviates confusing code in most of CIAO code base. Disadvantages: requires all scripts to live in the same directory as the config file; requires every script to modify ```sys.path``` at the very top, adding the session directory so that the only ```ciao_config.py``` file visible to any CIAO object is the session's. This is the approach I've selected, because it prioritizes ease of use, at the slight expense of transparency and flexibility.

## Topics for conversation

1. Other than condition number, what algorithmic or numerical tests can be employed to predict the performance of a given poke/control matrix?

2. What rules of thumb should be employed when deciding whether a spot should be used? Put differently, how should one choose ```rad``` when generating a SHWS mask, as described above?

3. What is the optimal procedure for calibrating using a planar reference beam, the `pseudocalibrate` function, and the `record reference` function?

4. What the heck is going on with tip and tilt, man?
