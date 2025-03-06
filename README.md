# README

**ARUCO_ROS_OBSERVER** a Python toolbox to create and observe ARUCO fiducial markers and 
publish them in ROS

Created by [Arturo Gil](http://arvc.umh.es/personal/arturo/index.php?lang=en&vista=normal&dest=inicio&idp=arturo&type=per&ficha=on): arturo.gil@umh.es. [Miguel Hernández University, Spain.](http://www.umh.es)

## Create ARUCO markers
A set of ARUCOs are already found at the aruco_markers directory
Rund aruco_creation.py

## Detect ARUCO markers
Use aruco_detect.py



pyARTE is distributed under LGPL license.

![Screenshot](screenshot.png)


# INSTALL

## DOWNLOAD THE CODE
Clone this repository:

```
>> git clone https://github.com/4rtur1t0/pyARTE.git
```

## INSTALL SYSTEM PACKAGES
Install some needed packages  (python3-tk is needed in some distributions to plot with matplotlib)
```
>> sudo apt install virtualenv
>> sudo apt install python3-dev
>> sudo apt install git
>> sudo  apt-get install python3-tk
```

## CREATE A VIRTUAL ENVIRONMENT
We will be creating a virtual environment at your user's home/Simulations directory: 
```
>> cd
>> mkdir Simulations
>> cd Simulations
>> virtualenv venv
```

Next, install some needed python packages. We only require matplotlib, numpy and pynput:
```
>> cd /home/user/Simulations/venv/bin
>> ./pip3 install numpy matplotlib pynput pyzmq cbor opencv-contrib-python
```


## TEST
Open Coppelia Sim.
Open the irb140.ttt scene. This scene is found in pyARTE/scenes/irb140.ttt.
Open and execute the pyARTE/practicals/irb140_first_script.py
Open and execute the pyARTE/practicals/applications/irb140_palletizing_color.py

The python scripts should connect to Coppelia and start the simulation.


