# README

**ARUCO_ROS_OBSERVER** a Python toolbox to create and observe ARUCO fiducial markers and 
publish them in ROS. The Camera-ARUCO transform is published

Created by [Arturo Gil](http://arvc.umh.es/personal/arturo/index.php?lang=en&vista=normal&dest=inicio&idp=arturo&type=per&ficha=on): arturo.gil@umh.es. [Miguel Hernández University, Spain.](http://www.umh.es)

## Create ARUCO markers
A set of ARUCOs are already found at the aruco_markers directory
Rund aruco_creation.py

## Detect ARUCO markers
Use aruco_detect.py to detect ARUCO markers on a set of images stored at any directory.


## Detect ARUCO markers and publish in ROS
Use aruco_rosnode.py
It is recommended to create a virtual environment
```
$ cd ARUCO_ROS_observer
$ ./venv/bin/python aruco_rosnode.py
```

# INSTALL

## DOWNLOAD THE CODE
Clone this repository:

```
$ git clone https://github.com/4rtur1t0/ARUCO_ROS_observer.git
$ virtualenv venv
$ ./venv/bin/pip install -r ARUCO_ROS_observer/requirements.txt
```

## INSTALL SYSTEM PACKAGES
Install some needed packages (if needed)
```
>> sudo apt install virtualenv
>> sudo apt install python3-dev
>> sudo apt install git
```


