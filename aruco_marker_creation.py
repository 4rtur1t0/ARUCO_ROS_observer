#!/usr/bin/env python
# encoding: utf-8
"""
   ARUCO creation
   Creates a dictionary of ARUCO MARKERS.
   The same dictionary (i.e. cv2.aruco.DICT_6X6_250) should be used
   in the camera.detect_arucos() method.

   4x4 dictionaries are recommended for SLAM
"""
from cv2 import aruco
import cv2
import matplotlib.pyplot as plt


def create_arucos(image_name, n_arucos, show=False):
    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    plt.figure()

    top, bottom, left, right = 150, 150, 150, 150
    borderType = cv2.BORDER_CONSTANT
    value = [255, 255, 255]
    dst = None
    for i in range(n_arucos):
        img = aruco.generateImageMarker(aruco_dict, i, 700)
        img_border = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  borderType, dst, value)
        cv2.imwrite(image_name + str(i) + '.png', img_border)

        if show:
            cv2.imshow('borders', img_border)
            cv2.waitKey(500)


if __name__ == "__main__":
    ouput_directory = './aruco_markers/markers/4x4/'
    output_name = 'aruco'
    # number of output ARUCOs
    n_arucos = 50
    # show images
    show = False
    create_arucos(image_name=ouput_directory+output_name, n_arucos=n_arucos, show=show)


