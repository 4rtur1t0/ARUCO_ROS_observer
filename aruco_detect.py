#!/usr/bin/env python
# encoding: utf-8
"""
   aruco DETECTION
   Just a demo that shows how to detect arucos
   Specify a directory with real or simulated ARUCOs to detect
"""
import cv2
import os


def detect_arucos(image_name, show):
    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    print('Processing image: ', image_name)
    # gray_image = cv2.imread('aruco_markers/real_images/aruco_test.png')
    gray_image = cv2.imread(image_name)
    if show:
        cv2.imshow('aruco_detect', gray_image)
        print('press key')
        cv2.waitKey(0)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_image, aruco_dict)
    dispimage = cv2.aruco.drawDetectedMarkers(gray_image, corners, ids, borderColor=(0, 0, 255))

    # display corner order (Board file)
    for item in corners:
        for i in range(4):
            cv2.putText(dispimage, str(i), item[0, i].astype(int), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 0, 0), 1,
                       cv2.LINE_AA)
    if show:
        cv2.imshow('aruco_detect', dispimage)
        print('press key')
        cv2.waitKey(0)
    print('end')


if __name__ == "__main__":
    directory = './aruco_markers/real_images/'
    directory_list = sorted(os.listdir(directory))
    show = True

    for image in directory_list:
        detect_arucos(image_name=directory+image, show=show)


