#!/usr/bin/env python
# encoding: utf-8
"""
   aruco DETECTION
   Just a demo that shows how to detect arucos and compute its pose.

   The estimated pose is
"""
import numpy as np
import cv2
import json
from artelib.vector import Vector
from artelib.rotationmatrix import RotationMatrix
from artelib.homogeneousmatrix import HomogeneousMatrix
import os
CALIB_FILE = 'camera_calib.json'  # default camera calibration file


def detect_arucos(image, show=True, aruco_size=0.15):
    # The dictionary should be defined as the one used in demos/aruco_markers/aruco_creation.py
    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        with open(CALIB_FILE) as file:
            data = json.load(file)
    except:
        print('Camera Calibration File not valid')
        exit()

    cameraMatrix = np.array(data['camera_matrix'])
    distCoeffs = np.array(data['distortion_coefficients'])

    gray_image = cv2.imread(image)
    # gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    if show:
        cv2.imshow('aruco_detect', gray_image)
        cv2.waitKey(100)
    # detect markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_image, aruco_dict)
    if len(corners) > 0:
        rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners,
                                                                       aruco_size,
                                                                       cameraMatrix,
                                                                       distCoeffs)

    if show:
        print('Press key!')
        cv2.imshow('aruco_detect', gray_image)
        cv2.waitKey(100)
        dispimage = cv2.aruco.drawDetectedMarkers(gray_image, corners, ids, borderColor=(0, 0, 255))
        # display corner order (Board file)
        for item in corners:
            for i in range(4):
                cv2.putText(dispimage, str(i), item[0, i].astype(int), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 0, 0), 1,
                            cv2.LINE_AA)
        if len(corners) > 0:
            # Draw axis for each marker
            for i in range(len(rvecs)):
                dispimage = cv2.drawFrameAxes(dispimage, cameraMatrix, distCoeffs, rvecs[i], tvecs[i],
                                              length=0.2,
                                              thickness=2)

        cv2.imshow('aruco_detect', dispimage)
        cv2.waitKey(1000)

    if len(corners) > 0:
        # compute homogeneous matrices
        tranformations = []
        for i in range(len(rvecs)):
            translation = Vector(tvecs[i][0])
            rotation, _ = cv2.Rodrigues(rvecs[i][0])
            rotation = RotationMatrix(rotation)
            Trel = HomogeneousMatrix(translation,
                                     rotation)
            tranformations.append(Trel)
        return ids, tranformations
    return None, None


if __name__ == "__main__":
    # reading directory files
    directory = './aruco_markers/real_images/'
    directory_list = sorted(os.listdir(directory))
    aruco_size = 0.21 # ARUCO size in meters
    show = True

    # process each image. For each image, present the Camera-ARUCO transformation and the Euclidean distance camera-ARUCO
    for image in directory_list:
        print('Processing image: ', directory+image)
        ids, transformations = detect_arucos(image=directory+image, show=show, aruco_size=aruco_size)
        if ids is None:
            continue

        for i in range(len(ids)):
            print('Found ARUCOS with id: ', ids[i])
            print('Camera-->ARUCO transformation:')
            transformations[i].print_nice()
            print('Distance (m): ', np.linalg.norm(transformations[i].pos()))
            print('Inverse:')
            it=transformations[i].inv()
            it.print_nice()


