#!/usr/bin/env python
# encoding: utf-8
"""
   aruco DETECTION and computation

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

# !/usr/bin/env python
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

IMATE_TOPIC = "/camera/image_raw"
CALIB_FILE = 'camera_calib.json'  # default camera calibration file

class ArucoMarkerListener:
    def __init__(self):
        # Initialize ROS Node
        rospy.init_node('aruco_marker_listener', anonymous=True)

        # Create a ROS image subscriber to listen to image topic
        self.image_sub = rospy.Subscriber(IMATE_TOPIC, Image, self.image_callback)

        # Publisher for publishing the homogeneous transformation matrix with timestamp
        self.matrix_pub = rospy.Publisher("/aruco_transformation_matrix", Float64MultiArray, queue_size=10)

        # Initialize CvBridge to convert ROS images to OpenCV images
        self.bridge = CvBridge()

        # set the ARUCO dict
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # open calif file
        try:
            with open(CALIB_FILE) as file:
                data = json.load(file)
        except:
            print('Camera Calibration File not valid')
            exit()

        self.camera_matrix = np.array(data['camera_matrix'])
        self.dist_coeffs = np.array(data['distortion_coefficients'])

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV Image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert ROS image to OpenCV image: {e}")
            return

        show = True

        # Detect ARUCO markers
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if show:
            cv2.imshow('aruco_detect', gray_image)
            cv2.waitKey(100)
        # detect markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_image, self.aruco_dict)
        if len(corners) > 0:
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners,
                                                                           aruco_size,
                                                                           self.camera_matrix,
                                                                           self.dist_coeffs)

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
                    dispimage = cv2.drawFrameAxes(dispimage, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i],
                                                  length=0.2,
                                                  thickness=2)

            cv2.imshow('aruco_detect', dispimage)
            cv2.waitKey(100)

        if len(corners) > 0:
            # compute homogeneous matrices and publish
            for i in range(len(rvecs)):
                translation = Vector(tvecs[i][0])
                rotation, _ = cv2.Rodrigues(rvecs[i][0])
                rotation = RotationMatrix(rotation)
                Trel = HomogeneousMatrix(translation,
                                         rotation)
                self.publish_matrix_with_timestamp(Trel.toarray(), ids[i], msg.stamp)
            return
        print('NO ARUCO COULD BE EXTRACTED')
        rospy.loginfo(f"NO ARUCO COULD BE EXTRACTED.")

    def publish_matrix_with_timestamp(self, T, aruco_id, timestamp):
        """
        To simplify everythin, the last row of the Homogeneous matrix
        is employed to publish the ARUCO id
        """
        rospy.loginfo(f"Homogeneous Transformation Matrix:\n{T}")
        rospy.loginfo(f"FOUND ARUCO id", aruco_id)
        T_flatten = T.flatten().tolist()  # Flatten the 4x4 matrix to a 1D array
        T_flatten[-1] = float(aruco_id)
        # Create a message to publish the matrix
        matrix_msg = Float64MultiArray()
        matrix_msg.data = T_flatten

        # Set the header timestamp to the current time
        # matrix_msg.header.stamp = rospy.Time.now()
        matrix_msg.header.stamp = timestamp

        # Set frame_id (optional, for clarity)
        matrix_msg.header.frame_id = "camera_link"

        # Publish the matrix message
        self.matrix_pub.publish(matrix_msg)



if __name__ == '__main__':
    try:
        aruco_listener = ArucoMarkerListener()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()


