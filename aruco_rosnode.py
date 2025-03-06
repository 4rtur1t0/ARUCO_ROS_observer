#!/usr/bin/env python
# encoding: utf-8
"""
   aruco DETECTION and computation

   Just a demo that shows how to detect arucos and compute its pose.

   The estimated pose is
"""
# !/usr/bin/env python
import json
from artelib.vector import Vector
from artelib.rotationmatrix import RotationMatrix
from artelib.homogeneousmatrix import HomogeneousMatrix
import os
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import Image
#from geometry_msgs.msg import TransformStamped
#from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import String
#from aruco_msgs.msg import MarkerArray
from cv_bridge import CvBridge

IMAGE_TOPIC = "/visible_image"
CALIB_FILE = 'camera_calib.json'  # default camera calibration file
ARUCO_SIZE = 0.21

class ArucoMarkerListener:
    def __init__(self):
        print('INIT ARUCO LISTENER')           
        
        # Initialize ROS Node
        rospy.init_node('aruco_marker_listener', anonymous=True)

        # Create a ROS image subscriber to listen to image topic
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.image_callback)

        # Publisher for publishing the homogeneous transformation matrix with timestamp
        self.pose_pub = rospy.Publisher("/aruco_observation", PoseStamped, queue_size=10)

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
        print('Image callback')

        # Detect ARUCO markers
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bilateralFilter(gray_image, 15, 75, 75)     
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_image = clahe.apply(gray_image)


        if show:
            cv2.imshow('aruco_detect', gray_image)
            cv2.waitKey(100)
        # detect markers
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_image, self.aruco_dict)
        if len(corners) > 0:
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners,
                                                                           ARUCO_SIZE,
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
                self.publish_matrix_with_timestamp(T=Trel, aruco_id=ids[i][0], timestamp=msg.header.stamp)
            return
        print('NO ARUCO COULD BE EXTRACTED')
        rospy.loginfo(f"NO ARUCO COULD BE EXTRACTED.")

    def publish_matrix_with_timestamp(self, T, aruco_id, timestamp):
        """
        To simplify everythin, the last row of the Homogeneous matrix
        is employed to publish the ARUCO id
        """
        rospy.loginfo(f"Homogeneous Transformation Matrix:\n{T}")
        rospy.loginfo(f"FOUND ARUCO id: {aruco_id}")        
    
        position = T.pos()
        # caution is a Homogeneous matrix from pyARTE
        # caution, order. [0, 1, 2, 3] = the quternion is in this library w, x, y, z
        quaternion = T.Q().toarray()

        msg_pose = PoseStamped()        
        msg_pose.header.stamp = timestamp
        msg_pose.header.frame_id = str(aruco_id)

        msg_pose.pose.position.x = position[0]
        msg_pose.pose.position.y = position[1]
        msg_pose.pose.position.z = position[2]

        msg_pose.pose.orientation.x = quaternion[1]
        msg_pose.pose.orientation.y = quaternion[2]
        msg_pose.pose.orientation.z = quaternion[3]
        msg_pose.pose.orientation.w = quaternion[0]

        self.pose_pub.publish(msg_pose)
        
        



if __name__ == '__main__':
    try:
        aruco_listener = ArucoMarkerListener()
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()


