#!/usr/bin/env python

# finish import part
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from lane_detection_main_camera import *


class main:

    def __init__(self):
        rospy.init_node('main',anonymous=False)
        #stop AIBot
	rospy.loginfo("To stop robot CTRL + c")
	#what function to call when you ctrol +c
	rospy.on_shutdown(self.shutdown)
        
        #image subscriber
        self.rate = rospy.Rate(10)
	self.bridge = CvBridge()
	self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)       
    	
        
    def callback(self,data):
        self.rate.sleep()
	try:
	    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
	except CvBridgeError as e:
	    print(e)
        
	img_line = lane_detection(cv_image)
	cv2.waitKey(1)
    
    def shutdown(self):
	#stop aibot
	rospy.loginfo("Stop robot")
	rospy.sleep(1)

if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except:
        rospy.loginfo("Lane Tracking node terminated.")
    
    


