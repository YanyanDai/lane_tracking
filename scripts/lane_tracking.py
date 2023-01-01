#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from lane_detection_main_tracking import *


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
        self.timer_to_sending_data = 0
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)
        
        #command publisher      
        self.cmd_vel = rospy.Publisher('/teleop_cmd_vel', Twist, queue_size=10)
        #
        #self.move_cmd = Twist()  
        #self.move_cmd.linear.x = 0.5
        #self.move_cmd.angular.z = 0
    	
        
    def callback(self,data):
        self.rate.sleep()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        img_tracking_point_lane,angle = lane_detection(cv_image)
        cv2.waitKey(1)
        
        #while not rospy.is_shutdown():
        if self.timer_to_sending_data % 5 == 0 and rospy.is_shutdown() == 0:
            rospy.loginfo("robot move")
            move_cmd = Twist()
            move_cmd.linear.x = 70
            move_cmd.angular.z = angle *30
            self.cmd_vel.publish(move_cmd)
            #self.rate.sleep()
            self.timer_to_sending_data = 0
    
        self.timer_to_sending_data += 1
    
    def shutdown(self):
	#stop aibot
	rospy.loginfo("Stop robot")
	#a default twist has linear.x of 0 and angular.z of 0. So it will stop AIBot
        self.cmd_vel.publish(Twist())

if __name__ == '__main__':
    try:
        main()
        rospy.spin()
    except:
        rospy.loginfo("Lane Tracking node terminated.")
        
    


