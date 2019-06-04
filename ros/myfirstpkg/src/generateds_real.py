#!/usr/bin/env python
import rospy

import message_filters
from sensor_msgs.msg import Image, JointState
from stereo_msgs.msg import DisparityImage
from actionlib_msgs.msg import GoalStatusArray

import cv2
from cv_bridge import CvBridge, CvBridgeError
import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import datetime as dt

import pickle
import csv

counter = 0
todaydate = (dt.date.today()).strftime('%Y%m%d')

if __name__ == '__main__':
    rospy.init_node('baxter_tf_listener')
    # rate = rospy.Rate(.5) # 10hz
    def callback(image, proprioseption, moving, disparityimage):
        try:
            st = yaml.load(str(moving))
            status1 = st["status_list"][0]['status']
            status2 = st["status_list"][1]['status']
        except IndexError:
            status2 = 0

        if ((status1 == 4) and (status2 == 3)) or ((status1 == 3) and (status2 == 3)) or ((status1 == 3) and (status2 == 0)):
            global counter
            counter +=1
            x = proprioseption
            print("Image are capturing...{}".format(counter))
            with open("./dataset/pro/{}_pro{}_{}.txt".format(todaydate, counter, proprioseption.header.stamp.secs), 'wb') as intofile:
                intofile.write(str(x))
            try:
                cv_image = bridge.imgmsg_to_cv2(image, "bgr8")
            except CvBridgeError as e:
                print(e)    
            cv2.imwrite("./dataset/images/{}_image{}_{}.jpg".format(todaydate, counter, image.header.stamp.secs), cv_image)
            #cv2.imwrite("./dataset/images_disparity/{}_image{}_{}.png".format(todaydate, counter, image.header.stamp.secs), cv_disparityimage)
        
            disparity_file = './dataset/images_disparity/{}_pro{}_{}.yaml'.format(todaydate, counter, proprioseption.header.stamp.secs)
            with open(disparity_file, 'w') as outfile:
                yaml.dump(disparityimage, outfile, Dumper=Dumper)
            #with open(disparity_file, 'r') as inputfile:
            #    here = yaml.load(inputfile, Loader=Loader)
            #    print(here.image)
        
    print(Image)
    print(JointState)
    print(GoalStatusArray)
    print(DisparityImage)
    bridge = CvBridge()
    image_sub = message_filters.Subscriber('/zed/zed_node/rgb_raw/image_raw_color', Image)
    info_sub = message_filters.Subscriber('/robot/joint_states', JointState)
    active_sub = message_filters.Subscriber('/move_group/status',  GoalStatusArray)
    disparityimage_sub = message_filters.Subscriber('/zed/zed_node/disparity/disparity_image', DisparityImage)
    # ts = message_filters.TimeSynchronizer([image_sub, info_sub], 10)
    # error time can be between two messsages 
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub, active_sub, disparityimage_sub], queue_size=10, slop=0.1)
    ts.registerCallback(callback)
    rospy.spin()
