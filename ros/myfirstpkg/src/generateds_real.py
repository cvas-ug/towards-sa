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
            #pickle-----------------
            #with open(config_dictionary_file, 'wb') as config_dictionary_file:
            #    pickle.dump(config_dictionary, config_dictionary_file)
            #-----------------------
            #cv_disparityimage = bridge.imgmsg_to_cv2(disparityimage.image, "32FC1") #"8UC1")
            #cv2.imshow("image window", cv_disparityimage)
            #csv--------------------
            #with open(config_dictionary_file,'wb') as fileobj:
            #    newFile = csv.writer(fileobj)
            #    newFile.writerow([config_dictionary])
            #-------------------
            #oldway-------------
            #newFile = open(config_dictionary_file,'w')
            #newFile.write(str(config_dictionary))
            #newFile.close()
            #-------------------
            try:
                cv_image = bridge.imgmsg_to_cv2(image, "bgr8")
            except CvBridgeError as e:
                print(e)    
            cv2.imwrite("./dataset/images/{}_image{}_{}.jpg".format(todaydate, counter, image.header.stamp.secs), cv_image)
            #cv2.imwrite("./dataset/images_disparity/{}_image{}_{}.png".format(todaydate, counter, image.header.stamp.secs), cv_disparityimage)
            #yaml---------------
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

#Baseline - no disparity capturing 4sec for 20 images - 10sec for 50 images
#csv 14sec for 20images
#pickle 5.5 for 20images - 15s for 50 images
#oldway  11 for 20 images
#YAML 20 FOR 3 images only.
#YAML cdump 5 for 20 images. - 10s for 50 images - winner
