#!/usr/bin/env python
import rospy

import message_filters
from sensor_msgs.msg import Image, JointState

import cv2
from cv_bridge import CvBridge, CvBridgeError
import yaml

import datetime as dt

counter = 0
todaydate = (dt.date.today()).strftime('%Y%m%d')

if __name__ == '__main__':
    rospy.init_node('baxter_tf_listener')
    # rate = rospy.Rate(.5) # 10hz
    print("many")
    def callback(image, proprioseption):
        # Solve all of perception here...
        global counter
        counter +=1
        #==============        
        x = proprioseption
#        print("Proooo------------------------------------")        
#        print(x)
#        print(type(x))
        
        with open("./dataset/pro/{}_pro{}_{}.txt".format(todaydate, counter, proprioseption.header.stamp.secs), 'wb') as intofile:
            intofile.write(str(x))        
#        print("Stringss----------------------------------")               
#        x = str(x)        
#        print(x)
#        print(type(x))
#        print("Load pro{}_{}.txt------------------------------------".format(counter, proprioseption.header.stamp.secs)) 
#        with open("./dataset/pro{}_{}.txt".format(counter, proprioseption.header.stamp.secs), 'r') as outputfile:
#            x = outputfile.read()
#        print(type(x))
#        print("Yaml load the txt file--------------------")
#        x = yaml.load(x)
#        # print(x["velocity"])
#        print(x)
#        print(type(x))
         #==============
        try:
            cv_image = bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv2.imshow("image window", cv_image)
        cv2.imwrite("./dataset/images/{}_image{}_{}.jpg".format(todaydate, counter, image.header.stamp.secs), cv_image)
        print("Image are capturing...{}".format(counter))
        # cv2.waitKey(3)
	# rate.sleep()
	# t.sleep(3)
        

    print(Image)
    print(JointState)
    bridge = CvBridge()
    image_sub = message_filters.Subscriber('/camera/rgb/image_raw', Image)
    info_sub = message_filters.Subscriber('/robot/joint_states', JointState)
    # ts = message_filters.TimeSynchronizer([image_sub, info_sub], 10)
    # error time can be between two messsages 
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, info_sub], queue_size=10, slop=0.1)
    ts.registerCallback(callback)
    rospy.spin()
