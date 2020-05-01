#this script to test that TimeSynchronizer is going fine.

import rospy
import message_filters
from sensor_msgs.msg import JointState, Image, CameraInfo
from actionlib_msgs.msg import GoalStatusArray
from stereo_msgs.msg import DisparityImage

def callback(js_sub, ci_sub, zr_sub, active_sub, zl_sub, disparityimage_sub):
    print("hi 2")
  # The callback processing the pairs of numbers that arrived at approximately the same time

rospy.init_node('baxter_generates')
js_sub = message_filters.Subscriber('/robot/joint_states', JointState)
ci_sub = message_filters.Subscriber('/cameras/left_hand_camera/camera_info', CameraInfo)

zr_sub = message_filters.Subscriber('/zed/zed_node/right_raw/image_raw_color', Image)
active_sub = message_filters.Subscriber('/move_group/status',  GoalStatusArray)

zl_sub = message_filters.Subscriber('/zed/zed_node/left_raw/image_raw_color', Image)
disparityimage_sub = message_filters.Subscriber('/zed/zed_node/disparity/disparity_image', DisparityImage)

ts = message_filters.ApproximateTimeSynchronizer([js_sub, ci_sub, zr_sub, active_sub, zl_sub, disparityimage_sub], 1, 0.1) #allow_headerless=True)
#ts = message_filters.TimeSynchronizer([zr_sub, zl_sub], 10)
ts.registerCallback(callback)
rospy.spin()
