#!usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from std_msgs.msg import String
import random

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('random_outoftherange', anonymous=True)
# The "RobotCommander" object is an interface to Baxter (or any robot) as a whole.
robot = moveit_commander.RobotCommander()
# This is an interface to the world surrounding the robot.
scene = moveit_commander.PlanningSceneInterface()
# This is an interface to one group of joints.  In our case, we want to use the "right_arm".
#We will use this to plan and execute motions
group_both = moveit_commander.MoveGroupCommander("both_arms")
group_right = moveit_commander.MoveGroupCommander("right_arm")
group_left = moveit_commander.MoveGroupCommander("left_arm")

pose_target_left = geometry_msgs.msg.Pose()
pose_target_right = geometry_msgs.msg.Pose()

# orientation.x, orientation.y, orientation.z, orientation.w, position.x, position.y, position.z
right_arm =[
[0.495088964857, 0.868820119093, 0.0052328973379, 0.0033368132184, 0.167799891523, -0.42014878352, 0.218327297379], #0 home
[-0.257, -0.403, 0.726, -0.495, 0.235, -1.298, 1.369],          #1 welcome
[0.342, 0.940, 0.005, 0.004, -0.108, -0.927, 0.617],            #2 Random1
[-0.079, 0.022, -0.495, 0.865,-0.005, -0.475, 2.418],    	#3 mo3aqab
[0.0534863527029, 0.99824127692, -0.00789752135613, 0.024314445976, 0.173722031251, -0.435980228145, 0.218794283283], #4home
[0.233, 0.972, -0.008, 0.022, 0.199, -0.985, 1.218],            #5 karate
[0.194, 0.907, -0.365, 0.078, 0.030, -1.226, 0.655], 		#6 Gerardo
[0.060, -0.001, 0.335, 0.940, 0.311, -0.435, 2.430], 		#7 mo3aqab2
[0.525, -0.551, 0.191, 0.620, -0.068, -0.988, 1.450], 		#8 random2
[-0.207, -0.229, 0.948, -0.081, -0.207, -0.229, 0.948, -0.081], #9random3
[-0.784, -0.185, 0.313, 0.503, -0.864, -0.203, 1.238], 		#10 ma 3indy
[0.388, 0.539, 0.395, 0.635, 0.147, -0.846, 0.895], 		#11 open
#[0.214, 0.976, -0.012, 0.046, 0.166, -0.665, 0.531] 		 #12 spider
[-0.251, 0.967, -0.028, 0.037, -0.016, -0.906, 0.508] 		#12 spider
]

left_arm =[
[-0.55990460938, 0.828529441367, -0.00555327412011, 0.00386708274987, 0.181666578573, 0.413565742627, 0.21833370248], #0 home
[-0.312633560481, 0.445291780918, 0.469006757274, 0.695706941419, 0.276027138406, 1.35779027746, 1.30002692835], #1 welcome
[-0.560, 0.828, -0.006, 0.004, 0.030, 0.937, 0.754],  #2 Random1
[0.050, -0.086, 0.558, 0.824, -0.078, 0.498, 2.404],  #3 mo3aqab
[-0.96083388986, 0.272626749104, 0.0357994117987, 0.0345151254332, 0.158498059536, 0.452384904453, 0.220246798588], #4 home
[0.768, -0.638, -0.020, -0.047, 0.214, 0.982, 1.220], #5 karate
[-0.268, 0.354, 0.587, 0.677, 0.020, 1.234, 1.987],   #6 Gerardo
[0.054, -0.023, 0.644, 0.763, 0.056, 0.542, 2.414],   #7 mo3aqab2
[-0.013, 0.519, -0.381, 0.765, 0.328, 0.374, 2.224],  #8 random2
[0.756, 0.649, -0.024, -0.087, 0.406, 0.740, 0.448],  #9 random3
[0.391, 0.664, -0.569, -0.287, -0.895, 0.206, 1.252], #10 ma 3indy
[0.157, 0.710, 0.246, 0.641, 0.301, 0.976, 0.883],    #11 open
#[0.799, -0.600, -0.030, -0.036, 0.143, 0.689, 0.531]  #12 spider
[0.168, 0.985, 0.026, -0.022, 0.014, 0.859, 0.485]    #12 spider
]

def pose_right(r=0):
	pose_target_right.orientation.x = right_arm[r][0] 
	pose_target_right.orientation.y = right_arm[r][1] 
	pose_target_right.orientation.z = right_arm[r][2] 
	pose_target_right.orientation.w = right_arm[r][3] 
	pose_target_right.position.x = right_arm[r][4]
	pose_target_right.position.y = right_arm[r][5]
	pose_target_right.position.z = right_arm[r][6]

def pose_left(l=0):
	pose_target_left.orientation.x =  left_arm[l][0]
	pose_target_left.orientation.y =  left_arm[l][1]
	pose_target_left.orientation.z =  left_arm[l][2]
	pose_target_left.orientation.w =  left_arm[l][3]
	pose_target_left.position.x =  left_arm[l][4]
	pose_target_left.position.y =  left_arm[l][5]
	pose_target_left.position.z =  left_arm[l][6]



# Now, add your Pose msg to the group's pose target
# group.set_pose_target(pose_target)

rr = rospy.Rate(10) # 10hz

l = 11
r = 11
print("Now execute : right ({}) and left({})".format(r,l))
pose_right(r)
pose_left(l)
group_both.set_pose_target(pose_target_left, end_effector_link = "left_gripper")
group_both.set_pose_target(pose_target_right, end_effector_link = "right_gripper")
plan = group_both.plan()
ret = group_both.execute(plan)
rr.sleep()
