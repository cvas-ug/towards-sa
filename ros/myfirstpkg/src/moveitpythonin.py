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
rospy.init_node('random_intherange', anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()

group_both = moveit_commander.MoveGroupCommander("both_arms")
group_right = moveit_commander.MoveGroupCommander("right_arm")
group_left = moveit_commander.MoveGroupCommander("left_arm")

pose_target_left = geometry_msgs.msg.Pose()
pose_target_right = geometry_msgs.msg.Pose()

# orientation.x, orientation.y, orientation.z, orientation.w, position.x, position.y, position.z
right_arm =[
[0.173, 0.908, 0.368, -0.102, 0.595, -0.232, 0.988], 	#0 grippers
[0.197, 0.727, 0.657, -0.036, 0.654, 0.197, 0.903],  	#1 My hands
[0.197, 0.727, 0.657, -0.037, 0.768, 0.045, 0.806],  	#2 hands far down
[0.473, 0.664, 0.497, 0.296, 0.805, -0.056, 0.786],    	#3 
[0.243, 0.727, 0.642, 0.014, 0.804, -0.055, 0.786],     #4
[-0.050, 0.881, 0.190, 0.431, 0.983, 0.003, 0.649],     #5 
[-0.050, 0.881, 0.191, 0.431, 0.997, -0.233, 0.692], 	#6
[0.125, 0.884, -0.173, 0.416, 0.998, -0.234, 0.692],	#7
[0.124, 0.884, -0.173, 0.416, 0.927, -0.143, 0.894],    #8
[-0.044, 0.938, 0.178, 0.294, 0.709, -0.064, 0.989],    #9 
[0.066, 0.987, 0.122, -0.082, 0.764, -0.099, 0.939],	#10 pick
[0.066, 0.987, 0.122, -0.081, 0.419, -0.051, 1.062],	#11
[0.179, 0.158, 0.970, 0.054, 0.555, -0.098, 1.030], 	#12 my gripper
[0.182, 0.697, 0.675, 0.162, 0.806, 0.057, 1.127],	#13
[0.182, 0.697, 0.674, 0.162, 0.855, 0.370, 0.915],	#14
[0.182, 0.697, 0.675, 0.162, 0.888, 0.265, 0.906],	#15 out/in
[0.659, 0.736, 0.123, 0.094, 0.452, -0.723, 0.530], 	#16 out/in
[-0.174, 0.657, 0.229, 0.697, 1.011, 0.244, 0.952]	#17 out/in
]

left_arm =[
[0.649, -0.495, 0.467, 0.341, 0.593, 0.110, 0.800],  #0
[0.689, -0.436, 0.436, 0.381, 0.670, 0.025, 1.098],  #1 My hands
[0.689, -0.437, 0.435, 0.381, 0.784, 0.137, 0.755],  #2 hands far down
[0.814, -0.576, -0.047, 0.057, 0.789, 0.171, 0.754], #3 
[0.282, -0.626, 0.696, -0.208, 0.790, 0.170, 0.755], #4
[0.449, 0.762, 0.048, 0.464, 0.989, 0.119, 0.627],   #5 
[0.448, 0.763, 0.047, 0.464, 0.924, 0.466, 0.699],   #6
[0.240, 0.688, 0.331, 0.599, 0.925, 0.466, 0.700],   #7
[0.240, 0.688, 0.332, 0.599, 0.830, 0.278, 1.096],   #8
[0.570, 0.693, -0.209, 0.390, 0.767, 0.198, 1.034],  #9
[0.755, 0.643, 0.029, 0.120, 0.816, 0.096, 0.924],   #10 pick
[0.765, 0.641, -0.057, 0.020, 0.512, 0.163, 0.990],  #11
[0.170, 0.235, -0.728, 0.621, 0.504, 0.077, 1.083],  #12 my gripper
[0.536, 0.534, -0.521, 0.397, 0.424, 0.098, 1.108],  #13
[0.537, 0.533, -0.520, 0.397, 0.298, 0.004, 1.306],  #14
[-0.147, 0.456, -0.605, 0.636, 1.120, 0.196, 1.710], #15 out/in
[0.421, -0.385, 0.819, 0.064, 0.743, -0.240, 1.186], #16 out/in
[0.854, -0.332, 0.346, -0.202, 0.527, 0.802, 0.490]  #17 out/in
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

first_time_right = True
first_time_left = True
rr = rospy.Rate(10) # 10hz
while not rospy.is_shutdown():
#Now, add your Pose msg to the group's pose target
    count = 0
    while count<=10 :
	print count
    	r = random.randint(0, 17)
    	print("Now execute : right({})".format(r))
    	pose_right(r)
    	group_right.set_pose_target(pose_target_right, end_effector_link = "right_gripper")
    	plan = group_right.plan()
    	ret = group_right.execute(plan)
    	rr.sleep()
	if first_time_right==True:
	    first_time_right = False
            count = -20
	count +=1
	
    count = 0
    while count<=10 :
	print count
    	l = random.randint(0, 17)
    	print("Now execute : left ({})".format(l))
    	pose_left(l)
    	group_left.set_pose_target(pose_target_left, end_effector_link = "left_gripper")
    	plan = group_left.plan()
    	ret = group_left.execute(plan)
    	rr.sleep()
	if first_time_left==True:
	    first_time_left = False
            count = -20
     	count +=1

    count = 0
    while count<=10 :
	print count
    	l = random.randint(0, 17)
    	r = random.randint(0, 17)
    	print("Now execute : right ({}) and left({})".format(r,l))
    	pose_right(r)
    	pose_left(l)
    	group_both.set_pose_target(pose_target_left, end_effector_link = "left_gripper")
    	group_both.set_pose_target(pose_target_right, end_effector_link = "right_gripper")
    	plan = group_both.plan()
    	ret = group_both.execute(plan)
    	rr.sleep()
	count +=1



