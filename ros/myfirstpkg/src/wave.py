#!usr/bin/env python
import rospy

# baxter_interface - Baxter Python API
import baxter_interface

# initialize our ROS node, registering it with the Master
rospy.init_node('Hello_Baxter')

# create an instance of baxter_interface's Limb class
limb = baxter_interface.Limb('right')

# get the right limb's current joint angles
angles = limb.joint_angles()

# print the current joint angles
print angles

# reassign new joint angles (all zeros) which we will later command to the limb
#rangles['right_s0']=1.1900197757717637
#rangles['right_s1']=1.580189606296038
#rangles['right_e0']=-0.31591922705039543
#rangles['right_e1']=0.7176624245312198
#rangles['right_w0']=-2.3575903577717883
#rangles['right_w1']=1.2540506988542894
#rangles['right_w2']=0.27234822304715767

#langles['left_s0']=0.030374727511451205
#langles['left_s1']=0.5092385557356529
#langles['left_e0']=0.10100607008182116
#langles['left_e1']=1.047004645160059
#langles['left_w0']=-1.4897786808240419
#langles['left_w1']=-0.034052693471161355
#langles['left_w2']=-0.13887496462490567

# print the joint angle command
print angles

# move the right arm to those joint angles
limb.move_to_joint_positions(angles)

# Baxter wants to say hello, let's wave the arm

# store the first wave position 
wave_1 = {'right_s0': -0.459, 'right_s1': -0.202, 'right_e0': 1.807, 'right_e1': 1.714, 'right_w0': -0.906, 'right_w1': -1.545, 'right_w2': -0.276}

# store the second wave position
wave_2 = {'right_s0': -0.395, 'right_s1': -0.202, 'right_e0': 1.831, 'right_e1': 1.981, 'right_w0': -1.979, 'right_w1': -1.100, 'right_w2': -0.448}

# wave three times
for _move in range(3):
    limb.move_to_joint_positions(wave_1)
    limb.move_to_joint_positions(wave_2)

