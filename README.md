# Towards Artificial Robotic Self-Awareness

**[Ali AlQallaf](https://github.com/Al-Qallaf) and [Gerardo Aragon-Camarasa](https://github.com/gerac83)**

Source code for Towards Artificial Robotic Self-Awareness

## Abstract

While humans are aware of their body and capabilities, robots are not. To address this, we present in this paper the first step towards an artificial self-aware architecture for dual-arm robots. Our approach is inspired by human self-awareness developmental levels and serves as the underlying building block for a robot to achieve awareness of itself while carrying out tasks in an environment. Our assumption is that a robot has to know itself before interacting with the environment in order to be able to support different robotic tasks. Hence, we developed a neural network architecture to enable a robot to know itself by differentiating its limbs from different environments using visual and proprioception sensory inputs. we demonstrate experimentally that a robot can distinguish itself with an accuracy of ~89% from four different uncluttered and cluttered environmental settings and under confounding and adversarial input signals.

## Implementation details

* Module implemented with Pytorch.

* We used the robot’s vision and proprioception capabilities as the sensory inputs for our approach. 

* Vision comprises RGB images captured using a stereo ZED camera from Stereolabs configured to output images at 720p resolution. Captured images contain a representation of the robot’s arms or environment.

* Proprioception consists of the robot’s joint states; being velocity, angular position, and motor torque. 

* Form four cases experimental cases, and follow the leave-one-out cross-validation strategy to test the trained model.

## Results

A robot can differentiate itself from the environment with an average classification accuracy of 88.7% using unseen test samples and across four different scenes’ groups.
