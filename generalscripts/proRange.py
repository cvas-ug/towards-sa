import yaml
import os
import torch

filespath = os.path.abspath(__package__)
# Get randomization range from env validation pro files
propath = filespath + '/' + '20190429' + '/' + 'val' + '/' + 'baxter' + '/' + 'pro' + '/'

# Get randomization range from baxter validation pro files
#propath = filespath + '/' + '20190416' + '/' + 'val' + '/' + 'baxter' + '/' + 'pro' + '/'

pro = os.listdir(propath)

procount = 0

def find_min_and_max():
    global procount 
    procount= 0
    max_velocity_value = 0
    min_velocity_value = 0
    max_effort_value = 0
    min_effort_value = 0
    max_position_value = 0
    min_position_value = 0
    for file in sorted(pro):
        procount += 1

        with open(propath + file) as f:
            doc = yaml.safe_load(f)

        if max_velocity_value < max(doc["velocity"]):
            max_velocity_value = max(doc["velocity"]) 
        if min_velocity_value > min(doc["velocity"]):
            min_velocity_value = min(doc["velocity"])
        
        if max_effort_value < max(doc["effort"]):
            max_effort_value = max(doc["effort"]) 
        if min_effort_value > min(doc["effort"]):
            min_effort_value = min(doc["effort"])
        
        if max_position_value < max(doc["position"]):
            max_position_value = max(doc["position"]) 
        if min_position_value > min(doc["position"]):
            min_position_value = min(doc["position"])

    print("Number of proprioseption compared is "+ procount.__str__())
    return max_velocity_value, min_velocity_value, max_effort_value, min_effort_value, max_position_value, min_position_value


if __name__ == "__main__":
    max_velocity, min_velocity, max_effort, min_effort, max_position, min_position = find_min_and_max()
    print("Max number of velocity: {}".format(max_velocity))
    print("Min number of velocity : {}".format(min_velocity))
    print("Max number of effort: {}".format(max_effort))
    print("Min number of effort : {}".format(min_effort))
    print("Max number of position: {}".format(max_position))
    print("Min number of position : {}".format(min_position))

    print("Number of Yaml files : {}".format(procount))
    print("Under folder: {}".format(propath))
    exit(0)

#1.2994171567839246
#-4.047847009135813
