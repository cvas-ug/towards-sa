import yaml
import os
import torch

filespath = os.path.abspath(__package__)
# Get randomization range from env validation pro files
propath = filespath + '/' + '20190429' + '/' + 'train' + '/' + 'env' + '/' + 'pro' + '/'

# Get randomization range from baxter validation pro files
#propath = filespath + '/' + '20190416' + '/' + 'val' + '/' + 'baxter' + '/' + 'pro' + '/'
pro = os.listdir(propath)

#To store/match files names with same names in this folder.
envpropath = filespath + '/' + '20190401' + '/' + 'val' + '/' + 'env' + '/' + 'pro' + '/'
envpro = os.listdir(envpropath)


#onefile = 'allrandomunsyncedprobasedonproofenvranges.txt'
#onefile = 'allrandomunsyncedprobasedonproofbaxterranges.txt'
onefile = 'allrandomunsyncedprobasedonproofenvranges_of_training_data.txt'

procount = 0
yaml_files_changed_count = 0

allpropath = filespath + '/' + '20190429' + '/' + 'randomuncyned' + '/'
totalprofiles = {}

def find_min_and_max():
    procount = 0
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

def generate_random(max_velocity, min_velocity, max_effort, min_effort, max_position, min_position):
    # generate random
    random_numbers_of_tensor = torch.Tensor(1,19).uniform_(min_velocity, max_velocity)
    #random_numbers_of_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    random_numbers_of_tensor = torch.squeeze(random_numbers_of_tensor)
    velocity_random_numbers_list = random_numbers_of_tensor.tolist()

    random_numbers_of_tensor = torch.Tensor(1,19).uniform_(min_effort, max_effort)
    #random_numbers_of_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    random_numbers_of_tensor = torch.squeeze(random_numbers_of_tensor)
    effort_random_numbers_list = random_numbers_of_tensor.tolist()
    
    random_numbers_of_tensor = torch.Tensor(1,19).uniform_(min_position, max_position)
    #random_numbers_of_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    random_numbers_of_tensor = torch.squeeze(random_numbers_of_tensor)
    position_random_numbers_list = random_numbers_of_tensor.tolist()

    return velocity_random_numbers_list, effort_random_numbers_list, position_random_numbers_list

def replace_yaml_information(velocity_random_list, effort_random_list, position_random_list):
    with open(envpropath + file) as f:
        doc = yaml.safe_load(f)
    doc['velocity'] = velocity_random_list
    doc['effort'] = effort_random_list
    doc['position'] = position_random_list
    return doc

def store_into_one_dictionary(current_dict):
    totalprofiles[file] = current_dict

def writeallproprioceptionsintoonefile():
    with open(allpropath + onefile, 'w') as f:
        yaml.safe_dump(totalprofiles, f)


if __name__ == "__main__":
    max_velocity, min_velocity, max_effort, min_effort, max_position, min_position = find_min_and_max()
    print("Max number of velocity: {}".format(max_velocity))
    print("Min number of velocity : {}".format(min_velocity))
    print("Max number of effort: {}".format(max_effort))
    print("Min number of effort : {}".format(min_effort))
    print("Max number of position: {}".format(max_position))
    print("Min number of position : {}".format(min_position))
    for file in sorted(envpro):
        velocity_random_list, effort_random_list, position_random_list = generate_random(max_velocity, min_velocity, max_effort, min_effort, max_position, min_position)
        yaml_files_changed_count += 1
        changed_dictionary = replace_yaml_information(velocity_random_list, effort_random_list, position_random_list)
        store_into_one_dictionary(changed_dictionary)
    writeallproprioceptionsintoonefile()
    print("Number of Yaml files updated with random list/numbers are: {}".format(yaml_files_changed_count))
    print("Changes are under folder: {}".format(propath))
    exit(0)

#1.2994171567839246
#-4.047847009135813
