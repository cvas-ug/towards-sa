import yaml
import os
import torch

filespath = os.path.abspath(__package__)
propath = filespath + '/' + '20190325' + '/' + 'val' + '/' + 'env' + '/' + 'pro' + '/'
pro = os.listdir(propath)

procount = 0
yaml_files_changed_count = 0

def find_min_and_max():
    procount = 0
    max_value = 0
    min_value = 0
    for file in sorted(pro):
        procount += 1

        with open(propath + file) as f:
            doc = yaml.safe_load(f)

        if max_value < max(doc["velocity"]):
            max_value = max(doc["velocity"]) 
        if min_value > min(doc["velocity"]):
            min_value = min(doc["velocity"])

    print("Number of proprioseption compared is "+ procount.__str__())
    return max_value, min_value

def generate_random(max_val, min_val):
    # generate random
    random_numbers_of_tensor = torch.Tensor(1,19).uniform_(-4.047847009135813,1.2994171567839246)
    #random_numbers_of_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    random_numbers_of_tensor = torch.squeeze(random_numbers_of_tensor)
    random_numbers_of_list = random_numbers_of_tensor.tolist()
    return random_numbers_of_list

def replace_yaml_information(random_list):
    with open(propath + file) as f:
        doc = yaml.safe_load(f)
    doc['velocity'] = random_list

    with open(propath + file, 'w') as f:
        yaml.safe_dump(doc, f)


if __name__ == "__main__":
    max_value, min_value = find_min_and_max()
    print("Max number is : {}".format(max_value))
    print("Min number is : {}".format(min_value))
    for file in sorted(pro):
        random_list_values = generate_random(max_value, min_value)
        yaml_files_changed_count += 1
        replace_yaml_information(random_list_values)
    print("Number of Yaml files updated with random list/numbers are: {}".format(yaml_files_changed_count))
    print("Changes are under folder: {}".format(propath))
    exit

#1.2994171567839246
#-4.047847009135813
