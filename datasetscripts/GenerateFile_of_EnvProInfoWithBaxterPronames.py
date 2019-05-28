#level1-Case3
import yaml
import os
import torch

filespath = os.path.abspath(__package__)
# Get information from env validation pro files
propath = filespath + '/' + '20190514-case3' + '/' + 'val' + '/' + 'env' + '/' + 'pro' + '/'
envpro = os.listdir(propath)

#To store/match files names with same names in this folder.
baxterpropath = filespath + '/' + '20190514-case3' + '/' + 'val' + '/' + 'baxter' + '/' + 'pro' + '/'
baxterpro = os.listdir(baxterpropath)


#onefile = 'allrandomunsyncedprobasedonproofenvranges.txt'
#onefile = 'allrandomunsyncedprobasedonproofbaxterranges.txt'
#onefile = 'allrandomunsyncedprobasedonproofenvranges_of_training_data.txt'
onefile = 'envproinformation_associatedwith_baxtervalpronames.txt'
procount = 0
yaml_files_changed_count = 0

#allpropath = filespath + '/' + '20190429' + '/' + 'randomuncyned' + '/'
allpropath = filespath + '/' + '20190514-case3' + '/' + 'envrange' + '/'
totalprofiles = {}

def get_yaml_information(envprofile):
    with open(propath + envprofile) as f:
        doc = yaml.safe_load(f)
    #doc['velocity'] = velocity_random_list
    #doc['effort'] = effort_random_list
    #doc['position'] = position_random_list
    return doc

def store_env_info_into_one_dictionary_with_baxter_filename(env_dictionary, baxterprofile):
    totalprofiles[baxterprofile] = env_dictionary

def writeallproprioceptionsintoonefile():
    with open(allpropath + onefile, 'w') as f:
        yaml.safe_dump(totalprofiles, f)


if __name__ == "__main__":
    #for file in sorted(envpro):
    for envprofile, baxterprofile in zip(envpro, baxterpro):
        yaml_files_changed_count += 1
        env_dictionary = get_yaml_information(envprofile)
        store_env_info_into_one_dictionary_with_baxter_filename(env_dictionary, baxterprofile)
    writeallproprioceptionsintoonefile()
    print("Number of Yaml files processed: {}".format(yaml_files_changed_count))
    print("Association file generated: {}".format(allpropath + onefile))
    exit(0)