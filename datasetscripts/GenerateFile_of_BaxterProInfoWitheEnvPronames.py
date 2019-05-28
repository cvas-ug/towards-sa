#level1-Case4

import yaml
import os
import torch

filespath = os.path.abspath(__package__)
# Get information from baxter validation pro files
propath = filespath + '/' + '20190514-case4' + '/' + 'val' + '/' + 'baxter' + '/' + 'pro' + '/'
baxterpro = os.listdir(propath)

#To store/match files names with same names in this folder.
envpropath = filespath + '/' + '20190514-case4' + '/' + 'val' + '/' + 'env' + '/' + 'pro' + '/'
envpro = os.listdir(envpropath)


#onefile = 'allrandomunsyncedprobasedonproofenvranges.txt'
#onefile = 'allrandomunsyncedprobasedonproofbaxterranges.txt'
#onefile = 'allrandomunsyncedprobasedonproofenvranges_of_training_data.txt'
onefile = 'baxterproinformation_associatedwith_valenvpronames.txt'
procount = 0
yaml_files_changed_count = 0

#allpropath = filespath + '/' + '20190429' + '/' + 'randomuncyned' + '/'
allpropath = filespath + '/' + '20190514-case4' + '/' + 'baxterrange' + '/'
totalprofiles = {}

def get_yaml_information(baxterprofile):
    with open(propath + baxterprofile) as f:
        doc = yaml.safe_load(f)
    return doc

def store_baxter_info_into_one_dictionary_with_env_filename(baxter_dictionary, envprofile):
    totalprofiles[envprofile] = baxter_dictionary

def writeallproprioceptionsintoonefile():
    with open(allpropath + onefile, 'w') as f:
        yaml.safe_dump(totalprofiles, f)


if __name__ == "__main__":
    #for file in sorted(envpro):
    for baxterprofile, envprofile in zip(baxterpro, envpro):
        yaml_files_changed_count += 1
        baxter_dictionary = get_yaml_information(baxterprofile)
        store_baxter_info_into_one_dictionary_with_env_filename(baxter_dictionary, envprofile)
    writeallproprioceptionsintoonefile()
    print("Number of Yaml files processed: {}".format(yaml_files_changed_count))
    print("Association file generated: {}".format(allpropath + onefile))
    exit(0)