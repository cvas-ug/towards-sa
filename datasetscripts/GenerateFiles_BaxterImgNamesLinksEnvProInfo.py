#level1-Case3
#swap information and generate files.

import yaml
import os
import torch

filespath = os.path.abspath(__package__)
filespath = os.path.dirname(filespath)
# Get proprioception information
propath = filespath + '/' + 'traindatacase3' + '/' + 'env' + '/' + 'pro' + '/'
envpro = os.listdir(propath)

#To store/match files names with same names in this folder.
baxterImgPath = filespath + '/' + 'traindatacase3' + '/' + 'baxter' + '/' + 'images' + '/'
baxterImg = os.listdir(baxterImgPath) #envpropath)


#onefile = 'allrandomunsyncedprobasedonproofenvranges.txt'
#onefile = 'allrandomunsyncedprobasedonproofbaxterranges.txt'
#onefile = 'allrandomunsyncedprobasedonproofenvranges_of_training_data.txt'
#onefile = 'baxterproinformation_associatedwith_valenvpronames.txt'
procount = 0
yaml_files_changed_count = 0

#allpropath = filespath + '/' + '20190429' + '/' + 'randomuncyned' + '/'
newfilepropath = filespath + '/' + 'traindatacase3' + '/' + 'converted' + '/'


def getprofilename(baxterimagefile):
    imagefilename = baxterimagefile.replace("image", "pro")
    imagefilename = imagefilename.replace("jpg", "yaml")
    return imagefilename

def get_yaml_information(envprofile):
    with open(propath + envprofile) as f:
        doc = yaml.load(f)
    return doc

def store_env_info_into_filename_comply_with_baxter_image_filename(env_pro_dictionary, pro_filename_generated_to_comply_with_baxter_image_name):
    #totalprofiles[envprofile] = baxter_dictionary
    with open(newfilepropath + pro_filename_generated_to_comply_with_baxter_image_name, 'w') as f:
        yaml.dump(env_pro_dictionary, f)

if __name__ == "__main__":
    #for file in sorted(envpro):
    for envprofile, baxterimgfile in zip(envpro, baxterImg):
        yaml_files_changed_count += 1
        pro_filename_generated_to_comply_with_baxter_image_name = getprofilename(baxterimgfile) 
        env_pro_dictionary = get_yaml_information(envprofile)

        store_env_info_into_filename_comply_with_baxter_image_filename(env_pro_dictionary, pro_filename_generated_to_comply_with_baxter_image_name)
    print("Number of Yaml files processed: {}".format(yaml_files_changed_count))
    print("Association files generated: {}".format(newfilepropath))
    exit(0)