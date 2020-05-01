import os
import errno

suffix = "fcenv"
right_image_directory_in_str = 'images_right_' + suffix
pro_directory_in_str = 'pro_' + suffix

left_image_directory_in_str = 'images_left_' + suffix
disparity_image_directory_in_str = 'images_disparity_' + suffix

procount = 0
rightimagecount = 0
leftimagecount = 0
disparityimagecount = 0

try:
    os.mkdir('unwanted')
    os.mkdir('unwanted/images_right_' + suffix)
    os.mkdir('unwanted/images_left_' + suffix)
    os.mkdir('unwanted/images_disparity_' + suffix)
    os.mkdir('unwanted/pro_' + suffix)
except OSError as e:
    print("folders already exists...")

pro = os.listdir(pro_directory_in_str)
rightimages = os.listdir(right_image_directory_in_str)
leftimages = os.listdir(left_image_directory_in_str)
disparityimages = os.listdir(disparity_image_directory_in_str)

imagestopro = [i.replace("imageright","pro") for i in rightimages]
imagestopro = [i.replace("jpg","yaml") for i in imagestopro]
leftimagestopro = [i.replace("imageleft","pro") for i in leftimages]
leftimagestopro = [i.replace("jpg","yaml") for i in leftimagestopro]
disparityimagestopro = [i.replace("disparity","pro") for i in disparityimages]

unwanted_right_images = set(imagestopro) - set(pro)
unwantedpro_right = set(pro) - set(imagestopro)
unwanted_left_images = set(leftimagestopro) - set(pro)
unwantedpro_left = set(pro) - set(leftimagestopro)
unwanted_disparity_images = set(disparityimagestopro) - set(pro)
unwantedpro_disparity = set(pro) - set(disparityimagestopro)

allunion = unwanted_right_images | unwanted_left_images | unwanted_disparity_images | unwantedpro_right | unwantedpro_left | unwantedpro_disparity
print(allunion)
imagecount = 0
subnames = ["imageright", "imageleft", "disparity", "pro"]
directories = [right_image_directory_in_str, left_image_directory_in_str, disparity_image_directory_in_str, pro_directory_in_str]

for file in sorted(allunion):
    # changing names of file and move if available 
    for subname, directory in zip(subnames, directories):
        if subname == "disparity":
            imagefilename = file.replace("pro", subname)
            imagefilename = imagefilename.replace("jpg", "yaml")
        if subname == "imageright" or subname == "imageleft":
            imagefilename = file.replace("pro", subname)
            imagefilename = imagefilename.replace("yaml", "jpg")
        if subname == "pro":
            imagefilename = file
        try:
            # move
            os.rename(directory+"/"+imagefilename, "unwanted/" + directory + "/"+imagefilename)
            imagecount += 1 
        except OSError as e:
            print("file not in "+directory )          

print("Number of images/pro moved is "+ imagecount.__str__())