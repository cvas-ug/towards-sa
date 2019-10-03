import os

#filespath = os.path.dirname(os.path.abspath(__file__))
#imagespath = filespath+ '/' + 'images'
#propath = filespath+ '/' + 'pro'
suffix = "fcenv"
right_image_directory_in_str = 'images_right_' + suffix
pro_directory_in_str = 'pro_' + suffix

left_image_directory_in_str = 'images_left_' + suffix
disparity_image_directory_in_str = 'images_disparity_' + suffix

procount = 0
rightimagecount = 0
leftimagecount = 0
disparityimagecount = 0

rightimages = os.listdir(right_image_directory_in_str)
pro = os.listdir(pro_directory_in_str)
leftimages = os.listdir(left_image_directory_in_str)
disparityimages = os.listdir(disparity_image_directory_in_str)

imagestopro = [i.replace("imageright","pro") for i in rightimages]
imagestopro = [i.replace("jpg","yaml") for i in imagestopro]

leftimagestopro = [i.replace("imageleft","pro") for i in leftimages]
leftimagestopro = [i.replace("jpg","yaml") for i in leftimagestopro]

disparityimagestopro = [i.replace("disparity","pro") for i in disparityimages]
#disparityimagestopro = [i.replace("jpg","yaml") for i in disparityimagestopro]

#-------------------------------------------------------------------
#-------------------------------------------------------------------
unwanted_right_images = set(imagestopro) - set(pro)
unwantedpro_right = set(pro) - set(imagestopro)

for file in sorted(unwanted_right_images):
    rightimagecount += 1
    # rename to comply with pro file name
    imagefilename = file.replace("pro", "imageright")
    imagefilename = imagefilename.replace("yaml", "jpg")
  
    # move
    os.rename(right_image_directory_in_str+"/"+imagefilename, "unwanted_right/"+imagefilename)

for file in sorted(unwantedpro_right):
    procount += 1

    # move
    os.rename(pro_directory_in_str+"/"+file, "unwanted_right_pro/"+file)

print("Number of proprioseption moved is "+ procount.__str__())
print("Number of right images moved is "+ rightimagecount.__str__())

#-------------------------------------------------------------------
#-------------------------------------------------------------------
unwanted_left_images = set(leftimagestopro) - set(pro)
unwantedpro_left = set(pro) - set(leftimagestopro)

procount = 0
for file in sorted(unwanted_left_images):
    leftimagecount += 1
    # rename to comply with pro file name
    imagefilename = file.replace("pro", "imageleft")
    imagefilename = imagefilename.replace("yaml", "jpg")
  
    # move
    os.rename(left_image_directory_in_str+"/"+imagefilename, "unwanted_left/"+imagefilename)

for file in sorted(unwantedpro_left):
    procount += 1

    # move
    os.rename(pro_directory_in_str+"/"+file, "unwanted_left_pro/"+file)

print("Number of proprioseption moved is "+ procount.__str__())
print("Number of left images moved is "+ leftimagecount.__str__())

#-------------------------------------------------------------------
#-------------------------------------------------------------------
unwanted_disparity_images = set(disparityimagestopro) - set(pro)
unwantedpro_disparity = set(pro) - set(disparityimagestopro)

procount = 0
for file in sorted(unwanted_disparity_images):
    disparityimagecount += 1
    # rename to comply with pro file name
    imagefilename = file.replace("pro", "disparity")
    #imagefilename = imagefilename.replace("yaml", "jpg")
  
    # move
    os.rename(disparity_image_directory_in_str+"/"+imagefilename, "unwanted_disparity/"+imagefilename)

for file in sorted(unwantedpro_disparity):
    procount += 1

    # move
    os.rename(pro_directory_in_str+"/"+file, "unwanted_disparity_pro/"+file)

print("Number of proprioseption moved is "+ procount.__str__())
print("Number of right images moved is "+ disparityimagecount.__str__())