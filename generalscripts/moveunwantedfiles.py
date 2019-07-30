import os

#filespath = os.path.dirname(os.path.abspath(__file__))
#imagespath = filespath+ '/' + 'images'
#propath = filespath+ '/' + 'pro'

image_directory_in_str = 'images'
pro_directory_in_str = 'pro'
procount = 0
imagecount = 0
images = os.listdir(image_directory_in_str)
pro = os.listdir(pro_directory_in_str)

imagestopro = [i.replace("image","pro") for i in images]
imagestopro = [i.replace("jpg","yaml") for i in imagestopro]

unwantedimages = set(imagestopro) - set(pro)
unwantedpro = set(pro) - set(imagestopro)

for file in sorted(unwantedimages):
    imagecount += 1
    # rename to comply with pro file name
    imagefilename = file.replace("pro", "image")
    imagefilename = imagefilename.replace("yaml", "jpg")
  
    # move
    os.rename(image_directory_in_str+"/"+imagefilename, "unwanted/"+imagefilename)

for file in sorted(unwantedpro):
    procount += 1

    # move
    os.rename(pro_directory_in_str+"/"+file, "unwanted/"+file)

print("Number of proprioseption moved is "+ procount.__str__())
print("Number of images moved is "+ imagecount.__str__())