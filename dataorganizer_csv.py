import csv
import os

# ****************** 
# Only edit variable below with the name of the dataset
# This script should be placed at the same level as the "data" directory
# ******************
DATASET = "20190925"
# ******************
theclass= "baxter"
# ******************

# ******************
# DO NOT EDIT FROM HERE!!!
# ******************
currentWorkingFilePath = os.path.dirname(__file__)
directoryPath = os.path.dirname(currentWorkingFilePath)

data_path = directoryPath

file_path = data_path +"/"+ DATASET + "/" + DATASET + ".csv"
with open(DATASET + ".csv", mode='w', newline='', encoding='utf-8') as dataset_file:
    dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    dataset_writer.writerow(['right', 'left', 'disparity', 'proprioception'])

    base_names = []
    directory = os.fsencode(data_path +"/"+ DATASET + "/" + theclass + "/images_right")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            #print(filename[:-4])
            base_names.append(filename[:-4])
            continue
        else:
            continue

    base_names.sort()
    for name in base_names:
        img_right_path = data_path +"/"+ DATASET + "/" + theclass + "/images_right/" + name + ".jpg"

        img_left_path = data_path +"/"+ DATASET + "/" + theclass + "/images_left/" + name + ".jpg"
        img_left_path = img_left_path.replace("imageright","imageleft")

        img_disparity = data_path +"/"+ DATASET + "/" + theclass + "/images_disparity/" + name + ".yaml"
        img_disparity = img_disparity.replace("imageright","disparity")

        pro_path = data_path +"/"+ DATASET + "/" + theclass + "/pro/" + name + ".yaml"
        pro_path = pro_path.replace("imageright","pro")
        
        dataset_writer.writerow([img_right_path, img_left_path, img_disparity, pro_path])

print("Dataset mapping to csv, Done!")