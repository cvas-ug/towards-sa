import csv
import os
import pandas as pd

from numpy.random import RandomState


# ****************** 
# Only edit variable below with the name of the dataset
# This script should be placed at the same level as the "data" directory
# ******************
DATASET = "20190925"
folders_class = ["me", "env"]
dataset_folders = ["fc", "fg"]
# ******************
theclass= "baxter"
# ******************
csv_file = '20190925.csv'
validation_split = 0.8
env_csv = '20190925_env.csv'
me_csv = '20190925_me.csv'

# ******************
# DO NOT EDIT FROM HERE!!!
# ******************
currentWorkingFilePath = os.path.dirname(__file__)
directoryPath = os.path.dirname(currentWorkingFilePath)

data_path = directoryPath
directories = os.listdir(directoryPath + "/" + DATASET)

for aclass in folders_class:
    file_path = data_path +"/"+ DATASET + "/" + DATASET + "_" + aclass + ".csv"
    with open(DATASET + "_" + aclass + ".csv", mode='w', newline='', encoding='utf-8') as dataset_file:
        dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        dataset_writer.writerow(['right', 'left', 'disparity', 'proprioception'])
        for folder in dataset_folders:
                base_names = []
                directory = os.fsencode(data_path +"/"+ DATASET + "/images_right_"+folder+aclass)
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
                    img_right_path = data_path +"/"+ DATASET + "/images_right_"+folder+aclass+"/"+ name + ".jpg"

                    img_left_path = data_path +"/"+ DATASET + "/images_left_"+folder+aclass+"/"+ name + ".jpg"
                    img_left_path = img_left_path.replace("imageright","imageleft")

                    img_disparity = data_path +"/"+ DATASET + "/images_disparity_"+folder+aclass+"/"+ name +".yaml"
                    img_disparity = img_disparity.replace("imageright","disparity")

                    pro_path = data_path +"/"+ DATASET + "/" + "pro_"+folder+aclass+"/"+ name + ".yaml"
                    pro_path = pro_path.replace("imageright","pro")
                    
                    dataset_writer.writerow([img_right_path, img_left_path, img_disparity, pro_path])

print("Dataset mapping to csv, Done!")


def create_datasets(env_csv, me_csv):
    # Creating training and validation splits: 
    dataframe_me = pd.read_csv(me_csv)
    rng_me = RandomState()
    train_me = dataframe_me.sample(frac=validation_split, random_state=rng_me)
    test_me = dataframe_me.loc[~dataframe_me.index.isin(train_me.index)]

    dataframe_env = pd.read_csv(env_csv)
    rng_env = RandomState()
    train_env = dataframe_env.sample(frac=validation_split, random_state=rng_env)
    test_env = dataframe_env.loc[~dataframe_env.index.isin(train_env.index)]

    both_classes_training_data = train_me.append(train_env)
    both_classes_test_data = test_me.append(test_env)

    both_classes_training_data.to_csv('train.csv', index=False)
    both_classes_test_data.to_csv('eval.csv', index=False)

create_datasets(env_csv, me_csv)