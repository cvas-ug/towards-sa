import csv
import os
import pandas as pd
import numpy as np

from numpy.random import RandomState


# ****************** 
# Only edit variable below with the name of the dataset
# This script should be placed at the same level as the "data" directory
# ******************
DATASET = "20190925"
folders_class = ["me", "env"]
#dataset_folders = ["il", "fg", "ft"] 
#dataset_folders = ["fc", "il", "fg"]
dataset_folders = ["fc", "fg", "ft"]
#dataset_folders = ["fc", "il", "ft"]
# ******************
theclass= "baxter"
# ******************
csv_file = '20190925.csv'
validation_split = 0.8
#env_csv = '20190925_env_ilfgft.csv'
#me_csv = '20190925_me_ilfgft.csv'

#env_csv = '20190925_env_fcilfg.csv'
#me_csv = '20190925_me_fcilfg.csv'

#env_csv = '20190925_env_fcfgft.csv'
#me_csv = '20190925_me_fcfgft.csv'

#env_csv = '20190925_env_fcilft.csv'
#me_csv = '20190925_me_fcilft.csv'

datagroup = "".join(dataset_folders)
# ******************
# DO NOT EDIT FROM HERE!!!
# ******************
currentWorkingFilePath = os.path.dirname(__file__)
directoryPath = os.path.dirname(currentWorkingFilePath)

data_path = directoryPath
directories = os.listdir(directoryPath + "/" + DATASET)

# Create a csv files from dataset class folders. 
#
def datasetfolder_tocsv():
    for aclass in folders_class:
        with open(DATASET + datagroup + "/" + DATASET + "_" + aclass + "_" + datagroup + ".csv", mode='w', newline='', encoding='utf-8') as dataset_file:
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

    both_classes_training_data.to_csv('20190925'+datagroup+'/train.csv', index=False)
    both_classes_test_data.to_csv('20190925'+datagroup+'/eval.csv', index=False)



# Create a csv file to be used as test group. 
#
def create_test_only_dataset(groupname, testfolder="testgroups"):
        testdir = "testgroups"
        with open(testdir + '/' + DATASET + groupname + ".csv", mode='w', newline='', encoding='utf-8') as dataset_file:
            dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            dataset_writer.writerow(['right', 'left', 'disparity', 'proprioception'])
            for aclass in folders_class:
                base_names = []
                directory = os.fsencode(data_path +"/"+ DATASET + "/images_right_" + groupname + aclass)
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
                    img_right_path = data_path +"/"+ DATASET + "/images_right_"+groupname+aclass+"/"+ name + ".jpg"

                    img_left_path = data_path +"/"+ DATASET + "/images_left_"+groupname+aclass+"/"+ name + ".jpg"
                    img_left_path = img_left_path.replace("imageright","imageleft")

                    img_disparity = data_path +"/"+ DATASET + "/images_disparity_"+groupname+aclass+"/"+ name +".yaml"
                    img_disparity = img_disparity.replace("imageright","disparity")

                    pro_path = data_path +"/"+ DATASET + "/" + "pro_"+groupname+aclass+"/"+ name + ".yaml"
                    pro_path = pro_path.replace("imageright","pro")
                    
                    dataset_writer.writerow([img_right_path, img_left_path, img_disparity, pro_path])


def create_case1234_dataset(dataintofourdivisions, generate_separate_csv):
    dataframe = pd.read_csv(dataintofourdivisions)
    env_records = dataframe.loc[dataframe.right.str.contains('images_right_*.*env'), :]
    me_records = dataframe.loc[dataframe.right.str.contains('images_right_*.*me'), :]

    split_into_cases(me_records, env_records, generate_separate_csv= generate_separate_csv, seed=2481)


def split_into_cases(me_df, env_df, case1_percent=.25, case2_percent=.25, case3_percent=.25, case4_percent=.25, generate_separate_csv= False, seed=None):
    np.random.seed(seed)    
    if len(me_df.index) < len(env_df.index):
        m = len(me_df.index)
    else:
        m = len(env_df.index)

    perm = np.random.permutation(me_df.index[:m])
    case1 = int(case1_percent * m)
    case2 = int(case2_percent * m)
    case3 = int(case3_percent * m)
    case4 = int(case4_percent * m)
    me_c1 = me_df.iloc[perm[:case1]]
    me_c2 = me_df.iloc[perm[case1:case2+case3]]
    me_c3 = me_df.iloc[perm[case2+case3:case2+case3+case4]]
    me_c4 = me_df.iloc[perm[case2+case3+case4:case2+case3+case4+case1]]

    #env_df = env_df.reset_index()
    eperm = np.random.permutation(env_df.index)
    #m = len(df.index)
    env_c1 = env_df.iloc[perm[:case1]]
    env_c2 = env_df.iloc[perm[case1:case2+case3]]
    env_c3 = env_df.iloc[perm[case2+case3:case2+case3+case4]]
    env_c4 = env_df.iloc[perm[case2+case3+case4:case2+case3+case4+case1]]

    if generate_separate_csv:
        generate_csv(me_c1, case=1)
        generate_csv(env_c1, case=2) 
        create_case3_or_case4_csv(me=me_c3, env=env_c3, case=3)
        create_case3_or_case4_csv(me=me_c4, env=env_c4, case=4)
        print("Four cases [cas1, case2, case3, and case4] csv files, created!, in "+DATASET + datagroup+" folder")
    else:
        env_c3['right']= me_c3['right'].values
        env_c4['proprioception']= me_c4['proprioception'].values
        allcases = create_cases1234_csv(me_c1, env_c1, env_c3, env_c4)
        generate_csv(allcases, "all")
        print("Four cases [cas1, case2, case3, and case4] csv file in one file, created!, in "+DATASET + datagroup+" folder")


# concatenate all cases into one traning csv file
#
def create_cases1234_csv(case1, case2, case3, case4):
    result = pd.concat([case1, case2, case3, case4])
    return result


def generate_csv(data, case):
    data.to_csv('20190925'+datagroup+'/train_case'+str(case)+'.csv', index=False)


def create_case3_or_case4_csv(me, env, case):
    #case3 is Me image 
    if case == 3:
        env['right']= me['right'].values
    #case4 is Me pro 
    if case == 4:
        env['proprioception']= me['proprioception'].values

    env.to_csv('20190925'+datagroup+'/train_case'+str(case)+'.csv', index=False)


# join two dataframes into one return them
def join_case(me,env):
    result = pd.concat([me, env])
    return result

# mapping to csv by generate two classes files 
#datasetfolder_tocsv()
# generate train and test csvs from csvs generated by datasetfolder_tocsv()
#create_datasets(DATASET+datagroup+"/"+env_csv, DATASET+datagroup+"/"+me_csv)

# create test csv group
#create_test_only_dataset("ft")
#print("Test dataset csv, created!")

# divide the training dataset into four cases
create_case1234_dataset(dataintofourdivisions="20190925fcfgft/train.csv", generate_separate_csv = True)
#create_case3and4_dataset(dataintofourdivisions="20190925fcfgft/20190925_me_fcfgft.csv")
