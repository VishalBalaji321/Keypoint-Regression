import pandas as pd
import numpy as np
import os
import random

# Convert .xml to .csv: https://www.convertcsv.com/xml-to-csv.htm

IMG_SIZE = 80
MAIN_DIRECTORY = 'Dataset_Cone_Keypoints'
TRAIN_VAL_SPLIT = 0.1 # 10% for val and 90% for training

def train_val_split(row_array, split=TRAIN_VAL_SPLIT):
    random.shuffle(row_array)

    len_data = len(row_array)

    # calculate the training data samples length
    valid_split = int(len_data * split)
    # calculate the training data samples length
    train_split = int(len_data - valid_split)
    training_samples = row_array[:train_split][:]
    valid_samples = row_array[-valid_split:][:]
    return training_samples, valid_samples



#anno_data = pd.read_csv('Dataset_Cone_Keypoints/new_indoor/annotations.csv')
# use just the required cols
required_cols = [
    "_name",
    "_width",
    "_height",
    "points/0/_label",
    "points/0/_points",
    "points/1/_label",
    "points/1/_points",
    "points/2/_label",
    "points/2/_points",
    "points/3/_label",
    "points/3/_points",
    "points/4/_label",
    "points/4/_points",
    "points/5/_label",
    "points/5/_points",
    "points/6/_label",
    "points/6/_points",
    "points/7/_label",
    "points/7/_points"
]

# Output Data format:
# col1 : No Name, col2 bis col16: 8 Keypoints, each occupying 2 cols for x and y
column_format = [None, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']

train_dataset = []
valid_dataset = [] 

for folders in os.listdir(MAIN_DIRECTORY):

    folder_path = MAIN_DIRECTORY + f'/{folders}'
    image_directory = folders + f'/images/'
    
    anno_data = pd.read_csv(f'{folder_path}/annotations.csv')
    anno_1 = anno_data[required_cols]

    new_dataset = []
    # Iterating over all rows

    #print(list(anno_data.columns))
    for index, row in anno_1.iterrows():
        # Only including images of sizes IMG_SIZExIMG_SIZE. Ignoring all the other images
        if (row['_width'] == IMG_SIZE) and (row['_height'] == IMG_SIZE):
            if isinstance(row['points/0/_label'], str):
                new_row = ['LOL', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                for nummer in range(8):
                    col_name = 'points/' + str(nummer) + '/_label'
                    points_name = 'points/' + str(nummer) + '/_points'
                    if isinstance(row[col_name], str):
                        
                        Label_Nr = int(row[col_name][1])
                        try:
                            x, y = row[points_name].split(',')
                        except ValueError:
                            pass
                        name_img = image_directory + row['_name']

                        new_row[0] = name_img
                        index = Label_Nr * 2 - 1

                        x = int(round(float(x)))
                        y = int(round(float(y)))

                        # To filter out when all the keypoints are not present
                        if x == 0 or y == 0:
                            x = np.nan
                            y = np.nan

                        new_row[index] = x
                        new_row[index + 1] = y

                new_dataset.append(new_row)
            else:
                new_row = [image_directory + row['_name'], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
                new_dataset.append(new_row)
            
    train, val = train_val_split(new_dataset)

    train_dataset.extend(train)
    valid_dataset.extend(val)

# final_dataset = pd.DataFrame(new_dataset, columns=column_format)
# final_dataset.to_csv('FinalAnnotations.csv')

train_dataFrame = pd.DataFrame(train_dataset, columns=column_format)
valid_dataFrame = pd.DataFrame(valid_dataset, columns=column_format)

train_dataFrame.to_csv('Train.csv', index=False)
valid_dataFrame.to_csv('Validation.csv', index=False)

print(train_dataFrame.head())
print(valid_dataFrame.head())
