import pandas as pd
import csv


data_list = []

data_1 = {
    'Model Name': "EffNetV2",
    'train_loss_1': 8.345,
    'val_loss_1': 13.2342,
    'final_accuracy': 90.45
}
data_list.append(data_1)

data_2 = {
    'Model Name': "MobileNet",
    'train_loss_1': 18.35,
    'val_loss_1': 23.42,
    'final_accuracy': 88.75
}
data_list.append(data_2)


keys = data_list[0].keys()
with open('trial.csv', 'w', newline='')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data_list)

for index, item in enumerate(data_list):
    print(index, item)