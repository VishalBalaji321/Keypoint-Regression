# import pandas as pd
# import csv


# data_list = []

# data_1 = {
#     'Model Name': "EffNetV2",
#     'train_loss_1': 8.345,
#     'val_loss_1': 13.2342,
#     'final_accuracy': 90.45
# }
# data_list.append(data_1)

# data_2 = {
#     'Model Name': "MobileNet",
#     'train_loss_1': 18.35,
#     'val_loss_1': 23.42,
#     'final_accuracy': 88.75
# }
# data_list.append(data_2)


# keys = data_list[0].keys()
# with open('trial.csv', 'w', newline='')  as output_file:
#     dict_writer = csv.DictWriter(output_file, keys)
#     dict_writer.writeheader()
#     dict_writer.writerows(data_list)

# for index, item in enumerate(data_list):
#     print(index, item)


# !!! Validating Plot for the graphs

# import matplotlib.pyplot as plt
# import numpy as np
# import config
# import random

# def plot_comparison_graph(data):
    
#     # Predetermining the plot parameters to set the fonts
#     SMALL_SIZE = 13
#     MEDIUM_SIZE = 18
#     BIGGER_SIZE = 30

#     plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#     plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
#     plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#     plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#     plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#     plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#     plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#     n_rows = 2
#     n_cols = 4
#     fig, axs = plt.subplots(n_rows, n_cols, figsize=(45, 20))
    
#     fig.suptitle(config.CURRENT_MODEL)
#     for ax, data_points in zip(axs.flat, data):
        
#         if data_points is not None:
#             if len(data_points) == 3:
#                 colors = ['red', 'blue', 'orange', 'green', 'cyan', 'purple', 'black']
                
#                 for batchSize, Color, fps_points in zip(config.INFER_BATCH_SIZES, colors, data_points[0]):
#                     ax.plot(batchSize, fps_points, marker='o', markerfacecolor=Color, markeredgecolor='black', markersize=15, label=f'BATCH SIZE: {batchSize}')
#                     ax.text(batchSize - 1, fps_points - 7, str(fps_points))
#                 ax.set(xlabel='Batch size', ylabel=data_points[1], title=data_points[2])
#                 ax.legend(loc='lower right')

#             else:
#                 #plt.figure(figsize=(10, 7))
#                 ax.plot(data_points[0][0], color='orange', label=data_points[1])
#                 ax.plot(data_points[0][1], color='red', label=data_points[2])
#                 ax.set(xlabel='Epochs', ylabel=data_points[3], title=data_points[3])
                

#                 if data_points[4] == 'min':
#                     min_value_1 = min(data_points[0][0])
#                     index_min_1 = data_points[0][0].index(min_value_1)
#                     #ax.annotate(str(min_value_1), xy=(index_min_1, min_value_1), color='black', ms=20)

#                     ax.plot(index_min_1, min_value_1, marker='o', markerfacecolor='orange', markeredgecolor='black')
#                     ax.text(index_min_1+1, min_value_1+1, str(min_value_1))

#                     min_value_2 = min(data_points[0][1])
#                     index_min_2 = data_points[0][1].index(min_value_2)
#                     #ax.annotate(str(min_value_2), xy=(index_min_2, min_value_2), color='black', ms=20)
#                     ax.plot(index_min_2, min_value_2, marker='o', markerfacecolor='red', markeredgecolor='black')
#                     ax.text(index_min_2+1, min_value_2+1, str(min_value_2))

#                     ax.legend(loc='upper right')
                
#                 if data_points[4] == 'max':
#                     min_value_1 = max(data_points[0][0])
#                     index_min_1 = data_points[0][0].index(min_value_1)
#                     #ax.annotate(str(min_value_1), xy=(index_min_1, min_value_1), color='black', ms=20)
#                     ax.plot(index_min_1, min_value_1, marker='o', markerfacecolor='orange', markeredgecolor='black')
#                     ax.text(index_min_1+1, min_value_1+1, str(min_value_1))

#                     min_value_2 = max(data_points[0][1])
#                     index_min_2 = data_points[0][1].index(min_value_2)
#                     # ax.annotate(str(min_value_2), xy=(index_min_2, min_value_2), color='black', ms=20)
#                     ax.plot(index_min_2, min_value_2, marker='o', markerfacecolor='red', markeredgecolor='black')
#                     ax.text(index_min_2+1, min_value_2+1, str(min_value_2))

#                     ax.legend(loc='lower right')
            
#     fig.delaxes(axs[1, 3])
    
#     plt.savefig(f"model_analysis.png")


# data_1 = list(np.random.randint(0, 100, size=50))
# data_2 = list(np.random.randint(0, 100, size=50))

# data_3 = list(np.random.randint(0, 100, size=50))
# data_4 = list(np.random.randint(0, 100, size=50))

# data_5  = list(np.random.randint(2, 200, size=7))

# plot_comparison_graph((
#     ((data_1, data_2), "training accuracy", "validation accuracy", "Accuracy", 'max'), 
#     ((data_3, data_4), "training loss", "validation loss", "Loss", 'min'),
#     ((data_1, data_2), "training accuracy", "validation accuracy", "Accuracy", 'max'), 
#     ((data_3, data_4), "training loss", "validation loss", "Loss", 'min'),
#     (data_5, "inference fps", "Inference speed"),
#     ((data_1, data_2), "training accuracy", "validation accuracy", "Accuracy", 'max'), 
#     ((data_3, data_4), "training loss", "validation loss", "Loss", 'min'),
# ))


import torch
from model import KeypointCustom
import config
from inference import InferDataloader
from tqdm import tqdm

model = KeypointCustom(isPretrained=False, requires_grad=False, model_name='tf_efficientnet_lite0')
model = model.return_loaded_model().to(config.DEVICE)

#checkpoint = torch.load('../weights/resnet18_30_epochs.pth', map_location=torch.device('cpu'))
#checkpoint = torch.load('../weights/efficientNet-b2_full_25_epochs.pth', map_location=torch.device('cpu'))
weights_path = r'C:\Users\visha\Desktop\Schanzer Racing\keypoint_regression\Keypoint-Regression\weights\efficientNet_lite_0_50epochs.pth'

checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
#model.cuda()
#model.half()

inference_avg_fps = []
for infer_batch_size in [1]:
  inference_avg_fps.append(InferDataloader(model, infer_batch_size))

print(inference_avg_fps)
