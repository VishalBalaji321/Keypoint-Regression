import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import config
from utils import save_model, accuracy, valid_keypoints_plot, plot_comparison_graph
import time
import statistics as s
import os
import torchmetrics as tm
import csv
from torch.utils.data import DataLoader
from inference import InferDataloader

from model import KeypointCustom
from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm

matplotlib.style.use('ggplot')


# training function
def fit(model, dataloader, data, TrainMetrics):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_acc = 0.0
    counter = 1

    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    
    #Creating a gradScaler at the beginning of the training
    scaler = GradScaler()
    
    for _, data in tqdm(enumerate(dataloader), total=num_batches):
        
        image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)
        optimizer.zero_grad()
        with autocast():
            outputs = model(image)
            loss = criterion(outputs, keypoints)
        
        if not loss.isnan():
            train_running_loss += loss.detach().cpu().numpy()
            counter += 1
        scaler.scale(loss).backward()
        #loss.backward()
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()

        outputs = outputs.float().cpu()
        keypoints = keypoints.float().cpu()

        train_acc += accuracy(outputs, keypoints)
        # Torchmetrics
        TrainMetrics['r2'](outputs, keypoints)
        TrainMetrics["meanSquared"](outputs, keypoints)
        TrainMetrics["meanSquaredLog"](outputs, keypoints)
        TrainMetrics["meanAbsolute"](outputs, keypoints)
        
    train_loss = round((train_running_loss / counter), 4)
    train_acc = round((train_acc / counter).item(), 3)
    # Torchmetrics
    train_r2 = round((TrainMetrics['r2'].compute()).item(), 3)
    train_ms = round((TrainMetrics['meanSquared'].compute()).item(), 3)
    train_msl = round((TrainMetrics['meanSquaredLog'].compute()).item(), 3)
    train_ma = round((TrainMetrics['meanAbsolute'].compute()).item(), 3)
    
    #Reset the torchmetrics
    for keys in TrainMetrics:
        TrainMetrics[keys].reset()
    
    return (train_loss, train_acc, train_r2, train_ms, train_msl, train_ma)

# validatioon function
def validate(model, dataloader, data, epoch, ValidMetrics):
    print('Validating')
    model.eval()

    valid_running_loss = 0.0
    counter = 1
    valid_acc = 0.0

    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            image = data['image'].to(config.DEVICE)
            keypoints = data['keypoints'].to(config.DEVICE)
            
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            if not loss.isnan():
                valid_running_loss += loss.detach().cpu().numpy()
                counter += 1

            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            valid_keypoints_plot(image, outputs, keypoints, epoch)
            
            # Accuracy
            outputs = outputs.float().cpu()
            keypoints = keypoints.float().cpu()
            
            valid_acc += accuracy(outputs, keypoints)
            # Torchmetrics
            ValidMetrics['r2'](outputs, keypoints)
            ValidMetrics["meanSquared"](outputs, keypoints)
            ValidMetrics["meanSquaredLog"](outputs, keypoints)
            ValidMetrics["meanAbsolute"](outputs, keypoints)
        
    valid_loss = round(valid_running_loss / counter, 4)
    valid_acc = round((valid_acc / counter).item(), 3)

    # Torchmetrics
    valid_r2 = round((ValidMetrics['r2'].compute()).item(), 3)
    valid_ms = round((ValidMetrics['meanSquared'].compute()).item(), 3)
    valid_msl = round((ValidMetrics['meanSquaredLog'].compute()).item(), 3)
    valid_ma = round((ValidMetrics['meanAbsolute'].compute()).item(), 3)
    
    #Reset the torchmetrics
    for keys in ValidMetrics:
        ValidMetrics[keys].reset()

    return (valid_loss, valid_acc, valid_r2, valid_ms, valid_msl, valid_ma)


# TorchMetrics
model_metrics = {
    "r2": tm.R2Score(num_outputs=16),
    "meanSquared": tm.MeanSquaredError(),
    "meanSquaredLog": tm.MeanSquaredLogError(),
    "meanAbsolute": tm.MeanAbsoluteError(),
}

final_model_summary_list = []

for modelName in config.models_to_evaluate:
    config.CURRENT_MODEL = modelName
    print(f"\nTraining Model: {config.CURRENT_MODEL}")
    
    # Loading the model
    model = KeypointCustom(isPretrained=False, requires_grad=True, fineTuning=False, model_name=config.CURRENT_MODEL)
    model = model.return_loaded_model().to(config.DEVICE)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # we need a loss function which is good for regression like SmmothL1Loss ...
    # ... or MSELoss -> Use this when working with GrayScale Images
    criterion = nn.SmoothL1Loss()

    # Creating the required folder directory
    model_directory = f'{config.OUTPUT_PATH}/{config.CURRENT_MODEL}'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(f'{model_directory}/weights'):
        os.makedirs(f'{model_directory}/weights')
    if not os.path.exists(f'{model_directory}/validation'):
        os.makedirs(f'{model_directory}/validation')
    if not os.path.exists(f'{model_directory}/inference_fp16_batch_size_1'):
        os.makedirs(f'{model_directory}/inference_fp16_batch_size_1')

    # These lists will be holding the model metrics which will later be used for analysis
    train_loss = []
    train_acc = []
    train_r2 = []
    train_ms = []
    train_msl = []
    train_ma = []

    val_loss = []
    val_acc = []
    val_r2 = []
    val_ms = []
    val_msl = []
    val_ma = []

    epoch_train_time = []
    val_time = []
    start_train = time.time()

    for epoch in range(config.EPOCHS):
        # Reducing the learning rate after a said number of epochs
        if epoch + 1 == config.DECAY_EPOCH:
            for _ in optimizer.param_groups:
                _['lr'] = config.DECAY_LR

        print(f"Epoch {epoch+1} of {config.EPOCHS} (model: {config.CURRENT_MODEL})")
        
        start_epoch = time.time()
        train_data = fit(model, train_loader, train_data, TrainMetrics=model_metrics)
        end_epoch = time.time()
        epoch_train_time.append(end_epoch - start_epoch)

        valid_data = validate(model, valid_loader, valid_data, epoch, ValidMetrics=model_metrics)
        end_val = time.time()
        val_time.append(end_val - end_epoch)
        
        # Logging the train and validation parameters
        train_loss.append(train_data[0])
        train_acc.append(train_data[1])
        train_r2.append(train_data[2])
        train_ms.append(train_data[3])
        train_msl.append(train_data[4])
        train_ma.append(train_data[5])

        val_loss.append(valid_data[0])
        val_acc.append(valid_data[1])
        val_r2.append(valid_data[2])
        val_ms.append(valid_data[3])
        val_msl.append(valid_data[4])
        val_ma.append(valid_data[5])

        print(f"Train:\nLoss: {train_data[0]}, Accuracy: {train_data[1]}, R2: {train_data[2]}, MeanSquared: {train_data[3]}, MeanSquaredLog: {train_data[4]}, MeanAbsolute: {train_data[5]}")
        print(f"Valid:\nLoss: {valid_data[0]}, Accuracy: {valid_data[1]}, R2: {valid_data[2]}, MeanSquared: {valid_data[3]}, MeanSquaredLog: {valid_data[4]}, MeanAbsolute: {valid_data[5]}\n")
    
        save_model(val_loss, (model, optimizer, criterion, epoch))

    end_train = time.time()

    total_train_time = round((end_train - start_train)/ 60)
    avg_epoch_train_time = round(s.mean(epoch_train_time), 2)
    avg_epoch_valid_time = round(s.mean(val_time), 2)

    print(f"Total Training Time: {total_train_time} min")
    print(f"Average train time per epoch: {avg_epoch_train_time}s")
    print(f"Average validation time per epoch: {avg_epoch_valid_time}s")

    # This plots graphs in the model directory between training and validation for different metrics
    plot_comparison_graph(train_loss, val_loss, "Training loss", "Validation loss", "Loss")
    plot_comparison_graph(train_acc, val_acc, "Training accuracy", "Validation accuracy", "Accuracy")
    plot_comparison_graph(train_r2, val_r2, "Training R2", "Validation R2", "R2 Score")
    plot_comparison_graph(train_ms, val_ms, "Training Error", "Validation Error", "Mean Squared Error")
    plot_comparison_graph(train_msl, val_msl, "Training Error", "Validation Error", "Mean Squared Log Error")
    plot_comparison_graph(train_ma, val_ma, "Training Error", "Validation Error", "Mean Absolute Error")

    # Testing inference with different batch sizes. Not meant for checking accuracy
    inference_avg_fps = []
    for infer_batch_size in config.INFER_BATCH_SIZES:
        inference_avg_fps.append(InferDataloader(model, infer_batch_size))
        
    # Exporting all the metrics into a dict file which will later be parsed into a .csv file
    model_summary = {
        'Model name': config.CURRENT_MODEL,
        'Final validation accuracy': val_acc[-1],
        'Final training accuracy': train_acc[-1],
        
        'Total training time(min)': total_train_time,
        'Average Epoch training time(s)': avg_epoch_train_time,
        'Average Epoch validation time(s)': avg_epoch_valid_time,
        
        'Inference_BATCH_SIZE_1': inference_avg_fps[0],
        'Inference_BATCH_SIZE_2': inference_avg_fps[1],
        'Inference_BATCH_SIZE_4': inference_avg_fps[2],
        'Inference_BATCH_SIZE_8': inference_avg_fps[3],
        'Inference_BATCH_SIZE_16': inference_avg_fps[4],
        'Inference_BATCH_SIZE_24': inference_avg_fps[5],
        'Inference_BATCH_SIZE_32': inference_avg_fps[6],

        'Final validation loss': val_loss[-1],
        'Final training loss': train_loss[-1],
        'Final validation R2': val_r2[-1],
        'Final training R2': train_r2[-1],
        'Final validation MeanSquared Error': val_ms[-1],
        'Final training MeanSquared Error': train_ms[-1],
        'Final validation MeanSquaredLog Error': val_msl[-1],
        'Final training MeanSquaredLog Error': train_msl[-1],
        'Final validation MeanAbsolute Error': val_ma[-1],
        'Final training MeanAbsolute Error': train_ma[-1]
    }

    # Dumping all the other metrics into the .csv file for future use
    for num, data in enumerate(train_acc):
        model_summary[f'train_acc_{num + 1}'] = data

    for num, data in enumerate(val_acc):
        model_summary[f'val_acc_{num + 1}'] = data

    for num, data in enumerate(train_loss):
        model_summary[f'train_loss_{num + 1}'] = data

    for num, data in enumerate(val_loss):
        model_summary[f'val_loss_{num + 1}'] = data

    for num, data in enumerate(train_r2):
        model_summary[f'train_r2_{num + 1}'] = data

    for num, data in enumerate(val_r2):
        model_summary[f'val_r2_{num + 1}'] = data

    for num, data in enumerate(train_ms):
        model_summary[f'train_ms_{num + 1}'] = data

    for num, data in enumerate(val_ms):
        model_summary[f'val_ms_{num + 1}'] = data

    for num, data in enumerate(train_msl):
        model_summary[f'train_msl_{num + 1}'] = data

    for num, data in enumerate(val_msl):
        model_summary[f'val_msl_{num + 1}'] = data

    for num, data in enumerate(train_ma):
        model_summary[f'train_ma_{num + 1}'] = data

    for num, data in enumerate(val_ma):
        model_summary[f'val_ma_{num + 1}'] = data

    final_model_summary_list.append(model_summary)

print('DONE TRAINING !! Now saving all the analysis.....')

keys = final_model_summary_list[0].keys()
with open(f'{config.OUTPUT_PATH}/model_analysis.csv', 'w', newline='')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(final_model_summary_list)

print(f"Finished !! Model Analysis summary file stored at: {config.OUTPUT_PATH}/model_analysis.csv")