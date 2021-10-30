import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import config
import utils
import time
import statistics as s
# import torchmetrics as tm

from model import KeypointCustom
from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm

matplotlib.style.use('ggplot')


# training function
def fit(model, dataloader, data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_acc = 0.0
    train_prec = 0.0
    train_recall = 0.0
    train_f1 = 0.0
    
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    counter = 1
    num_keypoints = 0
    #Creating a gradScaler at the beginning of the training
    scaler = GradScaler()
    
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        
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

        # Accuracy
        num_keypoints += keypoints.size(0)
        train_acc += ((100 * (keypoints == outputs.round()).sum()) / num_keypoints).item()

        # Precision, Recall, F1_Score
        tp = ((keypoints * outputs.round()).sum() / num_keypoints).item()
        fp = ((keypoints * (1 - outputs.round())).sum() / num_keypoints).item()
        fn = (((1 - keypoints) * outputs.round()).sum() / num_keypoints).item()

        precision = tp / (tp + fp + 1e-7)
        train_prec += precision
        
        recall = tp / (tp + fn + 1e-7)
        train_recall += recall
        
        train_f1 += (2 * precision * recall) / (precision + recall)

        # Torchmetrics
        # TrainMetrics['accuracy'](outputs, keypoints)
        # TrainMetrics["area_under_curve"](outputs, keypoints)
        # TrainMetrics["precision"](outputs, keypoints)
        # TrainMetrics['recall'](outputs, keypoints)
        # TrainMetrics["f1_score"](outputs, keypoints)


    train_loss = train_running_loss / counter
    train_acc = train_acc / counter
    train_recall = train_recall / counter
    train_prec = train_prec / counter
    train_f1 = train_f1 / counter

    # Torchmetrics
    # train_acc = TrainMetrics['accuracy'].compute()
    # train_auc = TrainMetrics["area_under_curve"].compute()
    # train_prec = TrainMetrics["precision"].compute()
    # train_recall = TrainMetrics['recall'].compute()
    # train_f1 = TrainMetrics["f1_score"].compute()

    print(f"Accuracy: {round(train_acc, 2)}, Precision: {round(train_prec, 2)}, Recall: {round(train_recall, 2)}, F1 Score: {round(train_f1, 2)}")
    
    # Reset the torchmetrics
    # for keys in TrainMetrics:
    #     TrainMetrics[keys].reset()
    
    return train_loss

# validatioon function
def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    model.cuda()
    model.half()

    valid_running_loss = 0.0
    counter = 1
    
    valid_acc = 0.0
    valid_prec = 0.0
    valid_recall = 0.0
    valid_f1 = 0.0

    num_keypoints = 0

    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):

            image = data['image'].to(config.DEVICE)
            image = image.half() 

            keypoints = data['keypoints'].to(config.DEVICE)
            keypoints = keypoints.half()
            
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            if not loss.isnan():
                valid_running_loss += loss.detach().cpu().numpy()
                counter += 1
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if (epoch+1) % 1 == 0 and i == 0:
                utils.valid_keypoints_plot(image, outputs, keypoints, epoch)
            # Accuracy
            num_keypoints += keypoints.size(0)
            valid_acc += ((100 * (keypoints == outputs.round()).sum()) / num_keypoints).item()

            # Precision, Recall, F1_Score
            tp = ((keypoints * outputs.round()).sum() / num_keypoints).item()
            fp = ((keypoints * (1 - outputs.round())).sum() / num_keypoints).item()
            fn = (((1 - keypoints) * outputs.round()).sum() / num_keypoints).item()

            precision = tp / (tp + fp + 1e-7)
            valid_prec += precision
            
            recall = tp / (tp + fn + 1e-7)
            valid_recall += recall
            
            valid_f1 += (2 * precision * recall) / (precision + recall)
    
    valid_loss = valid_running_loss / counter
    valid_acc = valid_acc / counter
    valid_recall = valid_recall / counter
    valid_prec = valid_prec / counter
    valid_f1 = valid_f1 / counter

    print(f"Accuracy: {round(valid_acc, 2)}, Precision: {round(valid_prec, 2)}, Recall: {round(valid_recall, 2)}, F1 Score: {round(valid_f1, 2)}")
    
    valid_loss = valid_running_loss/counter
    return valid_loss


# TorchMetrics -> Neat package for metrics but some weird errors for keypoints. Hence avoided

# train_metrics = {
#     "accuracy": tm.Accuracy(),
#     "area_under_curve": tm.AUC(),
#     "precision": tm.Precision(),
#     "recall": tm.Recall(),
#     "f1_score": tm.F1(),
# }

# valid_metrics = {
#     "accuracy": tm.Accuracy(),
#     "area_under_curve": tm.AUC(),
#     "precision": tm.Precision(),
#     "recall": tm.Recall(),
#     "f1_score": tm.F1(),
# }


# model 
#model = KeypointResNet(pretrained=True, requires_grad=True, model_name=config.RESNET_MODEL).to(config.DEVICE)
#model = KeypointEfficientNet(pretrained=True, requires_grad=True)
model = KeypointCustom(isPretrained=False, requires_grad=True, model_name=config.DEFAULT_MODEL)
model = model.return_loaded_model().to(config.DEVICE)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=config.LR)

# we need a loss function which is good for regression like SmmothL1Loss ...
# ... or MSELoss -> Use this when working with GrayScale Images
criterion = nn.SmoothL1Loss()

train_loss = []
val_loss = []

epoch_train_time = []
val_time = []
start_train = time.time()
for epoch in range(config.EPOCHS):
    print(f"Epoch {epoch+1} of {config.EPOCHS}")
    
    start_epoch = time.time()
    train_epoch_loss = fit(model, train_loader, train_data)
    end_epoch = time.time()
    epoch_train_time.append(end_epoch - start_epoch)

    val_epoch_loss = validate(model, valid_loader, valid_data, epoch)
    end_val = time.time()
    val_time.append(end_val - end_epoch)

    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {val_epoch_loss:.4f}')
    if (epoch % 5 == 0):
        torch.save({
            'epoch': config.EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{config.OUTPUT_PATH}/model_{epoch}.pth")
end_train = time.time()

print(f"Training Time: {end_train - start_train}")
print(f"Average train time per epoch: {s.mean(epoch_train_time)}")
print(f"Average inference time per image: {s.mean(val_time)}")
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.figtext(0.5, 0.01, f"Total training Time: {round(end_train - start_train, 2)}s", ha="center", fontsize=10)
plt.figtext(0.5, 0.11, f"Average train time per epoch: {round(s.mean(epoch_train_time), 2)}s", ha="center", fontsize=10)
plt.figtext(0.5, 0.21, f"Average inference time per image: {round(s.mean(val_time), 2)}s", ha="center", fontsize=10)


plt.savefig(f"{config.OUTPUT_PATH}/loss.png")
plt.show()
torch.save({
            'epoch': config.EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, f"{config.OUTPUT_PATH}/model_final.pth")
print('DONE TRAINING')