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

from model import KeypointCustom
from dataset import train_data, train_loader, valid_data, valid_loader
from tqdm import tqdm

matplotlib.style.use('ggplot')

# training function
def fit(model, dataloader, data):
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0
    
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    counter = 1
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
        
    train_loss = train_running_loss/counter
    return train_loss

# validatioon function
def validate(model, dataloader, data, epoch):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    counter = 1
    # calculate the number of batches
    num_batches = int(len(data)/dataloader.batch_size)
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)
            outputs = model(image)
            loss = criterion(outputs, keypoints)
            if not loss.isnan():
                valid_running_loss += loss.detach().cpu().numpy()
            # plot the predicted validation keypoints after every...
            # ... predefined number of epochs
            if (epoch+1) % 1 == 0 and i == 0:
                utils.valid_keypoints_plot(image, outputs, keypoints, epoch)
        
    valid_loss = valid_running_loss/counter
    return valid_loss

# model 
#model = KeypointResNet(pretrained=True, requires_grad=True, model_name=config.RESNET_MODEL).to(config.DEVICE)
#model = KeypointEfficientNet(pretrained=True, requires_grad=True)
model = KeypointCustom(isPretrained=False, requires_grad=True, model_name='tf_efficientnet_lite0')
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